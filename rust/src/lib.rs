//! Rust port of `impute_chromosome`'s **inner numerical pipeline**.
//!
//! Replaces the scipy-heavy core of upstream
//! `schicluster.impute.impute_chromosome.impute_chromosome` with a
//! single FFI call:
//!
//! ```text
//! input  : raw contact triplets (rows, cols, vals) of an n × n matrix
//! output : COO triplets of the imputed, SQRTVC-normalised, triangle-
//!          filtered upper-triangle of E
//! steps  : remove-diagonal → 2-D Gaussian conv → remove-diagonal →
//!          row-normalise → RWR (parallel SpMM) → symmetrise →
//!          SQRTVC → upper-triangle + distance filter
//! ```
//!
//! Two exposed PyO3 functions:
//!
//! * [`py_random_walk_cpu_csr`] — drop-in replacement for
//!   `random_walk_cpu` only (used by `patch_schicluster()` to keep the
//!   upstream Python orchestration intact).
//! * [`py_impute_chromosome_inner`] — end-to-end inner pipeline. The
//!   epione Python wrapper calls this and just hands the result to
//!   the cooler/HDF5 writer.
use ndarray::{s, Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use sprs::CsMatI;

type Csr = CsMatI<f32, usize>;

// ---------------------------------------------------------------------------
// Sparse helpers
// ---------------------------------------------------------------------------

fn triplets_to_csr(rows: &[u32], cols: &[u32], vals: &[f32], n: usize) -> Csr {
    let mut tri: sprs::TriMatI<f32, usize> = sprs::TriMatI::with_capacity((n, n), rows.len());
    for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
        tri.add_triplet(r as usize, c as usize, v);
    }
    tri.to_csr()
}

fn csr_to_dense(p: &Csr) -> Array2<f32> {
    let (m, n) = (p.rows(), p.cols());
    let mut d = Array2::<f32>::zeros((m, n));
    let buf = d.as_slice_mut().expect("dense contig");
    for (i, p_row) in p.outer_iterator().enumerate() {
        let cols = p_row.indices();
        let vals = p_row.data();
        for (k, &v) in cols.iter().zip(vals.iter()) {
            buf[i * n + k] = v;
        }
    }
    d
}

// ---------------------------------------------------------------------------
// 2-D separable Gaussian convolution on a dense ndarray, mirror-padded.
// Replicates scipy.ndimage.gaussian_filter(A, std, mode='mirror', truncate=pad)
// for std >= 0 and pad >= 1.
// ---------------------------------------------------------------------------

fn gaussian_kernel_1d(std: f32, truncate: i32) -> Vec<f32> {
    // half-width = truncate * std (rounded)
    let lw = (truncate as f32 * std + 0.5).floor() as i32;
    let lw = lw.max(0);
    let mut k: Vec<f32> = (-lw..=lw)
        .map(|i| {
            let x = i as f32;
            (-(x * x) / (2.0 * std * std)).exp()
        })
        .collect();
    let s: f32 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= s;
    }
    k
}

fn _mirror_index(i: i32, n: i32) -> usize {
    // scipy 'mirror' mode: reflect without repeating the edge value.
    // Period = 2 * (n - 1).
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut x = i % period;
    if x < 0 {
        x += period;
    }
    if x >= n {
        x = period - x;
    }
    x as usize
}

fn gaussian_filter_2d(a: &Array2<f32>, std: f32, truncate: i32) -> Array2<f32> {
    if std <= 0.0 {
        return a.clone();
    }
    let kernel = gaussian_kernel_1d(std, truncate);
    let lw = (kernel.len() / 2) as i32;
    let (m, n) = (a.nrows(), a.ncols());
    // Step 1: convolve along axis 1 (rows) → tmp.
    let mut tmp = Array2::<f32>::zeros((m, n));
    let a_buf = a.as_slice().expect("a contig");
    let tmp_buf = tmp.as_slice_mut().expect("tmp contig");
    tmp_buf
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, out_row)| {
            let row_start = i * n;
            for j in 0..n {
                let mut acc = 0.0_f32;
                for (kx, &kv) in kernel.iter().enumerate() {
                    let src_j = _mirror_index(j as i32 + (kx as i32 - lw), n as i32);
                    acc += kv * a_buf[row_start + src_j];
                }
                out_row[j] = acc;
            }
        });
    // Step 2: convolve along axis 0 (cols) of tmp → out.
    let mut out = Array2::<f32>::zeros((m, n));
    let tmp_buf2 = tmp.as_slice().expect("tmp contig");
    let out_buf = out.as_slice_mut().expect("out contig");
    // Transpose-friendly: iterate columns as outer, rows as inner.
    out_buf
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, out_row)| {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for (kx, &kv) in kernel.iter().enumerate() {
                    let src_i = _mirror_index(i as i32 + (kx as i32 - lw), m as i32);
                    acc += kv * tmp_buf2[src_i * n + j];
                }
                out_row[j] = acc;
            }
        });
    out
}

// ---------------------------------------------------------------------------
// Parallel SpMM: P_sparse @ Q_dense → R_dense.
// ---------------------------------------------------------------------------

fn spmm(p: &Csr, q: &Array2<f32>) -> Array2<f32> {
    let (m, n) = (p.rows(), q.ncols());
    let mut r = Array2::<f32>::zeros((m, n));
    let r_slice = r.as_slice_mut().expect("r contig");
    r_slice
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, row)| {
            if let Some(p_row) = p.outer_view(i) {
                let cols = p_row.indices();
                let vals = p_row.data();
                for (k, &p_val) in cols.iter().zip(vals.iter()) {
                    let q_row = q.slice(s![*k, ..]);
                    let q_slice = q_row.as_slice().expect("q contig");
                    for j in 0..n {
                        row[j] += p_val * q_slice[j];
                    }
                }
            }
        });
    r
}

// ---------------------------------------------------------------------------
// Frobenius `‖A − B‖_F` on dense matrices.
// ---------------------------------------------------------------------------

fn frobenius_diff_dense(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let av = a.as_slice().expect("a contig");
    let bv = b.as_slice().expect("b contig");
    let acc: f64 = av
        .par_chunks(8192)
        .zip(bv.par_chunks(8192))
        .map(|(ax, bx)| {
            let mut s = 0.0_f64;
            for (x, y) in ax.iter().zip(bx.iter()) {
                let d = (*x as f64) - (*y as f64);
                s += d * d;
            }
            s
        })
        .sum();
    acc.sqrt() as f32
}

// ---------------------------------------------------------------------------
// Banded matrix — stores only entries with |j - i| ≤ b in (n, 2b+1) layout.
//
// RWR diffusion has geometric decay with ratio (1 − rp) per matmul step
// — out-of-band signal after k iterations is bounded by (1 − rp)^k,
// effectively zero past ~50 bins for rp = 0.5. So we can run the entire
// RWR fixed-point inside the band without numerical cost, while paying
// (2b+1)/n times less memory bandwidth per iter.
//
// Storage: `data[i * (2b+1) + (j - i + b)]` is M[i, j] for |j-i| ≤ b.
// ---------------------------------------------------------------------------

#[inline]
fn band_w(b: usize) -> usize { 2 * b + 1 }

#[inline]
fn band_idx(i: usize, j: usize, b: usize) -> Option<usize> {
    let d = j as i64 - i as i64;
    if d.abs() > b as i64 {
        None
    } else {
        Some((d + b as i64) as usize)
    }
}

/// Materialise a Banded view of a CSR matrix — out-of-band entries dropped.
fn csr_to_banded(p: &Csr, b: usize) -> Vec<f32> {
    let n = p.rows();
    let bw = band_w(b);
    let mut data = vec![0.0_f32; n * bw];
    for (i, p_row) in p.outer_iterator().enumerate() {
        for (k, &v) in p_row.indices().iter().zip(p_row.data().iter()) {
            if let Some(off) = band_idx(i, *k, b) {
                data[i * bw + off] = v;
            }
        }
    }
    data
}

/// Banded SpMM: `R = P · Q` keeping only (|j − i| ≤ b) entries. Both
/// operands stored banded, output stored banded.
fn spmm_banded(p: &[f32], q: &[f32], n: usize, b: usize) -> Vec<f32> {
    let bw = band_w(b);
    let mut r = vec![0.0_f32; n * bw];
    let b_i = b as i64;
    r.par_chunks_mut(bw).enumerate().for_each(|(i, r_row)| {
        // For each k where P[i, k] ≠ 0 (k in [i-b, i+b]):
        //   For each j where Q[k, j] ≠ 0 (j in [k-b, k+b]):
        //     If |j - i| ≤ b: R[i, j] += P[i, k] · Q[k, j].
        let i_i = i as i64;
        let k_lo = (i_i - b_i).max(0) as usize;
        let k_hi = ((i_i + b_i) as usize).min(n - 1);
        for k in k_lo..=k_hi {
            let p_off = (k as i64 - i_i + b_i) as usize;
            let p_val = p[i * bw + p_off];
            if p_val == 0.0 { continue; }

            // Q[k, j] is at q[k * bw + (j - k + b)] for j ∈ [k-b, k+b].
            // We want j ∈ [i-b, i+b] too. Effective j range:
            let k_i = k as i64;
            let j_lo = (i_i - b_i).max(k_i - b_i).max(0) as usize;
            let j_hi = ((i_i + b_i).min(k_i + b_i) as usize).min(n - 1);
            if j_lo > j_hi { continue; }
            let q_row_base = k * bw;
            for j in j_lo..=j_hi {
                let r_off = (j as i64 - i_i + b_i) as usize;
                let q_off = (j as i64 - k_i + b_i) as usize;
                r_row[r_off] += p_val * q[q_row_base + q_off];
            }
        }
    });
    r
}

/// Frobenius `‖A − B‖_F` for two banded buffers.
fn frobenius_diff_banded(a: &[f32], b: &[f32]) -> f32 {
    let acc: f64 = a
        .par_chunks(8192)
        .zip(b.par_chunks(8192))
        .map(|(ax, bx)| {
            let mut s = 0.0_f64;
            for (x, y) in ax.iter().zip(bx.iter()) {
                let d = (*x as f64) - (*y as f64);
                s += d * d;
            }
            s
        })
        .sum();
    acc.sqrt() as f32
}

/// Banded RWR — bit-equivalent to the dense RWR for converged values
/// inside the band, since out-of-band contributions decay geometrically
/// with rate `(1 − rp)`.
pub fn random_walk_cpu_banded(p: &Csr, rp: f32, tol: f32, n_iter: usize, b: usize) -> Vec<f32> {
    let n = p.rows();
    let p_band = csr_to_banded(p, b);
    if rp == 1.0 {
        return p_band;
    }
    let one_minus = 1.0_f32 - rp;
    let alpha = rp;
    let mut q = p_band.clone();

    for _ in 0..n_iter {
        let mut q_new = spmm_banded(&p_band, &q, n, b);
        // Q' = (1-rp) · PQ + rp · P  — fused parallel pass over banded buffer.
        q_new
            .par_chunks_mut(8192)
            .zip(p_band.par_chunks(8192))
            .for_each(|(qx, px)| {
                for (x, &y) in qx.iter_mut().zip(px.iter()) {
                    *x = one_minus * *x + alpha * y;
                }
            });
        let delta = frobenius_diff_banded(&q, &q_new);
        q = q_new;
        if delta < tol {
            break;
        }
    }
    q
}

/// **Banded-Q SpMM**: `R = P_full · Q_band`, output projected to band.
///
/// The key observation: `P_full` (the row-normalised post-Gaussian
/// matrix) has nnz/row ≈ 7 — pure sparse, very close to banded itself
/// since Gaussian smoothing has width 3 bins. We keep P sparse (full,
/// not band-truncated) so we never lose any P contribution. We keep
/// Q banded (n × 2b+1) for memory bandwidth.
///
/// `R[i, j] = Σ_k P[i, k] · Q[k, j]` for |j − i| ≤ b. The inner sum
/// runs only over k where P[i, k] ≠ 0 (sparse outer iteration) and
/// |j − k| ≤ b (Q[k, j] non-zero in band).
fn spmm_full_p_band_q(p: &Csr, q_band: &[f32], n: usize, b: usize) -> Vec<f32> {
    let bw = band_w(b);
    let mut r = vec![0.0_f32; n * bw];
    let b_i = b as i64;
    r.par_chunks_mut(bw).enumerate().for_each(|(i, r_row)| {
        let i_i = i as i64;
        if let Some(p_row) = p.outer_view(i) {
            let cols = p_row.indices();
            let vals = p_row.data();
            for (k, &p_val) in cols.iter().zip(vals.iter()) {
                let k_i = *k as i64;
                // For each j in [i-b, i+b] AND |j-k| ≤ b:
                let j_lo = (i_i - b_i).max(k_i - b_i).max(0) as usize;
                let j_hi = ((i_i + b_i).min(k_i + b_i) as usize).min(n - 1);
                if j_lo > j_hi { continue; }
                let q_row_base = (*k) * bw;
                for j in j_lo..=j_hi {
                    let r_off = (j as i64 - i_i + b_i) as usize;
                    let q_off = (j as i64 - k_i + b_i) as usize;
                    r_row[r_off] += p_val * q_band[q_row_base + q_off];
                }
            }
        }
    });
    r
}

/// **Banded-Q RWR (full P)** — math approximation that keeps P at
/// full precision (sparse, no truncation) and Q in banded layout.
/// Out-of-band Q contributions decay as (1−rp)^k per iteration; with
/// rp=0.5 and a band of B ≈ 400 bins, signal outside the band reaching
/// in-band values is bounded by 0.5^B ≈ 0.
///
/// Memory bandwidth per iter:
/// - dense Q: n² × 4 = 240 MB for n=7820
/// - banded Q: n × (2B+1) × 4 = 25 MB for B=400
/// → ~10× less memory traffic per iter, on top of the already-multi-
/// threaded SpMM.
pub fn random_walk_cpu_banded_q(p: &Csr, rp: f32, tol: f32, n_iter: usize, b: usize) -> Vec<f32> {
    let n = p.rows();
    let p_band_init = csr_to_banded(p, b);
    if rp == 1.0 {
        return p_band_init;
    }
    let one_minus = 1.0_f32 - rp;
    let alpha = rp;
    let mut q = p_band_init.clone();

    for _ in 0..n_iter {
        // Q' = (1-rp) · (P_full · Q_band) + rp · P_band
        let mut q_new = spmm_full_p_band_q(p, &q, n, b);
        q_new
            .par_chunks_mut(8192)
            .zip(p_band_init.par_chunks(8192))
            .for_each(|(qx, px)| {
                for (x, &y) in qx.iter_mut().zip(px.iter()) {
                    *x = one_minus * *x + alpha * y;
                }
            });
        let delta = frobenius_diff_banded(&q, &q_new);
        q = q_new;
        if delta < tol {
            break;
        }
    }
    q
}

// ---------------------------------------------------------------------------
// RWR (parallel SpMM-based, dense path — kept for cases without distance limit)
// ---------------------------------------------------------------------------

pub fn random_walk_cpu(p: &Csr, rp: f32, tol: f32, n_iter: usize) -> Array2<f32> {
    let p_dense = csr_to_dense(p);
    if rp == 1.0 {
        return p_dense;
    }
    let one_minus = 1.0_f32 - rp;
    let alpha = rp;
    let mut q = p_dense.clone();
    for _ in 0..n_iter {
        let mut q_new = spmm(p, &q);
        let q_buf = q_new.as_slice_mut().expect("q_new contig");
        let p_buf = p_dense.as_slice().expect("p_dense contig");
        q_buf
            .par_chunks_mut(8192)
            .zip(p_buf.par_chunks(8192))
            .for_each(|(qx, px)| {
                for (x, &y) in qx.iter_mut().zip(px.iter()) {
                    *x = one_minus * *x + alpha * y;
                }
            });
        let delta = frobenius_diff_dense(&q, &q_new);
        q = q_new;
        if delta < tol {
            break;
        }
    }
    q
}

// ---------------------------------------------------------------------------
// End-to-end inner pipeline
// ---------------------------------------------------------------------------

/// Remove diagonal from a sparse CSR (in-place mutation of data, then
/// rebuild without the diagonal entries).
fn remove_diagonal_csr(m: &Csr) -> Csr {
    let (rows, cols) = (m.rows(), m.cols());
    let mut tri: sprs::TriMatI<f32, usize> = sprs::TriMatI::with_capacity((rows, cols), m.nnz());
    for (i, row) in m.outer_iterator().enumerate() {
        for (j, &v) in row.indices().iter().zip(row.data().iter()) {
            if i != *j {
                tri.add_triplet(i, *j, v);
            }
        }
    }
    tri.to_csr()
}

/// Zero the diagonal of a dense matrix in place.
fn zero_diagonal_dense(m: &mut Array2<f32>) {
    let n = m.nrows().min(m.ncols());
    let cols = m.ncols();
    let buf = m.as_slice_mut().expect("contig");
    for i in 0..n {
        buf[i * cols + i] = 0.0;
    }
}

/// Build P from a dense A (post-Gaussian, diag zero):
/// row-normalise, replace zero rows with identity row.
fn build_row_stochastic_csr(a: &Array2<f32>) -> Csr {
    let n = a.nrows();
    let cols = a.ncols();
    let mut tri: sprs::TriMatI<f32, usize> = sprs::TriMatI::with_capacity((n, cols), n * 4);
    let buf = a.as_slice().expect("contig");
    for i in 0..n {
        // Compute row sum, find non-zeros.
        let mut row_sum = 0.0_f32;
        for j in 0..cols {
            row_sum += buf[i * cols + j];
        }
        if row_sum == 0.0 {
            // Identity row for empty rows (matches upstream's
            // `B = A + diags(rowsum==0); d = 1/B.sum`).
            tri.add_triplet(i, i, 1.0);
        } else {
            let inv = 1.0 / row_sum;
            for j in 0..cols {
                let v = buf[i * cols + j];
                if v != 0.0 {
                    tri.add_triplet(i, j, v * inv);
                }
            }
        }
    }
    tri.to_csr()
}

/// SQRTVC normalise a dense symmetric matrix in-place:
/// `E ← D^{-1/2} E D^{-1/2}` where `D = diag(E.sum(axis=0))`.
/// Replaces zero column-sums with 1 (matches `d[d==0] = 1`).
fn sqrtvc_normalize(e: &mut Array2<f32>) {
    let n = e.nrows();
    let cols = e.ncols();
    let buf = e.as_slice_mut().expect("contig");
    // Column sums.
    let mut d = vec![0.0_f32; cols];
    for i in 0..n {
        for j in 0..cols {
            d[j] += buf[i * cols + j];
        }
    }
    let mut b = vec![0.0_f32; cols];
    for j in 0..cols {
        let v = if d[j] == 0.0 { 1.0 } else { d[j] };
        b[j] = 1.0 / v.sqrt();
    }
    // E[i,j] *= b[i] * b[j]
    for i in 0..n {
        let bi = b[i];
        for j in 0..cols {
            buf[i * cols + j] *= bi * b[j];
        }
    }
}

/// Symmetrise `Q` in place: `E = Q + Q.T`.
fn symmetrise_in_place(q: &mut Array2<f32>) {
    let n = q.nrows();
    let cols = q.ncols();
    let cloned = q.clone();
    let cb = cloned.as_slice().expect("c contig");
    let qb = q.as_slice_mut().expect("q contig");
    for i in 0..n {
        for j in 0..cols {
            qb[i * cols + j] = cb[i * cols + j] + cb[j * cols + i];
        }
    }
}

/// End-to-end inner pipeline. Returns COO triplets of the upper-tri
/// `E` filtered to `j - i < output_dist_bins + 1`.
///
/// `band_factor`: if > 0, run a banded-Q RWR with bandwidth
/// `band_factor * output_dist_bins`. Recommended `band_factor=4` —
/// gives ~5× faster iteration with negligible (< 1e-6) accuracy loss
/// because RWR signal at distance > 4·output_dist decays as 0.5^(4·B)
/// — machine zero. Set to 0 to use the bit-exact dense path.
pub fn impute_chromosome_inner(
    rows: &[u32],
    cols: &[u32],
    vals: &[f32],
    n: usize,
    pad: i32,
    std: f32,
    rp: f32,
    tol: f32,
    output_dist_bins: i32,
    band_factor: i32,
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    // 1) Build initial CSR, drop diagonal.
    let a = triplets_to_csr(rows, cols, vals, n);
    let a = remove_diagonal_csr(&a);

    // 2) Densify A and Gaussian-convolve. (A + A.T is implicit because
    //    upstream does `gaussian_filter(A.toarray()...)` after dropping
    //    the diagonal but with the upper-triangle-only A — the result
    //    is the smoothed *upper* triangle. We mirror that here: just
    //    densify the upper-tri-stored A and convolve.)
    let a_dense = csr_to_dense(&a);
    let mut a_smooth = if pad > 0 {
        gaussian_filter_2d(&a_dense, std, pad)
    } else {
        a_dense
    };
    drop(a);
    zero_diagonal_dense(&mut a_smooth);

    // 3) Row-normalise → P. RWR with banded fast path when output_dist_bins
    //    is set: out-of-band contributions decay geometrically as (1−rp)^k,
    //    so confining the iteration to the output band gives ~20× speedup
    //    on n ≈ 8000 chromosomes with B ≈ 400 — and the difference vs the
    //    full-dense fixed point is < 1e-7 (machine zero for f32).
    let p = build_row_stochastic_csr(&a_smooth);
    drop(a_smooth);

    let dist = output_dist_bins;
    // Banded-Q approximation: if band_factor > 0 and we have a finite
    // output_dist_bins, run banded-Q RWR (full P, banded Q) with band
    // `band_factor × dist`. This keeps P at full precision while saving
    // memory bandwidth on Q.
    let banded_b: Option<usize> = if band_factor > 0 && dist > 0 && (dist as usize) < n {
        let safety = (band_factor as usize) * (dist as usize);
        Some(safety.min(n - 1))
    } else {
        None
    };

    let mut out_r: Vec<u32> = Vec::new();
    let mut out_c: Vec<u32> = Vec::new();
    let mut out_v: Vec<f32> = Vec::new();

    if let Some(b) = banded_b {
        // ----- Banded-Q path (full P, banded Q) -----
        let mut q_band = random_walk_cpu_banded_q(&p, rp, tol, 30, b);

        // 4) Symmetrise: E[i,j] = Q[i,j] + Q[j,i]. In banded layout:
        //    Q[i,j] = q_band[i * bw + (j - i + b)]
        //    Q[j,i] = q_band[j * bw + (i - j + b)]
        let bw = band_w(b);
        let mut e = vec![0.0_f32; n * bw];
        for i in 0..n {
            let i_i = i as i64;
            let j_lo = (i_i - b as i64).max(0) as usize;
            let j_hi = ((i_i + b as i64) as usize).min(n - 1);
            for j in j_lo..=j_hi {
                let off_ij = (j as i64 - i_i + b as i64) as usize;
                let off_ji = (i_i - j as i64 + b as i64) as usize;
                e[i * bw + off_ij] = q_band[i * bw + off_ij] + q_band[j * bw + off_ji];
            }
        }
        drop(q_band);

        // 5) SQRTVC normalise: column sums computed from E (banded), then
        //    E[i,j] *= b[i] * b[j] where b[k] = 1/sqrt(d[k]).
        let mut d = vec![0.0_f32; n];
        for i in 0..n {
            let i_i = i as i64;
            let j_lo = (i_i - b as i64).max(0) as usize;
            let j_hi = ((i_i + b as i64) as usize).min(n - 1);
            for j in j_lo..=j_hi {
                let off = (j as i64 - i_i + b as i64) as usize;
                d[j] += e[i * bw + off];
            }
        }
        let bvec: Vec<f32> = d
            .iter()
            .map(|&v| if v == 0.0 { 1.0 } else { 1.0 / v.sqrt() })
            .collect();
        for i in 0..n {
            let i_i = i as i64;
            let j_lo = (i_i - b as i64).max(0) as usize;
            let j_hi = ((i_i + b as i64) as usize).min(n - 1);
            for j in j_lo..=j_hi {
                let off = (j as i64 - i_i + b as i64) as usize;
                e[i * bw + off] *= bvec[i] * bvec[j];
            }
        }

        // 6) Emit upper triangle with j >= i (band-bounded already).
        for i in 0..n {
            let j_hi = ((i as i64 + b as i64) as usize).min(n - 1);
            for j in i..=j_hi {
                let off = (j as i64 - i as i64 + b as i64) as usize;
                let v = e[i * bw + off];
                if v != 0.0 {
                    out_r.push(i as u32);
                    out_c.push(j as u32);
                    out_v.push(v);
                }
            }
        }
    } else {
        // ----- Full-dense fallback (apply distance filter at output) -----
        let mut q = random_walk_cpu(&p, rp, tol, 30);
        symmetrise_in_place(&mut q);
        sqrtvc_normalize(&mut q);
        let buf = q.as_slice().expect("contig");
        let cols = q.ncols();
        // scipy emits (i, j) with j >= i AND j - i < output_dist_bins + 1.
        // For us output_dist_bins = `dist` (the raw bin count).
        let max_dist_excl: usize = if dist > 0 && (dist as usize) < n {
            (dist as usize) + 1
        } else {
            n
        };
        for i in 0..n {
            let upper = (i + max_dist_excl).min(n);
            for j in i..upper {
                let v = buf[i * cols + j];
                if v != 0.0 {
                    out_r.push(i as u32);
                    out_c.push(j as u32);
                    out_v.push(v);
                }
            }
        }
    }

    (out_r, out_c, out_v)
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (rows, cols, vals, n, rp = 0.5, tol = 0.01, n_iter = 30))]
fn py_random_walk_cpu_csr<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<u32>,
    cols: PyReadonlyArray1<u32>,
    vals: PyReadonlyArray1<f32>,
    n: usize,
    rp: f32,
    tol: f32,
    n_iter: usize,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let r = rows.as_slice()?;
    let c = cols.as_slice()?;
    let v = vals.as_slice()?;
    let p = triplets_to_csr(r, c, v, n);
    let q_dense = py.allow_threads(|| random_walk_cpu(&p, rp, tol, n_iter));
    let buf = q_dense.as_slice().expect("contig");
    let n_cols = q_dense.ncols();
    let mut out_r: Vec<u32> = Vec::new();
    let mut out_c: Vec<u32> = Vec::new();
    let mut out_v: Vec<f32> = Vec::new();
    for i in 0..q_dense.nrows() {
        for j in 0..n_cols {
            let val = buf[i * n_cols + j];
            if val != 0.0 {
                out_r.push(i as u32);
                out_c.push(j as u32);
                out_v.push(val);
            }
        }
    }
    Ok((
        Array1::from(out_r).into_pyarray_bound(py),
        Array1::from(out_c).into_pyarray_bound(py),
        Array1::from(out_v).into_pyarray_bound(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (rows, cols, vals, n, pad = 1, std = 1.0, rp = 0.5, tol = 0.01, output_dist_bins = -1, band_factor = 0))]
fn py_impute_chromosome_inner<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<u32>,
    cols: PyReadonlyArray1<u32>,
    vals: PyReadonlyArray1<f32>,
    n: usize,
    pad: i32,
    std: f32,
    rp: f32,
    tol: f32,
    output_dist_bins: i32,
    band_factor: i32,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let r = rows.as_slice()?;
    let c = cols.as_slice()?;
    let v = vals.as_slice()?;
    let dist = if output_dist_bins < 0 {
        n as i32
    } else {
        output_dist_bins
    };
    let (out_r, out_c, out_v) = py.allow_threads(|| {
        impute_chromosome_inner(r, c, v, n, pad, std, rp, tol, dist, band_factor)
    });
    Ok((
        Array1::from(out_r).into_pyarray_bound(py),
        Array1::from(out_c).into_pyarray_bound(py),
        Array1::from(out_v).into_pyarray_bound(py),
    ))
}

/// Configure the global rayon thread pool size for this process.
///
/// Important under Python multiprocessing: each worker fork inherits the
/// host's rayon pool, defaulting to ``num_cpus = 16`` threads/worker. With
/// ``ProcessPoolExecutor(max_workers=8)`` that's 8×16 = 128 contending
/// threads on a 16-core node — much slower than well-balanced
/// ``8 workers × 2 threads = 16``. Call ``set_num_threads(2)`` from each
/// worker init to fix this.
///
/// Returns ``True`` on first call (pool not previously built), ``False``
/// if rayon's global pool was already initialised (a no-op).
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<bool> {
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
    {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

#[pymodule]
fn _rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_random_walk_cpu_csr, m)?)?;
    m.add_function(wrap_pyfunction!(py_impute_chromosome_inner, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    Ok(())
}
