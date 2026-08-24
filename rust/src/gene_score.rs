//! Port of scHiCluster/schicluster/draft/gene_score.py's per-gene window-sum
//! loop, shared by both `gene_score_impute` and `gene_score_raw`.
//!
//! Upstream evaluates `D[r0:r1, c0:c1].sum()` once per gene. With ~78.7k genes
//! that is 78.7k scipy submatrix allocations per cell. Here the CSR is visited
//! in place: for each row in the window, the sorted column indices are binary
//! searched for the column range, and the matching values are gathered into a
//! row-major buffer and reduced.
//!
//! # Why the reduction has the shape it does
//!
//! `csr.sum(axis=None)` is **not** a flat `csr.data.sum()`. scipy computes it as
//! `(self @ np.ones(n_cols, dtype=res_dtype)).sum()` — a CSR matvec against a
//! ones vector, which accumulates each row **serially in stored column order**,
//! followed by `np.add.reduce` over the dense row-sums vector, which uses
//! **pairwise** summation (8-way unrolled base case, 128-element blocksize).
//! `res_dtype` is the matrix's own dtype for floats, so an f32 cool reduces
//! entirely in f32.
//!
//! That matters: imputed cools store `count` as f32, so upstream's own answer
//! carries ~3.8e-6 of f32 rounding on a window summing to ~48. A straight f64
//! accumulation here is *more* accurate than upstream and therefore lands
//! ~3.8e-6 away from it, blowing the pre-registered 1e-6 absolute gate. A flat
//! f32 pairwise sum over the window's values is also wrong — verified against
//! scipy, it disagrees on a 22-nnz window.
//!
//! Since the manifest is read-only and the protocol forbids widening a
//! threshold to make a port pass, the reduction below reproduces scipy's
//! two-stage algorithm exactly, in the source dtype. Verified bit-equal to
//! `W.sum()` on every fixture gene and on 200 randomised windows.
//!
//! Parallelism is across genes only; each gene's reduction is a fixed order, so
//! this is admissibility class (E).

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Resolve one Python slice bound against an axis of length `n`, exactly as
/// CPython (and therefore scipy's sparse `__getitem__`) does: negative values
/// are offset by `n`, then the result is clamped into `[0, n]`.
///
/// This is what makes `gene_score_impute`'s `(xx-1)` start behave as `n-1` when
/// `xx == 0`, producing an empty window and a score of 0.0.
fn resolve_bound(v: i64, n: usize) -> usize {
    let n_i = n as i64;
    let r = if v < 0 { v + n_i } else { v };
    if r < 0 {
        0
    } else if r > n_i {
        n
    } else {
        r as usize
    }
}

/// numpy's `pairwise_sum` for float32, transcribed from
/// numpy/_core/src/umath/loops_utils.h.src. Bit-exact against `ndarray.sum()`.
fn pairwise_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f32;
        for &v in a {
            res += v;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let tail = n - (n % 8);
        let mut i = 8;
        while i < tail {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[tail..] {
            res += v;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_f32(&a[..n2]) + pairwise_f32(&a[n2..])
}

/// numpy's `pairwise_sum` for float64. Same shape as [`pairwise_f32`].
fn pairwise_f64(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f64;
        for &v in a {
            res += v;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let tail = n - (n % 8);
        let mut i = 8;
        while i < tail {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[tail..] {
            res += v;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_f64(&a[..n2]) + pairwise_f64(&a[n2..])
}

/// Sum a CSR matrix over each of `n_genes` rectangular windows.
///
/// `input_f32` selects the accumulation dtype so the reduction matches whatever
/// dtype the upstream CSR carried. Values always arrive as f64 (lossless for an
/// f32 source) and are narrowed on gather when `input_f32` is set.
#[allow(clippy::too_many_arguments)]
pub fn window_sums(
    indptr: &[i64],
    indices: &[i64],
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    row_start: &[i64],
    row_end: &[i64],
    col_start: &[i64],
    col_end: &[i64],
    input_f32: bool,
) -> Vec<f64> {
    (0..row_start.len())
        .into_par_iter()
        .map(|g| {
            let r0 = resolve_bound(row_start[g], n_rows);
            let r1 = resolve_bound(row_end[g], n_rows);
            let c0 = resolve_bound(col_start[g], n_cols);
            let c1 = resolve_bound(col_end[g], n_cols);
            if r0 >= r1 || c0 >= c1 {
                return 0.0;
            }
            let c0 = c0 as i64;
            let c1 = c1 as i64;

            // Stage 1: the matvec against a ones vector — one serial,
            // left-to-right accumulation per row, in stored column order.
            // Rows with no in-window entry still contribute an explicit 0.0,
            // because the dense row-sums vector scipy builds has one slot per
            // row and that length drives stage 2's pairwise blocking.
            let nr = r1 - r0;
            let mut rs_f32: Vec<f32> = if input_f32 { Vec::with_capacity(nr) } else { Vec::new() };
            let mut rs_f64: Vec<f64> = if input_f32 { Vec::new() } else { Vec::with_capacity(nr) };
            for r in r0..r1 {
                let lo = indptr[r] as usize;
                let hi = indptr[r + 1] as usize;
                let row_cols = &indices[lo..hi];
                // scipy guarantees sorted indices on a canonical CSR; the
                // Python wrapper calls sort_indices() defensively.
                let a = row_cols.partition_point(|&c| c < c0);
                let b = row_cols.partition_point(|&c| c < c1);
                if input_f32 {
                    let mut acc = 0.0f32;
                    for k in (lo + a)..(lo + b) {
                        acc += data[k] as f32;
                    }
                    rs_f32.push(acc);
                } else {
                    let mut acc = 0.0f64;
                    for k in (lo + a)..(lo + b) {
                        acc += data[k];
                    }
                    rs_f64.push(acc);
                }
            }
            // Stage 2: np.add.reduce over the row-sums vector.
            if input_f32 {
                pairwise_f32(&rs_f32) as f64
            } else {
                pairwise_f64(&rs_f64)
            }
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_rows, n_cols, row_start, row_end, col_start, col_end, input_f32))]
#[allow(clippy::too_many_arguments)]
pub fn py_gene_score_chrom<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    n_rows: usize,
    n_cols: usize,
    row_start: PyReadonlyArray1<'py, i64>,
    row_end: PyReadonlyArray1<'py, i64>,
    col_start: PyReadonlyArray1<'py, i64>,
    col_end: PyReadonlyArray1<'py, i64>,
    input_f32: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let data = data.as_slice()?;
    let row_start = row_start.as_slice()?;
    let row_end = row_end.as_slice()?;
    let col_start = col_start.as_slice()?;
    let col_end = col_end.as_slice()?;

    if indptr.len() != n_rows + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "indptr length {} does not match n_rows + 1 = {}",
            indptr.len(),
            n_rows + 1
        )));
    }
    if indices.len() != data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "indices and data must have equal length",
        ));
    }
    let n_genes = row_start.len();
    if row_end.len() != n_genes || col_start.len() != n_genes || col_end.len() != n_genes {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "row_start / row_end / col_start / col_end must have equal length",
        ));
    }

    let out = py.allow_threads(|| {
        window_sums(
            indptr, indices, data, n_rows, n_cols, row_start, row_end, col_start, col_end,
            input_f32,
        )
    });
    Ok(Array1::from_vec(out).into_pyarray_bound(py))
}
