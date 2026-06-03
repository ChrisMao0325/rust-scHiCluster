//! Port of scHiCluster/schicluster/loop/loop_bkg.py::calculate_chrom_background_normalization
//! (excluding the shuffle=True stochastic branch — handled by the Python wrapper).
//!
//! Pipeline:
//!   1. densify upper-tri triplets -> Array2<f32>
//!   2. zero diagonal
//!   3. per-diagonal pctl-99 clip + scipy.stats.zscore (ddof=0) + clip(-cap, cap),
//!      writing the zscored values into positives and `min(zscored)` into non-positives
//!   4. donut-minus convolution -> T
//!   5. upper-tri mask T; apply min_cutoff |.| filter to both E and T
//!   6. emit E and T upper-tri triplets row-major (T_out = E_keep ? e : 0 - T_keep ? t : 0)

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::conv::convolve2d_mirror;

const ZSCORE_NAN_REPLACEMENT: f32 = 0.0;

/// numpy.percentile default ("linear" interpolation): for a sorted ascending
/// slice of length n and a percentile p ∈ [0, 100], compute the value at the
/// fractional index k = p/100 * (n - 1) via linear interpolation.
///
/// This function does NOT modify the input slice — it sorts a temporary copy.
fn percentile_linear(values: &[f32], p: f32) -> f32 {
    debug_assert!(!values.is_empty());
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let frac = p / 100.0 * (n as f32 - 1.0);
    let lo = frac.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let w = frac - lo as f32;
    sorted[lo] * (1.0 - w) + sorted[hi] * w
}

/// scipy.stats.zscore with ddof=0: (x - mean) / std, std with denominator n.
fn zscore_ddof0(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = values.iter().copied().sum::<f32>() / n as f32;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    if std == 0.0 {
        // scipy returns NaNs here; upstream replaces with 0.
        return vec![0.0; n];
    }
    values.iter().map(|v| (v - mean) / std).collect()
}

fn nan_to_zero(values: &mut [f32]) {
    for v in values.iter_mut() {
        if v.is_nan() {
            *v = ZSCORE_NAN_REPLACEMENT;
        }
    }
}

/// Mutates `tmp` (one diagonal of E) in place to the upstream's
/// post-normalisation form.
fn normalise_diagonal(tmp: &mut [f32], cap: f32, log_e: bool) {
    let tmp_filter: Vec<bool> = tmp.iter().map(|v| *v > 0.0).collect();
    let mut positives: Vec<f32> = tmp.iter().copied().filter(|v| *v > 0.0).collect();
    if positives.is_empty() {
        for v in tmp.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let normalised: Vec<f32> = if log_e {
        let logs: Vec<f32> = positives.iter().map(|v| v.log10()).collect();
        let mut z = zscore_ddof0(&logs);
        nan_to_zero(&mut z);
        z.iter().map(|v| v.clamp(-cap, cap)).collect()
    } else {
        let cutoff = percentile_linear(&positives, 99.0);
        let clipped: Vec<f32> = positives.iter().map(|v| v.min(cutoff)).collect();
        let mut z = zscore_ddof0(&clipped);
        nan_to_zero(&mut z);
        z.iter().map(|v| v.clamp(-cap, cap)).collect()
    };

    let min_val = normalised
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);

    let mut k = 0usize;
    for (i, was_positive) in tmp_filter.iter().enumerate() {
        if *was_positive {
            tmp[i] = normalised[k];
            k += 1;
        } else {
            tmp[i] = min_val;
        }
    }
}

/// Build the donut-minus kernel of width `w = 2*pad+1` matching loop_bkg.py
/// lines 99-102. Returns (kernel, w).
fn donut_kernel(pad: usize, gap: usize) -> (Vec<f32>, usize) {
    let w = 2 * pad + 1;
    let mut k = vec![1.0_f32; w * w];
    let lo = pad - gap;
    let hi = pad + gap;
    for i in lo..=hi {
        for j in lo..=hi {
            k[i * w + j] = 0.0;
        }
    }
    let s: f32 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= s;
    }
    (k, w)
}

#[allow(clippy::too_many_arguments)]
pub fn loop_bkg_chrom(
    rows: &[u32],
    cols: &[u32],
    vals: &[f32],
    n: usize,
    dist_bins: usize,
    cap: f32,
    pad: usize,
    gap: usize,
    min_cutoff: f32,
    log_e: bool,
) -> (
    (Vec<u32>, Vec<u32>, Vec<f32>),
    (Vec<u32>, Vec<u32>, Vec<f32>),
) {
    // 1. densify upper-tri triplets
    let mut e_dense = Array2::<f32>::zeros((n, n));
    {
        let buf = e_dense.as_slice_mut().expect("contig");
        for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
            let r = r as usize;
            let c = c as usize;
            if r < n && c < n {
                buf[r * n + c] = v;
            }
        }
    }

    // 2. zero diagonal
    let buf = e_dense.as_slice_mut().expect("contig");
    for i in 0..n {
        buf[i * n + i] = 0.0;
    }

    // 3. per-diagonal pctl-99/zscore±cap normalisation
    let dist = dist_bins.min(n.saturating_sub(1));
    for d in 1..=dist {
        let length = n - d;
        let mut tmp = vec![0.0_f32; length];
        for k in 0..length {
            tmp[k] = buf[k * n + (k + d)];
        }
        normalise_diagonal(&mut tmp, cap, log_e);
        for k in 0..length {
            buf[k * n + (k + d)] = tmp[k];
        }
    }

    // 4. donut convolution -> T_raw (dense)
    let (kernel, kw) = donut_kernel(pad, gap);
    let t_dense_vec = convolve2d_mirror(buf, n, n, &kernel, kw, kw);

    // 5. upper-tri mask + sparse emission with min_cutoff and T = E - T
    let mut e_rows: Vec<u32> = Vec::new();
    let mut e_cols: Vec<u32> = Vec::new();
    let mut e_vals: Vec<f32> = Vec::new();
    let mut t_rows: Vec<u32> = Vec::new();
    let mut t_cols: Vec<u32> = Vec::new();
    let mut t_vals: Vec<f32> = Vec::new();

    let in_mask = |i: usize, j: usize| -> bool {
        // upstream mask: diagonal + diagonals 1..=dist_bins
        j >= i && (j - i) <= dist_bins
    };

    for i in 0..n {
        for j in i..n {
            if !in_mask(i, j) {
                continue;
            }
            let e = buf[i * n + j];
            let t_raw = t_dense_vec[i * n + j];

            let e_keep = if min_cutoff > 0.0 {
                e.abs() > min_cutoff
            } else {
                e != 0.0
            };
            let t_keep = if min_cutoff > 0.0 {
                t_raw.abs() > min_cutoff
            } else {
                t_raw != 0.0
            };

            let e_contrib = if e_keep { e } else { 0.0 };
            let t_contrib = if t_keep { t_raw } else { 0.0 };
            let t_out = e_contrib - t_contrib;

            if e_keep {
                e_rows.push(i as u32);
                e_cols.push(j as u32);
                e_vals.push(e);
            }
            if t_out != 0.0 {
                t_rows.push(i as u32);
                t_cols.push(j as u32);
                t_vals.push(t_out);
            }
        }
    }

    ((e_rows, e_cols, e_vals), (t_rows, t_cols, t_vals))
}

#[pyfunction]
#[pyo3(signature = (rows, cols, vals, n, dist_bins, cap=5.0, pad=5, gap=2, min_cutoff=1e-6, log_e=false))]
#[allow(clippy::too_many_arguments)]
pub fn py_loop_bkg_chrom<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, u32>,
    cols: PyReadonlyArray1<'py, u32>,
    vals: PyReadonlyArray1<'py, f32>,
    n: usize,
    dist_bins: usize,
    cap: f32,
    pad: usize,
    gap: usize,
    min_cutoff: f32,
    log_e: bool,
) -> PyResult<(
    (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>),
    (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>),
)> {
    let r = rows.as_slice()?;
    let c = cols.as_slice()?;
    let v = vals.as_slice()?;

    let ((er, ec, ev), (tr, tc, tv)) = py.allow_threads(|| {
        loop_bkg_chrom(r, c, v, n, dist_bins, cap, pad, gap, min_cutoff, log_e)
    });

    Ok((
        (
            ndarray::Array1::from(er).into_pyarray_bound(py),
            ndarray::Array1::from(ec).into_pyarray_bound(py),
            ndarray::Array1::from(ev).into_pyarray_bound(py),
        ),
        (
            ndarray::Array1::from(tr).into_pyarray_bound(py),
            ndarray::Array1::from(tc).into_pyarray_bound(py),
            ndarray::Array1::from(tv).into_pyarray_bound(py),
        ),
    ))
}
