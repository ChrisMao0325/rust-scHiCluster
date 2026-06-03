//! Port of scHiCluster/schicluster/loop/loop_calling.py::loop_background.
//!
//! Builds bl / donut / h / v background kernels (3 asymmetric, 1 symmetric),
//! convolves E with each (scipy mirror), masks by (E > 0), gathers values at
//! loop pixels (xs, ys).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::conv::convolve2d_mirror;

/// kernel_bl: w x w; ones in two L-shaped strips in the bottom-left corner.
fn build_bl_kernel(pad: usize, gap: usize) -> (Vec<f32>, usize, usize) {
    let w = 2 * pad + 1;
    let mut k = vec![0.0_f32; w * w];
    // upstream: kernel_bl[-pad:, :(pad-gap)] = 1
    for i in (w - pad)..w {
        for j in 0..(pad - gap) {
            k[i * w + j] = 1.0;
        }
    }
    // upstream: kernel_bl[-(pad-gap):, :pad] = 1
    for i in (w - (pad - gap))..w {
        for j in 0..pad {
            k[i * w + j] = 1.0;
        }
    }
    let s: f32 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= s;
    }
    (k, w, w)
}

/// kernel_donut: w x w; ones except central cross and central block, normalised.
fn build_donut_kernel(pad: usize, gap: usize) -> (Vec<f32>, usize, usize) {
    let w = 2 * pad + 1;
    let mut k = vec![1.0_f32; w * w];
    for j in 0..w {
        k[pad * w + j] = 0.0; // row `pad` zeroed
        k[j * w + pad] = 0.0; // column `pad` zeroed
    }
    for i in (pad - gap)..=(pad + gap) {
        for j in (pad - gap)..=(pad + gap) {
            k[i * w + j] = 0.0;
        }
    }
    let s: f32 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= s;
    }
    (k, w, w)
}

/// kernel_h: 3 x w (horizontal stripe); central gap zeroed in column direction.
fn build_h_kernel(pad: usize, gap: usize) -> (Vec<f32>, usize, usize) {
    let w = 2 * pad + 1;
    let kh = 3usize;
    let mut k = vec![1.0_f32; kh * w];
    for i in 0..kh {
        for j in (pad - gap)..=(pad + gap) {
            k[i * w + j] = 0.0;
        }
    }
    let s: f32 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= s;
    }
    (k, kh, w)
}

/// kernel_v: w x 3 (vertical stripe); central gap zeroed in row direction.
fn build_v_kernel(pad: usize, gap: usize) -> (Vec<f32>, usize, usize) {
    let w = 2 * pad + 1;
    let kw = 3usize;
    let mut k = vec![1.0_f32; w * kw];
    for i in (pad - gap)..=(pad + gap) {
        for j in 0..kw {
            k[i * kw + j] = 0.0;
        }
    }
    let s: f32 = k.iter().sum();
    for v in k.iter_mut() {
        *v /= s;
    }
    (k, w, kw)
}

/// Convolve E with `kernel`, mask by (E > 0), gather at (xs, ys).
fn scan_with_kernel(
    e: &[f32],
    n: usize,
    kernel: &[f32],
    kh: usize,
    kw: usize,
    xs: &[u32],
    ys: &[u32],
) -> Vec<f32> {
    let convd = convolve2d_mirror(e, n, n, kernel, kh, kw);
    let mut out = Vec::with_capacity(xs.len());
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let i = x as usize;
        let j = y as usize;
        let e_val = e[i * n + j];
        let v = if e_val > 0.0 { convd[i * n + j] } else { 0.0 };
        out.push(v);
    }
    out
}

pub fn scan_kernels_chrom(
    e: &[f32],
    n: usize,
    pad: usize,
    gap: usize,
    xs: &[u32],
    ys: &[u32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (kbl, kbl_h, kbl_w) = build_bl_kernel(pad, gap);
    let (kdo, kdo_h, kdo_w) = build_donut_kernel(pad, gap);
    let (kh_k, kh_h, kh_w) = build_h_kernel(pad, gap);
    let (kv_k, kv_h, kv_w) = build_v_kernel(pad, gap);
    let bl = scan_with_kernel(e, n, &kbl, kbl_h, kbl_w, xs, ys);
    let donut = scan_with_kernel(e, n, &kdo, kdo_h, kdo_w, xs, ys);
    let h = scan_with_kernel(e, n, &kh_k, kh_h, kh_w, xs, ys);
    let v = scan_with_kernel(e, n, &kv_k, kv_h, kv_w, xs, ys);
    (bl, donut, h, v)
}

#[pyfunction]
#[pyo3(signature = (e, pad, gap, xs, ys))]
pub fn py_scan_kernels_chrom<'py>(
    py: Python<'py>,
    e: PyReadonlyArray2<'py, f32>,
    pad: usize,
    gap: usize,
    xs: PyReadonlyArray1<'py, u32>,
    ys: PyReadonlyArray1<'py, u32>,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let e_view = e.as_array();
    let (n, n2) = (e_view.nrows(), e_view.ncols());
    if n != n2 {
        return Err(pyo3::exceptions::PyValueError::new_err("E must be square"));
    }
    let e_slice = e_view
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("E must be C-contiguous f32"))?;
    let xs_s = xs.as_slice()?;
    let ys_s = ys.as_slice()?;

    let (bl, donut, h, v) = py.allow_threads(|| {
        scan_kernels_chrom(e_slice, n, pad, gap, xs_s, ys_s)
    });

    Ok((
        ndarray::Array1::from(bl).into_pyarray_bound(py),
        ndarray::Array1::from(donut).into_pyarray_bound(py),
        ndarray::Array1::from(h).into_pyarray_bound(py),
        ndarray::Array1::from(v).into_pyarray_bound(py),
    ))
}
