//! General 2-D convolution matching scipy.ndimage.convolve(mode='mirror').
//!
//! Semantics (scipy convolve, NOT correlate):
//!   out[i, j] = Σ_p,q k[kh-1-p, kw-1-q] · a[mirror(i + p - oh), mirror(j + q - ow)]
//! where (oh, ow) = ((kh-1)//2, (kw-1)//2) when origin=0.
//!
//! For odd-sized symmetric kernels (the only kind scHiCluster's loop module
//! uses) this is equivalent to correlation with the same kernel — but the
//! port stays correct for any kernel shape, so the loop / scan code paths
//! and any future asymmetric kernels both work.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::mirror_index;

/// `out[i, j] = Σ_p,q kernel[kh-1-p, kw-1-q] · a[mirror(i + p - oh), mirror(j + q - ow)]`.
///
/// Output buffer is row-major contiguous of shape `(nrows, ncols)`. Parallel
/// across output rows (each row's reduction is fixed-order, so this is an
/// (E) exact rewrite vs a serial outer loop).
pub fn convolve2d_mirror(
    a: &[f32],
    nrows: usize,
    ncols: usize,
    kernel: &[f32],
    kh: usize,
    kw: usize,
) -> Vec<f32> {
    debug_assert_eq!(a.len(), nrows * ncols);
    debug_assert_eq!(kernel.len(), kh * kw);

    // scipy origin=0 anchor for convolve: oh = (kh - 1) / 2, ow = (kw - 1) / 2.
    let oh = ((kh as i32) - 1) / 2;
    let ow = ((kw as i32) - 1) / 2;

    let mut out = vec![0.0_f32; nrows * ncols];
    let nr_i32 = nrows as i32;
    let nc_i32 = ncols as i32;
    let kh_i32 = kh as i32;
    let kw_i32 = kw as i32;

    out.par_chunks_mut(ncols).enumerate().for_each(|(i, row_out)| {
        let i_i = i as i32;
        for j in 0..ncols {
            let j_i = j as i32;
            let mut acc = 0.0_f32;
            for p in 0..kh_i32 {
                let src_i = mirror_index(i_i + p - oh, nr_i32);
                let row_base = src_i * ncols;
                let k_row = (kh_i32 - 1 - p) as usize * kw;
                for q in 0..kw_i32 {
                    let src_j = mirror_index(j_i + q - ow, nc_i32);
                    let k_val = kernel[k_row + (kw_i32 - 1 - q) as usize];
                    acc += k_val * a[row_base + src_j];
                }
            }
            row_out[j] = acc;
        }
    });

    out
}

#[pyfunction]
#[pyo3(signature = (a, kernel))]
pub fn py_convolve2d_mirror<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    kernel: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let a_view = a.as_array();
    let k_view = kernel.as_array();
    let (nrows, ncols) = (a_view.nrows(), a_view.ncols());
    let (kh, kw) = (k_view.nrows(), k_view.ncols());

    let a_slice = a_view
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("a must be C-contiguous f32"))?;
    let k_slice = k_view
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("kernel must be C-contiguous f32"))?;

    let out_vec = py.allow_threads(|| {
        convolve2d_mirror(a_slice, nrows, ncols, k_slice, kh, kw)
    });

    let arr = ndarray::Array2::from_shape_vec((nrows, ncols), out_vec)
        .expect("shape matches");
    Ok(arr.into_pyarray_bound(py))
}
