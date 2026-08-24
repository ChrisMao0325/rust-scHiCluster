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

    // acceleration: ACCELERATION_PLAYBOOK §3.4 hoist-invariant-index-computation (E)
    //
    // mirror_index performs a modulo and two branches, and it is a pure
    // function of (offset, extent) — it never reads `a`. The column mirror
    // depends only on (j, q), yet the original loop recomputed it once per
    // (i, j, p, q): nrows*ncols*kh*kw calls, ~127M for a 1024x1024 input with
    // an 11x11 kernel. Tabulating both axes once turns every one of those into
    // an array load.
    //
    // Integer arithmetic only: the tables reproduce exactly the index sequence
    // the loop computed before, so each float multiply-add sees the same
    // operands in the same order. Bit-identical output, hence (E)-exact —
    // confirmed by conv.convolved holding its gate metric.
    let row_map: Vec<usize> = (0..(nrows + kh))
        .map(|t| mirror_index(t as i32 - oh, nr_i32))
        .collect();
    let col_map: Vec<usize> = (0..(ncols + kw))
        .map(|t| mirror_index(t as i32 - ow, nc_i32))
        .collect();

    // Pre-flip the kernel so the inner loop reads it forwards. Same values in
    // the same accumulation order — only the addressing changes.
    let mut kflip = vec![0.0_f32; kh * kw];
    for p in 0..kh {
        for q in 0..kw {
            kflip[p * kw + q] = kernel[(kh - 1 - p) * kw + (kw - 1 - q)];
        }
    }

    out.par_chunks_mut(ncols).enumerate().for_each(|(i, row_out)| {
        for j in 0..ncols {
            let mut acc = 0.0_f32;
            for p in 0..kh {
                let row_base = row_map[i + p] * ncols;
                let k_row = p * kw;
                for q in 0..kw {
                    acc += kflip[k_row + q] * a[row_base + col_map[j + q]];
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
