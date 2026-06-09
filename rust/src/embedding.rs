//! Port of scHiCluster/schicluster/embedding/calc_embedding.py::make_chrom_matrix's
//! inner extraction loop. SVD stays sklearn — Phase 4's gate is the cell × feature
//! matrix **before** SVD.
//!
//! Operations are pure f32 reads + a single f32 scalar multiply, no reduction —
//! the parity gate is deterministic-strict (atol=0), exact f32 bit-equality.

use ndarray::ArrayView3;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

/// Build the (n_features,) list of `(row, col)` pairs `make_idx` produces.
/// Mirrors numpy's `triu_indices(n, k=1)` row-major enumeration + col-row filter.
fn upper_tri_indices_dist_filtered(n: usize, dist_bins_plus_1: usize) -> Vec<(usize, usize)> {
    let mut idx = Vec::new();
    for r in 0..n {
        for c in (r + 1)..n {
            if c - r < dist_bins_plus_1 {
                idx.push((r, c));
            }
        }
    }
    idx
}

pub fn make_chrom_features(
    cells: ArrayView3<f32>,
    dist_bins_plus_1: usize,
    scale_factor: f32,
) -> Vec<Vec<f32>> {
    let n_cells = cells.shape()[0];
    let n_bins = cells.shape()[1];
    debug_assert_eq!(cells.shape()[2], n_bins);
    let idx = upper_tri_indices_dist_filtered(n_bins, dist_bins_plus_1);
    let n_features = idx.len();
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(n_cells);
    for ci in 0..n_cells {
        let mut row = Vec::with_capacity(n_features);
        for &(r, c) in idx.iter() {
            row.push(cells[(ci, r, c)] * scale_factor);
        }
        out.push(row);
    }
    out
}

#[pyfunction]
#[pyo3(signature = (cells, dist_bins_plus_1, scale_factor))]
pub fn py_make_chrom_features<'py>(
    py: Python<'py>,
    cells: PyReadonlyArray3<'py, f32>,
    dist_bins_plus_1: usize,
    scale_factor: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let view = cells.as_array();
    let n_cells = view.shape()[0];
    let n_bins = view.shape()[1];
    if view.shape()[2] != n_bins {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cells must be (n_cells, n_bins, n_bins) — square per-cell matrices",
        ));
    }
    let rows = py.allow_threads(|| make_chrom_features(view, dist_bins_plus_1, scale_factor));
    let n_features = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut flat = Vec::with_capacity(n_cells * n_features);
    for row in rows {
        flat.extend(row);
    }
    let arr = ndarray::Array2::from_shape_vec((n_cells, n_features), flat).expect("shape");
    Ok(arr.into_pyarray_bound(py))
}
