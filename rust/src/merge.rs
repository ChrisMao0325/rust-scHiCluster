//! Port of scHiCluster/schicluster/loop/merge_cell_to_group.py::merge_cells_for_single_chromosome.
//!
//! Accumulates (Σ_cells m_c, Σ_cells m_c .* m_c) over per-cell sparse upper-tri
//! matrices in flat COO form (cell_ids[k], rows[k], cols[k], vals[k]).
//! Emit order is row-major (BTreeMap iteration); values stored f32 after f64
//! intermediate accumulation.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use std::collections::BTreeMap;

pub fn merge_cells_sum(
    cell_ids: &[u32],
    rows: &[u32],
    cols: &[u32],
    vals: &[f32],
) -> (
    (Vec<u32>, Vec<u32>, Vec<f32>),
    (Vec<u32>, Vec<u32>, Vec<f32>),
) {
    debug_assert_eq!(cell_ids.len(), rows.len());
    debug_assert_eq!(rows.len(), cols.len());
    debug_assert_eq!(cols.len(), vals.len());

    let mut e_acc: BTreeMap<(u32, u32), f64> = BTreeMap::new();
    let mut e2_acc: BTreeMap<(u32, u32), f64> = BTreeMap::new();

    for ((&_cid, &r), (&c, &v)) in cell_ids
        .iter()
        .zip(rows.iter())
        .zip(cols.iter().zip(vals.iter()))
    {
        let key = (r, c);
        *e_acc.entry(key).or_insert(0.0) += v as f64;
        *e2_acc.entry(key).or_insert(0.0) += (v as f64) * (v as f64);
    }

    let mut e_rows = Vec::with_capacity(e_acc.len());
    let mut e_cols = Vec::with_capacity(e_acc.len());
    let mut e_vals = Vec::with_capacity(e_acc.len());
    for ((r, c), v) in e_acc.into_iter() {
        e_rows.push(r);
        e_cols.push(c);
        e_vals.push(v as f32);
    }

    let mut e2_rows = Vec::with_capacity(e2_acc.len());
    let mut e2_cols = Vec::with_capacity(e2_acc.len());
    let mut e2_vals = Vec::with_capacity(e2_acc.len());
    for ((r, c), v) in e2_acc.into_iter() {
        e2_rows.push(r);
        e2_cols.push(c);
        e2_vals.push(v as f32);
    }

    ((e_rows, e_cols, e_vals), (e2_rows, e2_cols, e2_vals))
}

#[pyfunction]
#[pyo3(signature = (cell_ids, rows, cols, vals))]
pub fn py_merge_cells_sum<'py>(
    py: Python<'py>,
    cell_ids: PyReadonlyArray1<'py, u32>,
    rows: PyReadonlyArray1<'py, u32>,
    cols: PyReadonlyArray1<'py, u32>,
    vals: PyReadonlyArray1<'py, f32>,
) -> PyResult<(
    (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>),
    (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>),
)> {
    let ids = cell_ids.as_slice()?;
    let r = rows.as_slice()?;
    let c = cols.as_slice()?;
    let v = vals.as_slice()?;

    let ((er, ec, ev), (e2r, e2c, e2v)) =
        py.allow_threads(|| merge_cells_sum(ids, r, c, v));

    Ok((
        (
            ndarray::Array1::from(er).into_pyarray_bound(py),
            ndarray::Array1::from(ec).into_pyarray_bound(py),
            ndarray::Array1::from(ev).into_pyarray_bound(py),
        ),
        (
            ndarray::Array1::from(e2r).into_pyarray_bound(py),
            ndarray::Array1::from(e2c).into_pyarray_bound(py),
            ndarray::Array1::from(e2v).into_pyarray_bound(py),
        ),
    ))
}
