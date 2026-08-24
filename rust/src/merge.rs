//! Port of scHiCluster/schicluster/loop/merge_cell_to_group.py::merge_cells_for_single_chromosome.
//!
//! Accumulates (Σ_cells m_c, Σ_cells m_c .* m_c) over per-cell sparse upper-tri
//! matrices in flat COO form (cell_ids[k], rows[k], cols[k], vals[k]).
//! Emit order is row-major (ascending packed key); values stored f32 after f64
//! intermediate accumulation.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;



/// Least-significant-digit radix sort on the packed u64 key, 8 bits per pass.
///
/// **Stable by construction**: each pass scans the input forward and appends to
/// per-digit running offsets, so equal keys keep their relative order. That is
/// what keeps this rewrite (E)-exact — every key's f64 additions still happen in
/// input order. `sort_unstable_by_key` is ~2x faster still and is deliberately
/// not used: it would reorder those additions, making the rewrite (B).
///
/// Pass count is derived from the largest key, so a small chromosome pays 3
/// passes rather than the 8 a full-width u64 sort would need.
fn radix_sort_stable(mut a: Vec<(u64, f32)>) -> Vec<(u64, f32)> {
    let n = a.len();
    if n < 2 {
        return a;
    }
    let max_key = a.iter().map(|&(k, _)| k).max().unwrap();
    let n_passes = if max_key == 0 {
        1
    } else {
        ((64 - max_key.leading_zeros()) as usize).div_ceil(8)
    };
    let mut b = vec![(0u64, 0.0f32); n];
    for p in 0..n_passes {
        let shift = p * 8;
        let mut count = [0usize; 256];
        for &(k, _) in a.iter() {
            count[((k >> shift) & 0xFF) as usize] += 1;
        }
        let mut sum = 0usize;
        for c in count.iter_mut() {
            let t = *c;
            *c = sum;
            sum += t;
        }
        for &item in a.iter() {
            let d = ((item.0 >> shift) & 0xFF) as usize;
            b[count[d]] = item;
            count[d] += 1;
        }
        std::mem::swap(&mut a, &mut b);
    }
    a
}

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

    // acceleration: ACCELERATION_PLAYBOOK §3.4 flat-sort-and-reduce (E)
    //
    // This used two BTreeMap<(u32, u32), f64>, so every input triplet paid two
    // O(log k) tree descents with a heap-allocated node per unique key. On a
    // realistic merge (50 cells x 40k nnz) that was ~8x slower than upstream's
    // scipy sparse accumulation, which does the same job at C level.
    //
    // The replacement packs the key into one u64 (row in the high half, col in
    // the low half, so ascending u64 order IS row-major order), then does a
    // single STABLE radix sort and reduces equal-key runs.
    //
    // (E)-exactness rests on two facts:
    //   1. A stable sort preserves input order among equal keys, so each key's
    //      f64 additions happen in exactly the order the BTreeMap performed
    //      them. Reordering them would be (B). `sort_unstable_by_key` would be
    //      faster and is deliberately NOT used for this reason.
    //   2. Ascending packed-key order reproduces BTreeMap's (row, col) tuple
    //      ordering exactly, so emission stays row-major.
    let n = rows.len();
    if n == 0 {
        return ((vec![], vec![], vec![]), (vec![], vec![], vec![]));
    }
    // Compress the key to row * ncols + col rather than (row << 32) | col: it
    // preserves row-major order identically while needing far fewer radix
    // passes (24 bits instead of 44 for a 3000-bin chromosome).
    let ncols = (*cols.iter().max().unwrap() as u64) + 1;
    let mut packed: Vec<(u64, f32)> = Vec::with_capacity(n);
    for ((&_cid, &r), (&c, &v)) in cell_ids
        .iter()
        .zip(rows.iter())
        .zip(cols.iter().zip(vals.iter()))
    {
        packed.push((r as u64 * ncols + c as u64, v));
    }
    let packed = radix_sort_stable(packed);

    let mut e_rows: Vec<u32> = Vec::new();
    let mut e_cols: Vec<u32> = Vec::new();
    let mut e_vals: Vec<f32> = Vec::new();
    let mut e2_rows: Vec<u32> = Vec::new();
    let mut e2_cols: Vec<u32> = Vec::new();
    let mut e2_vals: Vec<f32> = Vec::new();

    let mut i = 0usize;
    while i < packed.len() {
        let key = packed[i].0;
        let mut e_sum = 0.0f64;
        let mut e2_sum = 0.0f64;
        while i < packed.len() && packed[i].0 == key {
            let v = packed[i].1 as f64;
            e_sum += v;
            e2_sum += v * v;
            i += 1;
        }
        let r = (key / ncols) as u32;
        let c = (key % ncols) as u32;
        e_rows.push(r);
        e_cols.push(c);
        e_vals.push(e_sum as f32);
        e2_rows.push(r);
        e2_cols.push(c);
        e2_vals.push(e2_sum as f32);
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
