//! Port of scHiCluster/schicluster/draft/gene_score.py's per-gene window-sum
//! loop, shared by both `gene_score_impute` and `gene_score_raw`.
//!
//! Upstream evaluates `D[r0:r1, c0:c1].sum()` once per gene. With ~78.7k genes
//! that is 78.7k scipy submatrix allocations per cell. Here the CSR is visited
//! in place: for each row in the window, the sorted column indices are binary
//! searched for the column range and the matching `data` slice is summed.
//!
//! Parallelism is across genes only. Each gene's reduction stays serial and in
//! CSR row-major order, so this is admissibility class (E) relative to a serial
//! Rust baseline. It is NOT bit-equal to numpy's pairwise `.sum()`, which is why
//! `gene_score.impute` gates as deterministic-bounded at 1e-6 rather than strict.

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

/// Sum a CSR matrix over each of `n_genes` rectangular windows.
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
            let mut acc = 0.0f64;
            for r in r0..r1 {
                let lo = indptr[r] as usize;
                let hi = indptr[r + 1] as usize;
                let row_cols = &indices[lo..hi];
                // scipy guarantees sorted indices on a canonical CSR; the
                // Python wrapper calls sort_indices() defensively.
                let a = row_cols.partition_point(|&c| c < c0);
                let b = row_cols.partition_point(|&c| c < c1);
                for k in a..b {
                    acc += data[lo + k];
                }
            }
            acc
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_rows, n_cols, row_start, row_end, col_start, col_end))]
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
        )
    });
    Ok(Array1::from_vec(out).into_pyarray_bound(py))
}
