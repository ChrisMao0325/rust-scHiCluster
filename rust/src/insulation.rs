//! Port of scHiCluster/schicluster/domain/call_domain.py::single_chrom_calculate_insulation_score.
//!
//! For each row `i`:
//!   intra = (mat[a:i, a:i].sum() + mat[i:i+w, i:i+w].sum()) / area
//!   inter = mat[a:i, i:i+w].sum() / area
//! with `a = max(0, i-w)` and area constants matching upstream's two branches:
//!   i < w  : intra denom = i(i+1)/2 + w(w+1)/2  ; inter denom = i * (i + w)
//!   i >= w : intra denom = w(w+1)              ; inter denom = w * w
//!
//! score[0] = 1.0 (initial); score[i] = inter / (inter + intra) for i >= 1.
//! save_count=True returns a (n, 2) array with [inter, intra] per row.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

fn block_sum(m: &Array2<f32>, r0: usize, r1: usize, c0: usize, c1: usize) -> f64 {
    let mut acc = 0.0_f64;
    for r in r0..r1 {
        for c in c0..c1 {
            acc += m[(r, c)] as f64;
        }
    }
    acc
}

pub fn insulation_score_chrom(matrix: &Array2<f32>, w: usize) -> Vec<f32> {
    let n = matrix.nrows();
    let mut score = vec![1.0_f32; n];
    for i in 1..n {
        let (intra, inter) = if i < w {
            let intra_num =
                block_sum(matrix, 0, i, 0, i) + block_sum(matrix, i, (i + w).min(n), i, (i + w).min(n));
            let intra_denom =
                (i * (i + 1)) as f64 / 2.0 + (w * (w + 1)) as f64 / 2.0;
            let inter_num = block_sum(matrix, 0, i, i, (i + w).min(n));
            let inter_denom = (i * (i + w)) as f64;
            (intra_num / intra_denom, inter_num / inter_denom)
        } else {
            let intra_num = block_sum(matrix, i - w, i, i - w, i)
                + block_sum(matrix, i, (i + w).min(n), i, (i + w).min(n));
            let intra_denom = (w * (w + 1)) as f64;
            let inter_num = block_sum(matrix, i - w, i, i, (i + w).min(n));
            let inter_denom = (w * w) as f64;
            (intra_num / intra_denom, inter_num / inter_denom)
        };
        let denom = inter + intra;
        score[i] = if denom > 0.0 { (inter / denom) as f32 } else { 1.0 };
    }
    score
}

pub fn insulation_score_chrom_save_count(matrix: &Array2<f32>, w: usize) -> Vec<[f32; 2]> {
    let n = matrix.nrows();
    let mut out = vec![[1.0_f32, 1.0_f32]; n];
    for i in 1..n {
        let (intra, inter) = if i < w {
            let intra_num =
                block_sum(matrix, 0, i, 0, i) + block_sum(matrix, i, (i + w).min(n), i, (i + w).min(n));
            let intra_denom =
                (i * (i + 1)) as f64 / 2.0 + (w * (w + 1)) as f64 / 2.0;
            let inter_num = block_sum(matrix, 0, i, i, (i + w).min(n));
            let inter_denom = (i * (i + w)) as f64;
            (intra_num / intra_denom, inter_num / inter_denom)
        } else {
            let intra_num = block_sum(matrix, i - w, i, i - w, i)
                + block_sum(matrix, i, (i + w).min(n), i, (i + w).min(n));
            let intra_denom = (w * (w + 1)) as f64;
            let inter_num = block_sum(matrix, i - w, i, i, (i + w).min(n));
            let inter_denom = (w * w) as f64;
            (intra_num / intra_denom, inter_num / inter_denom)
        };
        out[i] = [inter as f32, intra as f32];
    }
    out
}

#[pyfunction]
#[pyo3(signature = (matrix, window_size, save_count=false))]
pub fn py_insulation_score_chrom<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f32>,
    window_size: usize,
    save_count: bool,
) -> PyResult<PyObject> {
    let m = matrix.as_array().to_owned();
    if save_count {
        let rows = py.allow_threads(|| insulation_score_chrom_save_count(&m, window_size));
        let n = rows.len();
        let mut flat = Vec::with_capacity(n * 2);
        for [inter, intra] in rows {
            flat.push(inter);
            flat.push(intra);
        }
        let arr = ndarray::Array2::from_shape_vec((n, 2), flat).expect("shape");
        Ok(arr.into_pyarray_bound(py).into_py(py))
    } else {
        let v = py.allow_threads(|| insulation_score_chrom(&m, window_size));
        let arr = ndarray::Array1::from(v);
        Ok(arr.into_pyarray_bound(py).into_py(py))
    }
}

// silence the unused-import lint that PyArray2/PyArray1 produce when only one branch is exercised
#[allow(dead_code)]
fn _unused_imports_anchor(_a: &PyArray1<f32>, _b: &PyArray2<f32>) {}
