//! Port of scHiCluster/schicluster/compartment/call_compartment.py::
//!   single_chrom_compartment + compartment_strength.
//!
//! Matches the upstream sparse pipeline using a dense f64 working buffer:
//!   1. zero the diagonal
//!   2. add identity on zero-sum columns
//!   3. row-major normalize each datum by its column sum
//!   4. comp = matrix · cpg_ratio
//!   5. strength = decay-normalized A/B partition sums

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

fn percentile_linear(values: &mut [f64], p: f64) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return values[0];
    }
    let frac = p / 100.0 * (n as f64 - 1.0);
    let lo = frac.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let w = frac - lo as f64;
    values[lo] * (1.0 - w) + values[hi] * w
}

pub fn compartment_chrom(
    matrix: &Array2<f32>,
    cpg_ratio: &[f32],
    calc_strength: bool,
) -> (Vec<f64>, Option<[f64; 3]>) {
    let n = matrix.nrows();
    assert_eq!(matrix.ncols(), n);
    assert_eq!(cpg_ratio.len(), n);

    // Step 1: copy to f64, zero diagonal.
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|r| (0..n).map(|c| matrix[(r, c)] as f64).collect())
        .collect();
    for i in 0..n {
        m[i][i] = 0.0;
    }

    // Step 2: column sums; identity on zero-sum columns.
    let mut col_sum = vec![0.0_f64; n];
    for r in 0..n {
        for c in 0..n {
            col_sum[c] += m[r][c];
        }
    }
    for j in 0..n {
        if col_sum[j] == 0.0 {
            m[j][j] = 1.0;
            col_sum[j] = 1.0;
        }
    }

    // Step 3: row-major normalize — divide each stored value m[r][c] by col_sum[r].
    // Note upstream uses `np.repeat(col_sum, nnz_per_row)` over CSR data: that's
    // the column sum **of the row**, i.e. each row r's stored values get divided
    // by col_sum[r]. (For a symmetric matrix col_sum[r] == row_sum[r].)
    for r in 0..n {
        if col_sum[r] != 0.0 {
            let inv = 1.0 / col_sum[r];
            for c in 0..n {
                m[r][c] *= inv;
            }
        }
    }

    // Step 4: comp = matrix · cpg_ratio.
    let cpg_f64: Vec<f64> = cpg_ratio.iter().map(|v| *v as f64).collect();
    let mut comp = vec![0.0_f64; n];
    for r in 0..n {
        let mut acc = 0.0_f64;
        for c in 0..n {
            acc += m[r][c] * cpg_f64[c];
        }
        comp[r] = acc;
    }

    let strength = if calc_strength {
        Some(compartment_strength_inner(&m, n, &comp, &cpg_f64))
    } else {
        None
    };

    (comp, strength)
}

fn compartment_strength_inner(
    matrix: &[Vec<f64>],
    n: usize,
    comp: &[f64],
    cpg: &[f64],
) -> [f64; 3] {
    let bin_filter: Vec<bool> = cpg.iter().map(|v| *v > 0.0).collect();
    let tmp: Vec<f64> = (0..n).filter(|&i| bin_filter[i]).map(|i| comp[i]).collect();
    if tmp.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let mut tmp_for_p80 = tmp.clone();
    let mut tmp_for_p20 = tmp.clone();
    let p80 = percentile_linear(&mut tmp_for_p80, 80.0);
    let p20 = percentile_linear(&mut tmp_for_p20, 20.0);

    // a_pos / b_pos in the FILTERED index space (the comp values past bin_filter)
    let filtered_idx: Vec<usize> = (0..n).filter(|&i| bin_filter[i]).collect();
    let a_pos: Vec<bool> = filtered_idx.iter().map(|&i| comp[i] > p80).collect();
    let b_pos: Vec<bool> = filtered_idx.iter().map(|&i| comp[i] < p20).collect();

    // decay[k] = mean of matrix's k-th diagonal (over all of n)
    let mut decay = vec![0.0_f64; n];
    for k in 0..n {
        let mut s = 0.0_f64;
        let mut count = 0usize;
        for i in 0..(n - k) {
            s += matrix[i][i + k];
            count += 1;
        }
        decay[k] = if count > 0 { s / count as f64 } else { 0.0 };
    }

    // Build E = matrix / decay[|col - row|] then index into [bin_filter, bin_filter].
    let m = filtered_idx.len();
    let mut e = vec![0.0_f64; m * m];
    for (i_out, &i_in) in filtered_idx.iter().enumerate() {
        for (j_out, &j_in) in filtered_idx.iter().enumerate() {
            let d = if j_in >= i_in { j_in - i_in } else { i_in - j_in };
            let denom = decay[d];
            let val = if denom != 0.0 { matrix[i_in][j_in] / denom } else { 0.0 };
            e[i_out * m + j_out] = val;
        }
    }

    let mut aa = 0.0_f64;
    let mut bb = 0.0_f64;
    let mut ab = 0.0_f64;
    for i in 0..m {
        for j in 0..m {
            let v = e[i * m + j];
            if a_pos[i] && a_pos[j] {
                aa += v;
            }
            if b_pos[i] && b_pos[j] {
                bb += v;
            }
            if a_pos[i] && b_pos[j] {
                ab += v;
            }
        }
    }
    [aa, bb, ab]
}

#[pyfunction]
#[pyo3(signature = (matrix, cpg_ratio, calc_strength=false))]
pub fn py_compartment_chrom<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f32>,
    cpg_ratio: PyReadonlyArray1<'py, f32>,
    calc_strength: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, PyObject)> {
    let m = matrix.as_array().to_owned();
    let cpg = cpg_ratio.as_slice()?.to_vec();
    let (comp, strength) = py.allow_threads(|| compartment_chrom(&m, &cpg, calc_strength));
    let comp_arr = ndarray::Array1::from(comp).into_pyarray_bound(py);
    let strength_obj: PyObject = match strength {
        Some(s) => {
            let arr = ndarray::Array1::from(s.to_vec()).into_pyarray_bound(py);
            arr.into_py(py)
        }
        None => py.None(),
    };
    Ok((comp_arr, strength_obj))
}
