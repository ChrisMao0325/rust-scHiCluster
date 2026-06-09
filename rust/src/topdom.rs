//! Native port of scHiCluster/schicluster/domain/TopDom.R::TopDom.
//!
//! Replaces the rpy2 -> R round-trip with a self-contained Rust implementation.
//! Structure mirrors the R original 1-to-1 (function names + comments preserved
//! so a reviewer can diff side-by-side).

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

const PVALUE_CUT: f64 = 0.05;

// ----- Normal CDF (Abramowitz 7.1.26 erf, error ~1.5e-7) ----------------------

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + p * xa);
    let y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-xa * xa).exp();
    sign * y
}

fn pnorm(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

// ----- Average ranks with ties (matches R's rank(ties.method="average")) -----

fn ranks_average(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && values[order[j]] == values[order[i]] {
            j += 1;
        }
        // ranks are 1-indexed; average of (i+1)..=j
        let avg = (i as f64 + 1.0 + j as f64) / 2.0;
        for k in i..j {
            ranks[order[k]] = avg;
        }
        i = j;
    }
    ranks
}

// ----- R's wilcox.test(x, y, alternative="less", exact=FALSE, correct=TRUE) --

fn wilcoxon_ranksum_less(x: &[f64], y: &[f64]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 1.0;
    }
    let n_x = x.len() as f64;
    let n_y = y.len() as f64;

    let mut combined: Vec<f64> = Vec::with_capacity(x.len() + y.len());
    combined.extend_from_slice(x);
    combined.extend_from_slice(y);
    let ranks = ranks_average(&combined);
    let w: f64 = ranks[..x.len()].iter().sum::<f64>() - n_x * (n_x + 1.0) / 2.0;

    let mut sorted = combined.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut tie_correction = 0.0_f64;
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] == sorted[i] {
            j += 1;
        }
        let t = (j - i) as f64;
        if t > 1.0 {
            tie_correction += t * t * t - t;
        }
        i = j;
    }

    let mu = n_x * n_y / 2.0;
    let n_total = n_x + n_y;
    let mut var = n_x * n_y * (n_total + 1.0) / 12.0;
    if n_total > 1.0 {
        var -= n_x * n_y * tie_correction / (12.0 * n_total * (n_total - 1.0));
    }
    if var <= 0.0 {
        return if w < mu { 0.0 } else if w > mu { 1.0 } else { 0.5 };
    }
    let sigma = var.sqrt();
    let z = (w - mu + 0.5) / sigma;
    pnorm(z)
}

// ----- TopDom: diamond signal -------------------------------------------------

fn diamond_mean(matrix: &[Vec<f64>], n: usize, i: usize, size: usize) -> f64 {
    if i + 1 >= n {
        return f64::NAN;
    }
    let r_lo = if i + 1 >= size { i + 1 - size } else { 0 };
    let r_hi = i + 1;
    let c_lo = i + 1;
    let c_hi = (i + size + 1).min(n);
    let nr = r_hi - r_lo;
    let nc = c_hi - c_lo;
    if nr == 0 || nc == 0 {
        return f64::NAN;
    }
    let mut acc = 0.0_f64;
    for r in r_lo..r_hi {
        for c in c_lo..c_hi {
            acc += matrix[r][c];
        }
    }
    acc / (nr * nc) as f64
}

// ----- TopDom: gap regions ---------------------------------------------------

fn which_gap_region2(matrix: &[Vec<f64>], n: usize, w: usize) -> Vec<usize> {
    let mut idx = Vec::new();
    for i in 0..n {
        let c_lo = if i >= w { i - w } else { 0 };
        let c_hi = (i + w + 1).min(n);
        let mut s = 0.0_f64;
        for c in c_lo..c_hi {
            s += matrix[i][c];
        }
        if s == 0.0 {
            idx.push(i);
        }
    }
    idx
}

// ----- TopDom: contiguous non-gap regions ------------------------------------

#[derive(Clone, Copy, Debug)]
struct Region {
    start: usize,
    end: usize,
}

fn which_process_region(rmv: &[usize], n: usize, min_size: usize) -> Vec<Region> {
    let rmv_set: std::collections::HashSet<usize> = rmv.iter().copied().collect();
    let proc_set: Vec<usize> = (0..n).filter(|i| !rmv_set.contains(i)).collect();
    let mut regions = Vec::new();
    if proc_set.is_empty() {
        return regions;
    }
    let mut i = 0;
    while i < proc_set.len() {
        let start = proc_set[i];
        let mut j = i + 1;
        while j < proc_set.len() && proc_set[j] - proc_set[j - 1] <= 1 {
            j += 1;
        }
        let end = proc_set[j - 1];
        if end >= start && end - start >= min_size {
            regions.push(Region { start, end });
        }
        i = j;
    }
    regions
}

// ----- TopDom: Data.Norm + Change.Point + Detect.Local.Extreme ---------------

fn data_norm(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut rx = vec![0.0; n];
    let mut ry = vec![0.0; n];
    if n == 0 {
        return (rx, ry);
    }
    rx[0] = x[0];
    ry[0] = y[0];
    if n == 1 {
        return (rx, ry);
    }
    let dx: Vec<f64> = (1..n).map(|i| (x[i] - x[i - 1]).abs()).collect();
    let dy: Vec<f64> = (1..n).map(|i| (y[i] - y[i - 1]).abs()).collect();
    let mean_dx: f64 = dx.iter().sum::<f64>() / dx.len() as f64;
    let mean_dy: f64 = dy.iter().sum::<f64>() / dy.len() as f64;
    let sx = if mean_dx > 0.0 { 1.0 / mean_dx } else { 1.0 };
    let sy = if mean_dy > 0.0 { 1.0 / mean_dy } else { 1.0 };
    for i in 1..n {
        rx[i] = rx[i - 1] + (x[i] - x[i - 1]) * sx;
        ry[i] = ry[i - 1] + (y[i] - y[i - 1]) * sy;
    }
    (rx, ry)
}

fn change_point(x: &[f64], y: &[f64]) -> Vec<usize> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut fv = vec![f64::NAN; n];
    let mut cp = vec![0usize];
    fv[0] = 0.0;
    let mut i = 0usize;
    while i < n.saturating_sub(1) {
        let mut j = i + 1;
        fv[j] = ((x[j] - x[i]).powi(2) + (y[j] - y[i]).powi(2)).sqrt();
        while j < n - 1 {
            j += 1;
            let denom = ((x[j] - x[i]).powi(2) + (y[j] - y[i]).powi(2)).sqrt();
            if denom == 0.0 {
                cp.push(j - 1);
                j -= 1;
                break;
            }
            let mut sum_err = 0.0_f64;
            for k in (i + 1)..j {
                sum_err += ((y[j] - y[i]) * x[k]
                    - (x[j] - x[i]) * y[k]
                    - (x[i] * y[j])
                    + (x[j] * y[i]))
                    .abs();
            }
            fv[j] = denom - sum_err / denom;
            if fv[j].is_nan() || fv[j - 1].is_nan() {
                cp.push(j - 1);
                j -= 1;
                break;
            }
            if fv[j] < fv[j - 1] {
                cp.push(j - 1);
                j -= 1;
                break;
            }
        }
        i = j;
    }
    cp.push(n - 1);
    cp
}

fn detect_local_extreme(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut ret = vec![0.0_f64; n];
    if n == 0 {
        return ret;
    }
    let mut x_local: Vec<f64> = x.iter().map(|v| if v.is_nan() { 0.0 } else { *v }).collect();
    if n <= 3 {
        let mut min_i = 0;
        let mut max_i = 0;
        for i in 1..n {
            if x_local[i] < x_local[min_i] {
                min_i = i;
            }
            if x_local[i] > x_local[max_i] {
                max_i = i;
            }
        }
        ret[min_i] = -1.0;
        ret[max_i] = 1.0;
        return ret;
    }
    let x_idx: Vec<f64> = (1..=n).map(|v| v as f64).collect();
    let (nx, ny) = data_norm(&x_idx, &x_local);
    x_local = ny;
    let cp = change_point(&nx, &x_local);
    if cp.len() <= 2 || cp.len() == n {
        return ret;
    }
    for i in 1..(cp.len() - 1) {
        let c = cp[i];
        let cm = cp[i - 1];
        if c >= 1 && c + 1 < n {
            if x_local[c] >= x_local[c - 1] && x_local[c] >= x_local[c + 1] {
                ret[c] = 1.0;
            } else if x_local[c] < x_local[c - 1] && x_local[c] < x_local[c + 1] {
                ret[c] = -1.0;
            }
        }
        let mut local_min = f64::INFINITY;
        let mut local_min_i = cm;
        let mut local_max = f64::NEG_INFINITY;
        let mut local_max_i = cm;
        for j in cm..=c {
            if x_local[j] < local_min {
                local_min = x_local[j];
                local_min_i = j;
            }
            if x_local[j] > local_max {
                local_max = x_local[j];
                local_max_i = j;
            }
        }
        let min_val = x_local[cm].min(x_local[c]);
        let max_val = x_local[cm].max(x_local[c]);
        if local_min < min_val {
            ret[local_min_i] = -1.0;
        }
        if local_max > max_val {
            ret[local_max_i] = 1.0;
        }
    }
    ret
}

// ----- TopDom: Get.Diamond.Matrix2 + Upstream/Downstream triangles ----------

fn get_diamond_matrix2_flat(matrix: &[Vec<f64>], n: usize, i: usize, size: usize) -> Vec<f64> {
    let mut out = Vec::new();
    if i >= n - 1 {
        return out;
    }
    let lower = (i + 1).min(n - 1);
    let upper = (i + size).min(n - 1);
    for k in 0..size {
        if i >= k && i + 1 <= n {
            let row = i - k;
            let cols_n = upper - lower + 1;
            for c in 0..cols_n {
                out.push(matrix[row][lower + c]);
            }
        }
    }
    out
}

fn upper_triangle_flat(matrix: &[Vec<f64>], r0: usize, r1: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for r in r0..r1 {
        for c in (r + 1)..r1 {
            out.push(matrix[r][c]);
        }
    }
    out
}

fn get_upstream_triangle(matrix: &[Vec<f64>], n: usize, i: usize, size: usize) -> Vec<f64> {
    let lower = if i >= size { i - size } else { 0 };
    upper_triangle_flat(matrix, lower, (i + 1).min(n))
}

fn get_downstream_triangle(matrix: &[Vec<f64>], n: usize, i: usize, size: usize) -> Vec<f64> {
    if i + 1 >= n {
        return Vec::new();
    }
    let upper = (i + size + 1).min(n);
    upper_triangle_flat(matrix, i + 1, upper)
}

// ----- TopDom: Get.Pvalue ---------------------------------------------------

fn get_pvalue(matrix_scaled: &[Vec<f64>], n: usize, size: usize) -> Vec<f64> {
    let mut pvalue = vec![1.0_f64; n];
    for i in 0..n.saturating_sub(1) {
        let dia = get_diamond_matrix2_flat(matrix_scaled, n, i, size);
        let ups = get_upstream_triangle(matrix_scaled, n, i, size);
        let downs = get_downstream_triangle(matrix_scaled, n, i, size);
        let mut others = ups;
        others.extend(downs);
        let dia_clean: Vec<f64> = dia.into_iter().filter(|v| !v.is_nan()).collect();
        let others_clean: Vec<f64> = others.into_iter().filter(|v| !v.is_nan()).collect();
        let p = wilcoxon_ranksum_less(&dia_clean, &others_clean);
        pvalue[i] = if p.is_nan() { 1.0 } else { p };
    }
    pvalue
}

// ----- TopDom: Convert.Bin.To.Domain.TMP ------------------------------------

#[derive(Clone, Debug)]
pub struct BedRow {
    pub from_id: usize,
    pub to_id: usize,
    pub tag: u8, // 0=gap, 1=domain, 2=boundary
}

const TAG_GAP: u8 = 0;
const TAG_DOMAIN: u8 = 1;
const TAG_BOUNDARY: u8 = 2;

fn convert_bin_to_domain(
    n: usize,
    signal_idx: &[usize],
    gap_idx: &[usize],
    pvalues: &[f64],
) -> Vec<BedRow> {
    let not_gap: Vec<usize> = (0..n).filter(|i| !gap_idx.contains(i)).collect();
    let gap_regions = which_process_region(&not_gap, n, 0);
    let mut gap_rows: Vec<BedRow> = gap_regions
        .iter()
        .map(|r| BedRow { from_id: r.start, to_id: r.end, tag: TAG_GAP })
        .collect();

    let mut rmv: std::collections::HashSet<usize> = signal_idx.iter().copied().collect();
    rmv.extend(gap_idx.iter().copied());
    let rmv_v: Vec<usize> = rmv.into_iter().collect();
    let domain_regions = which_process_region(&rmv_v, n, 0);
    let mut domain_rows: Vec<BedRow> = domain_regions
        .iter()
        .map(|r| BedRow { from_id: r.start, to_id: r.end, tag: TAG_DOMAIN })
        .collect();

    let not_signal: Vec<usize> = (0..n).filter(|i| !signal_idx.contains(i)).collect();
    let boundary_regions = which_process_region(&not_signal, n, 1);
    let boundary_rows: Vec<BedRow> = boundary_regions
        .iter()
        .filter_map(|r| {
            if r.start + 1 < n {
                Some(BedRow { from_id: r.start + 1, to_id: r.end, tag: TAG_BOUNDARY })
            } else {
                None
            }
        })
        .collect();

    gap_rows.extend(domain_rows.drain(..));
    gap_rows.sort_by_key(|r| r.from_id);

    for row in gap_rows.iter_mut() {
        if row.tag == TAG_DOMAIN {
            let n_constr = (row.from_id..=row.to_id)
                .filter(|&i| pvalues.get(i).map(|p| *p < PVALUE_CUT).unwrap_or(false))
                .count();
            let span = row.to_id - row.from_id + 1;
            if n_constr == span {
                row.tag = TAG_BOUNDARY;
            }
        }
    }

    let mut merged: Vec<BedRow> = Vec::new();
    let mut stack: Option<BedRow> = None;
    for row in gap_rows.into_iter() {
        if row.tag == TAG_BOUNDARY {
            stack = Some(match stack {
                Some(s) => BedRow {
                    from_id: s.from_id.min(row.from_id),
                    to_id: s.to_id.max(row.to_id),
                    tag: TAG_BOUNDARY,
                },
                None => row,
            });
        } else {
            if let Some(s) = stack.take() {
                merged.push(s);
            }
            merged.push(row);
        }
    }
    if let Some(s) = stack {
        merged.push(s);
    }
    merged.extend(boundary_rows);
    merged.sort_by_key(|r| r.from_id);
    merged
}

// ----- TopDom: end-to-end orchestrator --------------------------------------

fn topdom_chrom_inner(
    matrix: &[Vec<f64>],
    n: usize,
    window_size: usize,
    stat_filter: bool,
) -> Vec<BedRow> {
    let mut mean_cf = vec![0.0_f64; n];
    for i in 0..n {
        mean_cf[i] = diamond_mean(matrix, n, i, window_size);
    }

    let gap_idx = which_gap_region2(matrix, n, window_size);
    let mut local_ext = vec![-0.5_f64; n];
    let proc_regions = which_process_region(&gap_idx, n, 3);
    for r in proc_regions.iter() {
        let slice: Vec<f64> = mean_cf[r.start..=r.end].to_vec();
        let ext = detect_local_extreme(&slice);
        for (k, v) in ext.iter().enumerate() {
            local_ext[r.start + k] = *v;
        }
    }

    let mut pvalue = vec![1.0_f64; n];
    if stat_filter {
        let mut scaled: Vec<Vec<f64>> = matrix.iter().map(|r| r.clone()).collect();
        for d in 1..=(2 * window_size) {
            if d >= n {
                break;
            }
            let len = n - d;
            let mut diag: Vec<f64> = Vec::with_capacity(len);
            for i in 0..len {
                diag.push(matrix[i][i + d]);
            }
            let mean = diag.iter().sum::<f64>() / len as f64;
            let var = diag.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (len as f64 - 1.0).max(1.0);
            let sd = var.sqrt();
            if sd > 0.0 {
                for i in 0..len {
                    scaled[i][i + d] = (matrix[i][i + d] - mean) / sd;
                    scaled[i + d][i] = scaled[i][i + d];
                }
            } else {
                for i in 0..len {
                    scaled[i][i + d] = 0.0;
                    scaled[i + d][i] = 0.0;
                }
            }
        }

        for r in proc_regions.iter() {
            let sub_n = r.end - r.start + 1;
            let sub: Vec<Vec<f64>> = (r.start..=r.end)
                .map(|i| scaled[i][r.start..=r.end].to_vec())
                .collect();
            let p_local = get_pvalue(&sub, sub_n, window_size);
            for (k, v) in p_local.iter().enumerate() {
                pvalue[r.start + k] = *v;
            }
        }

        for i in 0..n {
            if (local_ext[i] - (-1.0)).abs() < 1e-12 && pvalue[i] < PVALUE_CUT {
                local_ext[i] = -2.0;
            }
        }
        for v in local_ext.iter_mut() {
            if (*v - (-1.0)).abs() < 1e-12 {
                *v = 0.0;
            } else if (*v - (-2.0)).abs() < 1e-12 {
                *v = -1.0;
            }
        }
    }

    let signal_idx: Vec<usize> = (0..n).filter(|&i| local_ext[i] == -1.0).collect();
    let gap_idx_final: Vec<usize> = (0..n).filter(|&i| local_ext[i] == -0.5).collect();
    convert_bin_to_domain(n, &signal_idx, &gap_idx_final, &pvalue)
}

#[pyfunction]
#[pyo3(signature = (matrix, window_size, stat_filter=true))]
pub fn py_topdom_chrom<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f32>,
    window_size: usize,
    stat_filter: bool,
) -> PyResult<Bound<'py, PyList>> {
    let arr = matrix.as_array();
    let n = arr.nrows();
    if arr.ncols() != n {
        return Err(pyo3::exceptions::PyValueError::new_err("matrix must be square"));
    }
    let mat: Vec<Vec<f64>> = (0..n)
        .map(|r| (0..n).map(|c| arr[(r, c)] as f64).collect())
        .collect();
    let rows = py.allow_threads(|| topdom_chrom_inner(&mat, n, window_size, stat_filter));

    let list = PyList::empty_bound(py);
    for row in rows {
        let tup = pyo3::types::PyTuple::new_bound(
            py,
            &[
                row.from_id.into_py(py),
                row.to_id.into_py(py),
                (row.tag as u8).into_py(py),
            ],
        );
        list.append(tup)?;
    }
    Ok(list)
}

#[allow(dead_code)]
fn _unused_anchor(_a: &PyArray1<f32>) {}
