//! Port of scHiCluster/schicluster/cool/contact_distance.py::compute_decay.
//!
//! Upstream reads the whole gzipped contact TSV into a pandas DataFrame just to
//! use four columns. Here the file is streamed line by line in constant memory
//! and no DataFrame is ever built. This is the one place in the port where Rust
//! owns file I/O; see the design spec §4.2.
//!
//! Both outputs are integer counts, exact under any summation order, so they
//! gate as deterministic-strict.

use flate2::read::MultiGzDecoder;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};

/// Locate `v` in `edges` under `np.histogram`'s rules: bins are right-open
/// except the final bin, which is right-closed; values outside
/// `[edges[0], edges[last]]` are dropped entirely.
fn hist_bin(edges: &[f64], v: f64) -> Option<usize> {
    let nb = edges.len() - 1;
    if !(v >= edges[0] && v <= edges[nb]) {
        return None;
    }
    if v == edges[nb] {
        return Some(nb - 1);
    }
    // partition_point counts edges <= v; v >= edges[0] guarantees count >= 1.
    let i = edges.partition_point(|&e| e <= v) - 1;
    Some(i.min(nb - 1))
}

fn open_maybe_gzip(path: &str) -> std::io::Result<Box<dyn Read>> {
    let file = File::open(path)?;
    if path.ends_with(".gz") {
        // MultiGzDecoder, not GzDecoder: concatenated gzip members are common
        // in contact files produced by streaming writers.
        Ok(Box::new(MultiGzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}

pub struct Decay {
    pub hist: Vec<u64>,
    pub sparsity: Vec<(String, u64)>,
}

#[allow(clippy::too_many_arguments)]
pub fn compute_decay(
    path: &str,
    chroms: &[String],
    edges: &[f64],
    resolution: i64,
    chrom1: usize,
    pos1: usize,
    chrom2: usize,
    pos2: usize,
) -> Result<Decay, String> {
    let known: HashSet<&str> = chroms.iter().map(|s| s.as_str()).collect();
    let n_bins = edges.len() - 1;
    let mut hist = vec![0u64; n_bins];
    let mut pairs: HashMap<String, HashSet<(i64, i64)>> = HashMap::new();

    let reader = BufReader::new(open_maybe_gzip(path).map_err(|e| format!("{}: {}", path, e))?);
    let max_col = chrom1.max(pos1).max(chrom2).max(pos2);

    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("{}:{}: {}", path, lineno + 1, e))?;
        let line = line.trim_end_matches(['\r', '\n']);
        // pandas read_csv defaults to skip_blank_lines=True.
        if line.trim().is_empty() {
            continue;
        }
        let mut c1 = "";
        let mut p1 = "";
        let mut c2 = "";
        let mut p2 = "";
        let mut seen = 0usize;
        for (i, f) in line.split('\t').enumerate() {
            if i == chrom1 {
                c1 = f;
            }
            if i == pos1 {
                p1 = f;
            }
            if i == chrom2 {
                c2 = f;
            }
            if i == pos2 {
                p2 = f;
            }
            seen = i;
        }
        if seen < max_col {
            return Err(format!(
                "{}:{}: expected at least {} tab-separated fields, found {}",
                path,
                lineno + 1,
                max_col + 1,
                seen + 1
            ));
        }
        // Cis contacts on known chroms only — upstream's two filters, in order.
        if c1 != c2 || !known.contains(c1) {
            continue;
        }
        // Upstream does NOT pass comment='#' here (unlike filter-contact), so a
        // '#' line is a hard failure upstream too. Fail loudly rather than skip.
        let p1: i64 = p1.trim().parse().map_err(|_| {
            format!(
                "{}:{}: cannot parse pos1 {:?} as an integer",
                path,
                lineno + 1,
                p1
            )
        })?;
        let p2: i64 = p2.trim().parse().map_err(|_| {
            format!(
                "{}:{}: cannot parse pos2 {:?} as an integer",
                path,
                lineno + 1,
                p2
            )
        })?;

        // Histogram uses RAW positions; upstream only floor-divides afterwards.
        if let Some(b) = hist_bin(edges, (p2 - p1).abs() as f64) {
            hist[b] += 1;
        }
        let b1 = p1.div_euclid(resolution);
        let b2 = p2.div_euclid(resolution);
        if b1 != b2 {
            // Ordered pair, matching upstream's groupby on (chrom, pos1, pos2).
            pairs.entry(c1.to_string()).or_default().insert((b1, b2));
        }
    }

    // Chroms with no surviving off-diagonal pair are absent from upstream's
    // value_counts() output, so omit rather than emit zeros. Sorted by chrom
    // name so both parity dumps agree on ordering.
    let mut sparsity: Vec<(String, u64)> = pairs
        .into_iter()
        .map(|(k, v)| (k, v.len() as u64))
        .collect();
    sparsity.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(Decay { hist, sparsity })
}

#[pyfunction]
#[pyo3(signature = (path, chroms, bin_edges, resolution, chrom1, pos1, chrom2, pos2))]
#[allow(clippy::too_many_arguments)]
pub fn py_contact_decay_cell<'py>(
    py: Python<'py>,
    path: &str,
    chroms: Vec<String>,
    bin_edges: PyReadonlyArray1<'py, f64>,
    resolution: i64,
    chrom1: usize,
    pos1: usize,
    chrom2: usize,
    pos2: usize,
) -> PyResult<(Vec<u64>, Vec<(String, u64)>)> {
    let edges = bin_edges.as_slice()?;
    if edges.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bin_edges must contain at least 2 edges",
        ));
    }
    if resolution <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "resolution must be positive",
        ));
    }
    let out = py
        .allow_threads(|| {
            compute_decay(path, &chroms, edges, resolution, chrom1, pos1, chrom2, pos2)
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((out.hist, out.sparsity))
}
