//! Port of scHiCluster/schicluster/loop/loop_calling.py::find_summit.
//!
//! O(N^2) neighbour graph over loop pixels in bin coordinates, then max-E heap
//! peel with downhill BFS absorbing neighbours of strictly lower E. Output is
//! (selected idx into input order, cluster sizes).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone)]
struct HeapEntry {
    neg_e: f32,
    idx: u32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.neg_e == other.neg_e && self.idx == other.idx
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a *max*-heap. We want "highest E first" which means
        // "smallest neg_e first" — invert the neg_e order. Tie-break: smaller
        // idx first => invert.
        other
            .neg_e
            .partial_cmp(&self.neg_e)
            .unwrap_or(Ordering::Equal)
            .then(other.idx.cmp(&self.idx))
    }
}

pub fn find_summit_chrom(
    xs: &[u32],
    ys: &[u32],
    es: &[f32],
    dist_thres_bins: u32,
) -> (Vec<u32>, Vec<u32>) {
    let n = xs.len();
    debug_assert_eq!(xs.len(), ys.len());
    debug_assert_eq!(ys.len(), es.len());
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // argsort by x (ascending). Stable sort breaks ties by original index.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| xs[a].cmp(&xs[b]));

    // Build neighbour graph: O(N^2) but stops once x-delta exceeds dist.
    let mut neighbour: Vec<Vec<u32>> = vec![Vec::new(); n];
    if n >= 2 {
        for i in 0..(n - 1) {
            let a = order[i];
            let x_a = xs[a];
            let y_a = ys[a];
            for j in (i + 1)..n {
                let b = order[j];
                let x_b = xs[b];
                if x_b.saturating_sub(x_a) > dist_thres_bins {
                    break;
                }
                let y_b = ys[b];
                let dy = if y_a > y_b { y_a - y_b } else { y_b - y_a };
                if dy <= dist_thres_bins {
                    neighbour[a].push(b as u32);
                    neighbour[b].push(a as u32);
                }
            }
        }
    }

    // Max-heap keyed on E with deterministic ascending-idx tie-break.
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(n);
    for k in 0..n {
        heap.push(HeapEntry { neg_e: -es[k], idx: k as u32 });
    }

    let mut flag = vec![false; n];
    let mut summit_idx: Vec<u32> = Vec::new();
    let mut summit_size: Vec<u32> = Vec::new();
    let mut tot = n;

    while tot > 0 {
        let mut t = match heap.pop() {
            Some(e) => e.idx as usize,
            None => break,
        };
        while flag[t] {
            t = match heap.pop() {
                Some(e) => e.idx as usize,
                None => return (summit_idx, summit_size),
            };
        }
        let root = t;
        let mut q: Vec<usize> = vec![root];
        flag[root] = true;
        tot -= 1;
        let mut head = 0usize;
        let mut flag_tmp = vec![false; n];
        flag_tmp[root] = true;
        while head < q.len() {
            // copy to avoid borrow issues
            let neighbours_of_head: Vec<u32> = neighbour[q[head]].clone();
            for t2_u32 in neighbours_of_head {
                let t2 = t2_u32 as usize;
                if !flag_tmp[t2] && es[t2] < es[q[head]] {
                    if !flag[t2] {
                        flag[t2] = true;
                        tot -= 1;
                    }
                    flag_tmp[t2] = true;
                    q.push(t2);
                }
            }
            head += 1;
        }
        summit_idx.push(root as u32);
        summit_size.push(q.len() as u32);
    }

    (summit_idx, summit_size)
}

#[pyfunction]
#[pyo3(signature = (xs, ys, es, dist_thres_bins))]
pub fn py_find_summit_chrom<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, u32>,
    ys: PyReadonlyArray1<'py, u32>,
    es: PyReadonlyArray1<'py, f32>,
    dist_thres_bins: u32,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let x_s = xs.as_slice()?;
    let y_s = ys.as_slice()?;
    let e_s = es.as_slice()?;
    let (idx, sizes) = py.allow_threads(|| find_summit_chrom(x_s, y_s, e_s, dist_thres_bins));
    Ok((
        ndarray::Array1::from(idx).into_pyarray_bound(py),
        ndarray::Array1::from(sizes).into_pyarray_bound(py),
    ))
}
