//! Shared sparse / banded / dense helpers used across modules.
//!
//! Moved out of `lib.rs` so new modules can depend on this without making
//! `lib.rs` grow further.

use ndarray::Array2;
use sprs::CsMatI;

pub type Csr = CsMatI<f32, usize>;

pub fn triplets_to_csr(rows: &[u32], cols: &[u32], vals: &[f32], n: usize) -> Csr {
    let mut tri: sprs::TriMatI<f32, usize> = sprs::TriMatI::with_capacity((n, n), rows.len());
    for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
        tri.add_triplet(r as usize, c as usize, v);
    }
    tri.to_csr()
}

pub fn csr_to_dense(p: &Csr) -> Array2<f32> {
    let (m, n) = (p.rows(), p.cols());
    let mut d = Array2::<f32>::zeros((m, n));
    let buf = d.as_slice_mut().expect("dense contig");
    for (i, p_row) in p.outer_iterator().enumerate() {
        let cols = p_row.indices();
        let vals = p_row.data();
        for (k, &v) in cols.iter().zip(vals.iter()) {
            buf[i * n + k] = v;
        }
    }
    d
}

/// scipy 'mirror' reflect index — period `2 * (n - 1)`, no edge repeat.
#[inline]
pub fn mirror_index(i: i32, n: i32) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut x = i % period;
    if x < 0 {
        x += period;
    }
    if x >= n {
        x = period - x;
    }
    x as usize
}

#[inline]
pub fn band_w(b: usize) -> usize { 2 * b + 1 }

#[inline]
pub fn band_idx(i: usize, j: usize, b: usize) -> Option<usize> {
    let d = j as i64 - i as i64;
    if d.abs() > b as i64 {
        None
    } else {
        Some((d + b as i64) as usize)
    }
}

pub fn csr_to_banded(p: &Csr, b: usize) -> Vec<f32> {
    let n = p.rows();
    let bw = band_w(b);
    let mut data = vec![0.0_f32; n * bw];
    for (i, p_row) in p.outer_iterator().enumerate() {
        for (k, &v) in p_row.indices().iter().zip(p_row.data().iter()) {
            if let Some(off) = band_idx(i, *k, b) {
                data[i * bw + off] = v;
            }
        }
    }
    data
}
