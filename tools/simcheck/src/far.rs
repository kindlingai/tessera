//! Alike crates that sit apart in the tree.

use crate::windows::Window;
use rayon::prelude::*;

/// Two crates and how much each leans to the other.
pub struct Pair {
    pub a: usize,
    pub b: usize,
    /// Windows in each crate.
    pub sizes: (usize, usize),
    /// Windows nearer the other crate than their own.
    pub leans: (usize, usize),
    /// Directory steps from one crate to the other.
    pub steps: usize,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The deepest crate directory above `path`.
fn home(path: &str, crates: &[String]) -> usize {
    (0..crates.len())
        .filter(|&c| crates[c].is_empty() || path.starts_with(&format!("{}/", crates[c])))
        .max_by_key(|&c| crates[c].len())
        .unwrap_or(0)
}

/// Whether two crates share a top-level folder.
fn kin(a: &str, b: &str) -> bool {
    a.split('/').next() == b.split('/').next()
}

fn steps(a: &str, b: &str) -> usize {
    let (a, b): (Vec<&str>, Vec<&str>) = (a.split('/').collect(), b.split('/').collect());
    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    a.len() + b.len() - 2 * common
}

/// Every crate pair with any lean, most pulled first.
pub fn pairs(wins: &[Window], embs: &[Vec<f32>], crates: &[String]) -> Vec<Pair> {
    let n = crates.len();
    let of: Vec<usize> = wins.iter().map(|w| home(&w.path, crates)).collect();
    // Each window's best score per crate, outside its file.
    let best: Vec<Vec<f32>> = (0..wins.len())
        .into_par_iter()
        .map(|i| {
            let mut row = vec![f32::NEG_INFINITY; n];
            let near = |j: &usize| kin(&crates[of[*j]], &crates[of[i]]);
            for j in (0..wins.len()).filter(near) {
                if wins[j].path == wins[i].path {
                    continue;
                }
                let s = dot(&embs[i], &embs[j]);
                row[of[j]] = row[of[j]].max(s);
            }
            row
        })
        .collect();
    let mut size = vec![0; n];
    let mut lean = vec![vec![0usize; n]; n];
    for (i, row) in best.iter().enumerate() {
        let c = of[i];
        size[c] += 1;
        let top = (0..n).max_by(|&x, &y| row[x].total_cmp(&row[y]));
        if let Some(t) = top.filter(|&t| t != c) {
            lean[c][t] += 1;
        }
    }
    let mut out = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            let none = lean[a][b] + lean[b][a] == 0;
            if size[a] == 0 || size[b] == 0 || none || !kin(&crates[a], &crates[b]) {
                continue;
            }
            out.push(Pair {
                a,
                b,
                sizes: (size[a], size[b]),
                leans: (lean[a][b], lean[b][a]),
                steps: steps(&crates[a], &crates[b]),
            });
        }
    }
    out.sort_by(|x, y| pull(y).total_cmp(&pull(x)));
    out
}

/// The larger share of one crate leaning to the other.
pub fn pull(p: &Pair) -> f32 {
    let a = p.leans.0 as f32 / p.sizes.0 as f32;
    a.max(p.leans.1 as f32 / p.sizes.1 as f32)
}

/// Test code reads alike in every crate.
pub fn test(path: &str) -> bool {
    path.ends_with("_test.rs") || path.contains("/tests/")
}

/// One file, its crate, and the nearest other crate.
pub struct File {
    pub path: String,
    pub home: usize,
    /// Score against its own crate, the file left out.
    pub own: f32,
    pub other: usize,
    pub near: f32,
    pub windows: usize,
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let n = dot(&v, &v).sqrt().max(f32::MIN_POSITIVE);
    v.iter_mut().for_each(|x| *x /= n);
    v
}

fn add(into: &mut Vec<f32>, v: &[f32], sign: f32) {
    into.resize(v.len(), 0.0);
    into.iter_mut().zip(v).for_each(|(a, b)| *a += sign * b);
}

/// Files by lean, toward crates marked `movable`.
pub fn files(wins: &[Window], embs: &[Vec<f32>], crates: &[String], movable: &[bool]) -> Vec<File> {
    let mut paths: Vec<&str> = wins.iter().map(|w| w.path.as_str()).collect();
    paths.dedup();
    let mut sums: Vec<Vec<f32>> = vec![Vec::new(); paths.len()];
    let mut size = vec![0; paths.len()];
    let mut at = 0;
    for (w, e) in wins.iter().zip(embs) {
        if w.path != paths[at] {
            at += 1;
        }
        add(&mut sums[at], e, 1.0);
        size[at] += 1;
    }
    let cents: Vec<Vec<f32>> = sums.into_iter().map(unit).collect();
    let homes: Vec<usize> = paths.iter().map(|p| home(p, crates)).collect();
    let mut crate_sum = vec![Vec::new(); crates.len()];
    let mut count = vec![0; crates.len()];
    for (c, v) in homes.iter().zip(&cents) {
        add(&mut crate_sum[*c], v, 1.0);
        count[*c] += 1;
    }
    let crate_cents: Vec<Vec<f32>> = crate_sum.iter().cloned().map(unit).collect();
    let mut out: Vec<File> = (0..paths.len())
        .filter(|&f| count[homes[f]] > 1)
        .filter_map(|f| {
            let h = homes[f];
            let mut rest = crate_sum[h].clone();
            add(&mut rest, &cents[f], -1.0);
            let (other, near) = (0..crates.len())
                .filter(|&c| c != h && count[c] > 0 && movable[c])
                .filter(|&c| kin(&crates[c], &crates[h]))
                .map(|c| (c, dot(&cents[f], &crate_cents[c])))
                .max_by(|a, b| a.1.total_cmp(&b.1))?;
            Some(File {
                path: paths[f].to_string(),
                home: h,
                own: dot(&cents[f], &unit(rest)),
                other,
                near,
                windows: size[f],
            })
        })
        .collect();
    out.sort_by(|a, b| (b.near - b.own).total_cmp(&(a.near - a.own)));
    out
}

/// A new file nearer another crate than its own.
pub struct Lean {
    pub path: String,
    pub home: String,
    pub own: f32,
    pub other: String,
    pub near: f32,
}

/// The files in `added` that lean past `margin`, most first.
pub fn leans(
    wins: &[Window],
    embs: &[Vec<f32>],
    crates: &[String],
    movable: &[bool],
    (added, margin): (&[String], f32),
) -> Vec<Lean> {
    files(wins, embs, crates, movable)
        .into_iter()
        .filter(|f| f.near - f.own > margin && added.contains(&f.path))
        .map(|f| Lean {
            path: f.path,
            home: crates[f.home].clone(),
            own: f.own,
            other: crates[f.other].clone(),
            near: f.near,
        })
        .collect()
}
