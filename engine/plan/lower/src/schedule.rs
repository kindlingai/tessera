//! Orders ops so the fewest temporary elements are live.

use crate::kernels::crosses;
use crate::select::{Home, Selected, Val};

/// Facts an order needs about each value.
struct Lives<'a> {
    s: &'a Selected,
    /// Per value: ops yet to read it. Outputs stay positive.
    readers: Vec<usize>,
}

impl<'a> Lives<'a> {
    fn new(s: &'a Selected) -> Self {
        let mut readers = vec![0; s.values.len()];
        for op in &s.ops {
            for v in distinct(&op.reads) {
                readers[v.0 as usize] += 1;
            }
        }
        for (v, _) in &s.outputs {
            readers[v.0 as usize] += 1;
        }
        Self { s, readers }
    }

    fn temp(&self, v: Val) -> bool {
        self.s.values[v.0 as usize].home == Home::Temp
    }

    /// Temporary elements op `i` writes. Kept rows count zero.
    fn written(&self, i: usize) -> isize {
        let op = &self.s.ops[i];
        let count = |v: Val| {
            if self.temp(v) {
                self.s.values[v.0 as usize].len as isize
            } else {
                0
            }
        };
        count(op.write) + op.also.iter().copied().map(count).sum::<isize>()
    }

    /// Elements op `i` allocates, less those it lets go.
    fn growth(&self, i: usize) -> isize {
        let op = &self.s.ops[i];
        let len = |v: Val| self.s.values[v.0 as usize].len as isize;
        let dies = |v: &Val| self.temp(*v) && self.readers[v.0 as usize] == 1;
        let freed: isize = distinct(&op.reads).filter(dies).map(len).sum();
        // In place or not, a dying input frees its size.
        self.written(i) - freed
    }

    fn run(&mut self, i: usize) {
        for v in distinct(&self.s.ops[i].reads) {
            self.readers[v.0 as usize] -= 1;
        }
    }
}

fn distinct(reads: &[Val]) -> impl Iterator<Item = Val> + '_ {
    let first = |(at, v): &(usize, &Val)| !reads[..*at].contains(v);
    reads.iter().enumerate().filter(first).map(|(_, v)| *v)
}

/// Runs the ready op that grows live memory least.
pub fn schedule(s: &Selected) -> Vec<usize> {
    // Per op: its readers, and how many inputs are unwritten.
    let mut writer = vec![None; s.values.len()];
    for (i, op) in s.ops.iter().enumerate() {
        writer[op.write.0 as usize] = Some(i);
        for &v in &op.also {
            writer[v.0 as usize] = Some(i);
        }
    }
    let mut after: Vec<Vec<usize>> = vec![Vec::new(); s.ops.len()];
    let mut waiting = vec![0; s.ops.len()];
    for (i, op) in s.ops.iter().enumerate() {
        for v in distinct(&op.reads) {
            if let Some(w) = writer[v.0 as usize] {
                after[w].push(i);
                waiting[i] += 1;
            }
        }
    }
    let is_comm: Vec<bool> = s.ops.iter().map(|op| crosses(op.kernel.name())).collect();
    let mut comm_live = vec![false; s.values.len()];
    let mut ready: Vec<usize> = (0..s.ops.len()).filter(|i| waiting[*i] == 0).collect();
    let mut lives = Lives::new(s);
    let mut order = Vec::with_capacity(s.ops.len());
    while !ready.is_empty() {
        // Issue comm ops immediately, overlap independent compute, then join.
        let tier = |idx: usize| -> u8 {
            if is_comm[idx] {
                0
            } else if s.ops[idx].reads.iter().any(|v| comm_live[v.0 as usize]) {
                2
            } else {
                1
            }
        };
        let best = (0..ready.len())
            .min_by_key(|at| (tier(ready[*at]), lives.growth(ready[*at]), ready[*at]))
            .expect("ready is not empty");
        let next = ready.swap_remove(best);
        lives.run(next);
        if is_comm[next] {
            comm_live[s.ops[next].write.0 as usize] = true;
            for &v in &s.ops[next].also {
                comm_live[v.0 as usize] = true;
            }
        } else {
            for &v in &s.ops[next].reads {
                comm_live[v.0 as usize] = false;
            }
        }
        order.push(next);
        for &r in &after[next] {
            waiting[r] -= 1;
            if waiting[r] == 0 {
                ready.push(r);
            }
        }
    }
    assert_eq!(order.len(), s.ops.len(), "traced order is always possible");
    order
}

/// The most temporary elements alive at once under `order`.
pub fn peak(s: &Selected, order: &[usize]) -> usize {
    let mut lives = Lives::new(s);
    let (mut live, mut most) = (0isize, 0isize);
    for &i in order {
        // The write exists before the reads are let go.
        let growth = lives.growth(i);
        let written = lives.written(i);
        most = most.max(live + written.max(growth));
        live += growth;
        lives.run(i);
    }
    most as usize
}
