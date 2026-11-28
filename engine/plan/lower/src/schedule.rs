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
    order(s, false)
}

/// Runs ready ops in concurrent waves with low growth.
pub fn schedule_waves(s: &Selected) -> Vec<usize> {
    order(s, true)
}

fn order(s: &Selected, waves: bool) -> Vec<usize> {
    // Track RAW, WAR, and WAW edges for shared kept buffers.
    let mut last_writer = vec![None::<usize>; s.values.len()];
    let mut readers_since = vec![Vec::<usize>::new(); s.values.len()];
    let mut after: Vec<Vec<usize>> = vec![Vec::new(); s.ops.len()];
    let mut waiting = vec![0usize; s.ops.len()];
    let mut link = |from: usize, to: usize| {
        if from != to && !after[from].contains(&to) {
            after[from].push(to);
            waiting[to] += 1;
        }
    };
    for (i, op) in s.ops.iter().enumerate() {
        for v in distinct(&op.reads) {
            if let Some(w) = last_writer[v.0 as usize] {
                link(w, i);
            }
            readers_since[v.0 as usize].push(i);
        }
        for v in std::iter::once(op.write).chain(op.also.iter().copied()) {
            if let Some(prev_w) = last_writer[v.0 as usize] {
                link(prev_w, i);
            }
            for prev_r in readers_since[v.0 as usize].drain(..) {
                link(prev_r, i);
            }
            last_writer[v.0 as usize] = Some(i);
        }
    }
    let is_comm: Vec<bool> = s.ops.iter().map(|op| crosses(op.kernel.name())).collect();
    let mut comm_seq = vec![0u32; s.values.len()];
    let mut next_seq = 0u32;
    let mut min_wave = vec![0u32; s.ops.len()];
    let mut cur_wave = 0u32;
    let mut ready: Vec<usize> = (0..s.ops.len()).filter(|i| waiting[*i] == 0).collect();
    let mut lives = Lives::new(s);
    let mut order = Vec::with_capacity(s.ops.len());
    while !ready.is_empty() {
        // Issue comm ops first, overlap independent compute, join oldest.
        let tier = |idx: usize| -> u32 {
            if is_comm[idx] {
                return 0;
            }
            let age = s.ops[idx]
                .reads
                .iter()
                .map(|v| comm_seq[v.0 as usize])
                .max()
                .unwrap_or(0);
            1 + age
        };
        let wave_bucket = |idx: usize| -> u8 {
            if !waves {
                return 0;
            }
            // Drain active-wave ops before opening the next wave.
            let w = min_wave[idx];
            u8::from(w == 0 || w > cur_wave)
        };
        let best = (0..ready.len())
            .min_by_key(|at| {
                let idx = ready[*at];
                (tier(idx), wave_bucket(idx), lives.growth(idx), idx)
            })
            .expect("ready is not empty");
        let next = ready.swap_remove(best);
        if waves && min_wave[next] > cur_wave {
            cur_wave = min_wave[next];
        }
        lives.run(next);
        if is_comm[next] {
            next_seq += 1;
            comm_seq[s.ops[next].write.0 as usize] = next_seq;
            for &v in &s.ops[next].also {
                comm_seq[v.0 as usize] = next_seq;
            }
        } else {
            for &v in &s.ops[next].reads {
                comm_seq[v.0 as usize] = 0;
            }
        }
        order.push(next);
        let done_wave = if waves {
            cur_wave.max(min_wave[next])
        } else {
            0
        };
        for &r in &after[next] {
            if waves {
                min_wave[r] = min_wave[r].max(done_wave + 1);
            }
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
