//! The dense trellis matmul that the device trellis rules lower.

use tessera_graph::Op;
use tessera_lower::form::Load as Stored;
use tessera_lower::sites::Rotated;
use tessera_lower::{Arg, Draft, Kernel, Read, Step, T, Term, Tree};
use tessera_mem::trellis::parse_form;

/// `x` times a 2D trellis leaf `w`, maybe between rotations.
#[derive(Clone, Copy, Debug)]
pub struct Dense {
    pub x: T,
    pub w: T,
    /// `suh` and `svh`, when rotated.
    pub turn: Option<[T; 2]>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub bits: u32,
    pub book: u32,
    pub x_bf16: bool,
    pub form: &'static dyn Stored,
}

impl Dense {
    /// The trellis matmul at `at`, or why it cannot lower.
    pub fn find(tree: &Tree, at: T, block: usize) -> Option<Result<Self, String>> {
        Self::find_with(tree, at, block, false)
    }

    /// As `find`, allowing bf16 activation inputs when `allow_bf16`.
    pub fn find_with(
        tree: &Tree,
        at: T,
        block: usize,
        allow_bf16: bool,
    ) -> Option<Result<Self, String>> {
        let node = tree.node(at);
        let rotated = tree.site::<Rotated>(at);
        let (x, w, turn) = match rotated {
            Some(r) if r.block as usize == block => (r.x, r.w, Some([r.suh, r.svh])),
            _ if node.term == Term::Op(Op::Matmul) => (node.args[0], node.args[1], None),
            _ => return None,
        };
        if !tree.node(w).form.name().starts_with("trellis") {
            return None;
        }
        Some(Self::check(tree, at, [x, w], turn, block, allow_bf16))
    }

    fn check(
        tree: &Tree,
        at: T,
        [x, w]: [T; 2],
        turn: Option<[T; 2]>,
        block: usize,
        allow_bf16: bool,
    ) -> Result<Self, String> {
        let f32_form = |t: T| tree.node(t).form.name() == "f32";
        if !f32_form(at) {
            let name = tree.node(at).form.name();
            return Err(format!("output form {name} (requires f32)"));
        }
        let form = tree.node(w).form;
        let Some((bits, book)) = parse_form(form.name()) else {
            return Err(format!("unsupported trellis form {}", form.name()));
        };
        let [n, k] = tree.node(w).shape[..] else {
            return Err("trellis weight is not 2D".into());
        };
        if !tree.leaf(w) {
            return Err("trellis weight is not a direct leaf".into());
        }
        for (name, size) in [("n", n), ("k", k)] {
            if !size.is_multiple_of(block) {
                return Err(format!(
                    "{name}={size} is not a multiple of {block} for trellis"
                ));
            }
        }
        let x_name = tree.node(x).form.name();
        let x_bf16 = match x_name {
            "f32" => false,
            "bf16" if allow_bf16 => true,
            _ => return Err(format!("input form {x_name} (requires f32)")),
        };
        let m = tree.elements(x) / k;
        if m == 0 {
            return Err("m=0".into());
        }
        if let Some([suh, svh]) = turn {
            if tree.elements(suh) != k || tree.elements(svh) != n {
                return Err("rotation scale length mismatch".into());
            }
            if !f32_form(suh) || !f32_form(svh) {
                return Err("rotation scales must be f32".into());
            }
        }
        Ok(Self {
            x,
            w,
            turn,
            m,
            n,
            k,
            bits,
            book,
            x_bf16,
            form,
        })
    }

    pub fn turned(&self) -> bool {
        self.turn.is_some()
    }

    /// One `main` launch, or `splits` partial sums `reduce` adds.
    pub fn drafts(
        &self,
        main: impl Kernel + 'static,
        splits: usize,
        reduce: impl Kernel + 'static,
    ) -> Vec<Draft> {
        let len = self.m * self.n;
        let (suh, svh) = match self.turn {
            Some([u, v]) => (Some(u), Some(v)),
            None => (None, None),
        };
        let inputs = [self.x, self.w].into_iter().map(Read::Node);
        if splits == 1 {
            let reads = inputs.chain([suh, svh].into_iter().flatten().map(Read::Node));
            return vec![Draft::new(main, reads, len)];
        }
        let parts = inputs.chain(suh.map(Read::Node));
        let sum = [Read::Draft(0)].into_iter().chain(svh.map(Read::Node));
        vec![
            Draft::new(main, parts, splits * len),
            Draft::new(reduce, sum, len),
        ]
    }
}

/// Buffers, then `[m, n, k, splits]`, bits, book and turn.
pub fn launch_args(step: &Step, sizes: [usize; 4], bits: u32, book: u32, turned: bool) -> Vec<Arg> {
    let [m, n, k, splits] = sizes;
    let suh = if turned { step.read(2) } else { step.read(0) };
    let svh = if turned && splits == 1 {
        step.read(3)
    } else {
        step.read(0)
    };
    vec![
        step.read(0),
        step.read(1),
        suh,
        svh,
        step.out(),
        Arg::count(m),
        Arg::count(n),
        Arg::count(k),
        Arg::count(splits),
        Arg::value(&bits),
        Arg::value(&book),
        Arg::count(usize::from(turned)),
    ]
}
