//! What a launch does: ops per trip, and its load.

/// What one thread does on each trip of its loop.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ops {
    /// Float adds, multiplies and multiply-adds.
    pub float: f64,
    /// Integer and index arithmetic, compares and branches.
    pub int: f64,
    /// Divides, roots, exponentials and the like.
    pub special: f64,
    pub loads: f64,
    pub stores: f64,
    /// The share of loads that open a new cache line.
    pub fresh: f64,
    /// Bytes one load brings. Zero means a float.
    pub width: f64,
    /// Flops done on the matrix unit, with f32 operands.
    pub tile_flops: f64,
    /// Floats loaded from group memory into tiles.
    pub shared: f64,
    /// Registers each thread holds for tiles. Zero: unknown.
    pub held: f64,
    /// Fields read as narrow floats, counted per format.
    pub f16: f64,
    pub e4m3: f64,
    pub e2m1: f64,
}

/// Integer and float ops one widening by arithmetic takes.
const WIDEN: (f64, f64) = (6.0, 2.0);

impl Ops {
    pub fn width(&self) -> f64 {
        if self.width > 0.0 { self.width } else { WORD }
    }

    /// The conversions, costed as a chip's ops.
    pub fn priced(mut self, native_e4m3: bool) -> Ops {
        let e4m3 = if native_e4m3 { (2.0, 1.0) } else { WIDEN };
        let spent = [
            (self.f16, (1.0, 1.0)),
            (self.e4m3, e4m3),
            (self.e2m1, WIDEN),
        ];
        for (n, (int, float)) in spent {
            self.int += n * int;
            self.float += n * float;
        }
        (self.f16, self.e4m3, self.e2m1) = (0.0, 0.0, 0.0);
        self
    }
}

/// A launch, as the work and traffic it causes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Load {
    pub threads: usize,
    /// Threads that start together as one group.
    pub group: usize,
    /// Loop trips each thread makes.
    pub trips: f64,
    pub trip: Ops,
    /// Bytes of each buffer the launch reads through.
    pub reads: Vec<f64>,
    pub written: f64,
    /// Bytes one thread waits on at a time.
    pub flight: f64,
    /// Group memory each group of threads takes.
    pub group_bytes: f64,
}

impl Load {
    /// Float ops every thread together runs.
    pub fn floats(&self) -> f64 {
        self.threads as f64 * self.trips * self.trip.float
    }

    /// Bytes all loads bring. Shared buffers count per load.
    pub fn loaded(&self) -> f64 {
        self.threads as f64 * self.trips * self.trip.loads * self.trip.width()
    }
}

/// Bytes one load brings: a float.
const WORD: f64 = 4.0;
