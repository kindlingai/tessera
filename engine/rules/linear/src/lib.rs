//! Generated matmul kernels, registered as lowering rules.

/// Sizes, work and form of a kernel over a `Product`.
macro_rules! over_product {
    () => {
        fn sizes(&self) -> Vec<usize> {
            self.product.sizes()
        }

        fn work(&self, _: usize) -> f64 {
            self.product.work()
        }

        fn accepts(&self, input: usize, form: &dyn lower::Form) -> bool {
            self.product.accepts(input, form)
        }
    };
}

mod lanes;
mod product;
mod thin;
mod tiled;

pub use lanes::Lanes;
pub use product::Product;
pub use thin::{THIN_ROWS, Thin};
pub use tiled::{SCHEDULES, Schedule, Tiled};
