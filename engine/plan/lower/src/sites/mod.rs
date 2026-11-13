//! Patterns rules recognize at tree nodes, one type each.

mod attend;
mod chain;
mod gather;
mod matmul;
mod stage;
mod top;

pub use attend::Attend;
pub use chain::Chain;
pub use gather::Gather;
pub use matmul::{After, Finished, Matmul};
pub use stage::Stage;
pub use top::Top;

use crate::tree::{T, Tree};

/// A pattern a rule asks for at a node.
pub trait Pattern: Sized {
    fn find(tree: &Tree, at: T) -> Option<Self>;
}
