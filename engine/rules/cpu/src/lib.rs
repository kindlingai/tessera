//! CPU kernels for matmuls: short bodies over `dot`.

mod dense;
#[cfg(test)]
mod dense_test;
mod norm;
