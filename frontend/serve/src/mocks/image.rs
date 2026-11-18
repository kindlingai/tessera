//! Pictures for `/v1/images/generations`.

use sensai_tasks::Error;
use sensai_tasks::api::{GenerateImage, Image, Sizes};

/// Red is the seed, blue the index.
pub(super) struct MockImage;

impl GenerateImage for MockImage {
    fn sizes(&self) -> Sizes {
        Sizes {
            step: 2,
            largest: 8,
            default: 2,
        }
    }

    fn generate(&self, _: &str, n: usize, seed: u32, size: usize) -> Result<Vec<Image>, Error> {
        let one = |i: usize| Image {
            width: size,
            height: size,
            rgba: [seed as u8, 0, i as u8, 255].repeat(size * size),
        };
        Ok((0..n).map(one).collect())
    }
}
