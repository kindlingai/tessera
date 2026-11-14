/// A fixed stream of test values in `[-1, 1)`.
pub struct Noise(pub u64);

impl Noise {
    pub fn draw(&mut self) -> f32 {
        let step = self.0.wrapping_mul(6_364_136_223_846_793_005);
        self.0 = step.wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}
