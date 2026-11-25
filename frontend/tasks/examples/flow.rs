//! Times FLUX.2 transformer steps on noise at one latent size.

use std::time::Instant;
use tessera_tasks::flow::Flow;

/// Text rows klein pads every prompt to.
const TEXT: usize = 512;

fn main() -> Result<(), tessera_tasks::Error> {
    let args: Vec<String> = std::env::args().collect();
    let [_, dir, side, steps] = args.as_slice() else {
        return Err("usage: flow <model dir> <latent side> <steps>".into());
    };
    let (side, steps): (usize, usize) = (side.parse()?, steps.parse()?);
    let started = Instant::now();
    let mut flow = Flow::open(
        dir.as_ref(),
        &tessera_frontend_settings::read(),
        [1, TEXT, side * side],
        &Default::default(),
        &Default::default(),
    )?;
    eprintln!("opened in {:?}", started.elapsed());
    let mut x: Vec<f32> = (0..side * side * 128)
        .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let text = vec![0.5f32; TEXT * 7680];
    for k in 0..steps {
        let started = Instant::now();
        let v = flow.velocity(&x, &text, side, 1.0 - k as f32 / steps as f32)?;
        x.iter_mut()
            .zip(&v)
            .for_each(|(x, v)| *x -= v / steps as f32);
        eprintln!("step {k}: {:?}", started.elapsed());
    }
    Ok(())
}
