//! Times the grid on this GPU and stores the samples.

use calibrate::{Grid, path};
use lower::Device;
use std::error::Error;
use std::process::ExitCode;
use std::time::Instant;

/// A run starts the stored file over.
fn run() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let mut device = Device::guess();
    let launches = Grid::full().run(&mut device)?.len();
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "{}: timed {launches} launches in {seconds:.0} s",
        device.name
    );
    let path = path(&device.name);
    device.save(&path)?;
    println!("{} samples in {}", device.samples(), path.display());
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tessera-calibrate: {e}");
            ExitCode::FAILURE
        }
    }
}
