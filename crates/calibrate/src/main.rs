//! Probes this GPU, times the grid and stores the samples.

use calibrate::probe::{fit, read, suite};
use calibrate::{Grid, judge, path};
use lower::{Device, Step};
use sim::{Chip, known};
use std::error::Error;
use std::process::ExitCode;
use std::time::Instant;

/// Prints, per kernel, the model against the clock.
fn report(chip: &dyn Chip, timed: &[(Step, f64)]) {
    println!("{} against timed launches:", chip.name());
    for v in judge(chip, timed) {
        let [low, median, high] = v.ratio;
        println!(
            "  {:9} {:5} launches {:8.3} s  estimate/timed {low:.2} {median:.2} {high:.2}",
            v.kernel, v.launches, v.seconds
        );
    }
}

/// Probes the chip, times the grid, restarts the stored file.
fn run() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let readings = read(&suite())?;
    let mut device = Device::guess();
    let timed = Grid::full().run(&mut device)?;
    let seconds = start.elapsed().as_secs_f64();
    let launches = timed.len();
    println!(
        "{}: timed {launches} launches in {seconds:.0} s",
        device.name
    );
    let pool = fit(format!("{}, probed", device.name), &readings);
    for (name, c) in pool.constants() {
        println!("  {name:7} {:10.4e}  {}", c.value, c.note);
    }
    report(&pool, &timed);
    if let Some(table) = known(&device.name) {
        report(&*table, &timed);
    }
    let path = path(&device.name);
    device.save(&path)?;
    println!("{} samples in {}", device.samples(), path.display());
    Ok(())
}

/// Prints each probe's reading.
fn probe() -> Result<(), Box<dyn Error>> {
    for r in read(&suite())? {
        let (p, us) = (r.probe, r.seconds * 1e6);
        let per = r.seconds * 1e9 / p.trips.max(1) as f64;
        println!(
            "{:?} x{} threads, {} trips: {us:.1} us, {per:.1} ns a trip",
            p.kind, p.threads, p.trips
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let probing = std::env::args().nth(1).as_deref() == Some("probe");
    match if probing { probe() } else { run() } {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tessera-calibrate: {e}");
            ExitCode::FAILURE
        }
    }
}
