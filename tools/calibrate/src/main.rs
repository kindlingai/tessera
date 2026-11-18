//! Probes this GPU, times the grid, plans or shapes.

use sensai_calibrate::probe::{fit, links, read, suite};
use sensai_calibrate::{Grid, headroom, judge};
use sensai_calibrate::{bench, guided, plans};
use sensai_lower::{Cluster, Device, Pick, Settings, Step};
use sensai_sim::{Chip, known};
use std::error::Error;
use std::process::ExitCode;
use std::time::Instant;

/// Prints, per kernel, the model against the clock.
fn report(chip: &dyn Chip, timed: &[(Step, f64)], settings: &Settings) {
    println!("{} against timed launches:", chip.name());
    for v in judge(chip, timed, settings) {
        let [low, median, high] = v.ratio;
        println!(
            "  {:9} {:5} launches {:8.3} s  estimate/timed {low:.2} {median:.2} {high:.2}",
            v.kernel, v.launches, v.seconds
        );
    }
}

/// Prints, per kernel, the time above the chip's best case.
fn room(chip: &dyn Chip, timed: &[(Step, f64)]) {
    println!("headroom, timed over the least the work needs:");
    let all: f64 = timed.iter().map(|(_, seconds)| seconds).sum();
    for h in headroom(chip, timed) {
        let share = 100.0 * h.seconds / all;
        let over = h.bound.map_or("no load stated".into(), |b| {
            format!("{:.1}x", h.seconds / b)
        });
        println!("  {:9} {share:5.1}% of the time  {over}", h.kernel);
    }
}

/// Prints each link: its latency, then its rate.
fn hops(cluster: &Cluster) {
    for l in &cluster.links {
        let (from, to) = (&cluster.zones[l.from].name, &cluster.zones[l.to].name);
        let (us, gb) = (l.latency * 1e6, l.rate / 1e9);
        println!("  {from} -> {to}: {us:.2} us, then {gb:.1} GB/s");
    }
}

/// Probes the chip, times the grid, restarts the stored file.
fn run(settings: &Settings) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let readings = read(&suite(), settings)?;
    let cluster = links(settings)?;
    let mut device = Device::guess(settings);
    let timed = Grid::full().run(&mut device, settings)?;
    // The host zone's native kernels, timed on the host.
    let cpu = Settings {
        backend: Pick::Cpu,
        ..settings.clone()
    };
    let mut host = Device::guess(&cpu);
    Grid::host().run(&mut host, &cpu)?;
    let seconds = start.elapsed().as_secs_f64();
    let launches = timed.len();
    println!(
        "{}: timed {launches} launches in {seconds:.0} s",
        device.name
    );
    let fit = fit(&readings).filled(&device.name);
    let pool = fit.pool(&format!("{}, probed", device.name));
    for (name, c) in pool.constants() {
        println!("  {name:7} {:10.4e}  {}", c.value, c.note);
    }
    report(&pool, &timed, settings);
    if let Some(table) = known(&device.name) {
        report(&*table, &timed, settings);
    }
    room(&pool, &timed);
    hops(&cluster);
    // The stored device carries the fit, for unmeasured kernels.
    let device = device.fitted(fit).zoned(cluster).hosted(&host);
    let path = Device::path(settings, &device.platform).ok_or("no home: set TESSERA_COSTS")?;
    device.save(&path)?;
    let (gpu, cpu) = (device.samples(), device.host_samples());
    println!("{gpu} samples and {cpu} on the host in {}", path.display());
    Ok(())
}

/// Prints each probe's reading.
fn probe(settings: &Settings) -> Result<(), Box<dyn Error>> {
    for r in read(&suite(), settings)? {
        let (p, us) = (r.probe, r.seconds * 1e6);
        let per = r.seconds * 1e9 / p.trips.max(1) as f64;
        println!(
            "{:?} x{} threads, {} trips: {us:.1} us, {per:.1} ns a trip",
            p.kind, p.threads, p.trips
        );
    }
    hops(&links(settings)?);
    Ok(())
}

/// Times whole plans. Fails if a check misses.
fn plans(settings: &Settings) -> Result<(), Box<dyn Error>> {
    if settings.gpu().is_none() {
        return Err("no GPU here; plans are timed on one".into());
    }
    let misses = plans::check(settings)?;
    for miss in &misses {
        println!("miss: {miss}");
    }
    match misses.is_empty() {
        true => Ok(()),
        false => Err(format!("{} checks missed", misses.len()).into()),
    }
}

/// A shape slower than its baseline by this share fails.
const SLACK: f64 = 0.2;

/// Times shared shapes, saved or held to a baseline.
fn marks(settings: &Settings, args: &[String]) -> Result<(), Box<dyn Error>> {
    let marks = bench::bench(settings)?;
    for m in &marks {
        let gflops = m.flops / m.seconds / 1e9;
        println!("{:40} {:10.3e} s {gflops:8.1} GFLOP/s", m.name, m.seconds);
    }
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    if let Some(path) = flag("--save") {
        std::fs::write(path, bench::text(&marks))?;
    }
    let Some(path) = flag("--against") else {
        return Ok(());
    };
    let slower = bench::slower(&marks, &std::fs::read_to_string(path)?, SLACK);
    for s in &slower {
        println!("slower: {s}");
    }
    match slower.is_empty() {
        true => Ok(()),
        false => Err(format!("{} shapes slower than {path}", slower.len()).into()),
    }
}

fn main() -> ExitCode {
    let settings = sensai_frontend_settings::read();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match match args.first().map(String::as_str) {
        Some("bench") => marks(&settings, &args[1..]),
        Some("probe") => probe(&settings),
        Some("plans") => plans(&settings),
        Some("guided") => guided::report(
            &settings,
            args.get(1)
                .filter(|a| *a == "--favor")
                .and(args.get(2))
                .cloned(),
        )
        .map_err(Into::into),
        _ => run(&settings),
    } {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tessera-calibrate: {e}");
            ExitCode::FAILURE
        }
    }
}
