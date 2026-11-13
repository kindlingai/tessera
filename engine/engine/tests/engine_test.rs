//! The engine's decode loop against the interpreter.

use engine::{Engine, Event, Reason, Request};
use gpukit::testing::Noise;
use graph::Graph;
use interp::{Store, Tensor, run_kept};
use lower::Pins;
use models::qwen3::{Config, stepper};

fn weights(g: &Graph, noise: &mut Noise) -> Vec<Tensor> {
    let size = |e: &graph::DimExpr| e.poly(&g.dims).unwrap().as_const().unwrap() as usize;
    let fill = |w: &graph::Weight| {
        let shape: Vec<usize> = w.ty.shape.0.iter().map(size).collect();
        let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
        Tensor::f32(shape, data)
    };
    g.weights.iter().map(fill).collect()
}

/// Greedy decoding through the interpreter, one row at a time.
fn reference(g: &Graph, weights: &[Tensor], prompt: &[u32], n: usize) -> Vec<u32> {
    let mut store = Store::default();
    let mut tokens = prompt.to_vec();
    let mut out = Vec::new();
    let (mut past, mut next) = (0, prompt.to_vec());
    for _ in 0..n {
        let shape = vec![1, next.len()];
        let positions = (past as u32..(past + next.len()) as u32).collect();
        let feeds = [
            Tensor::u32(shape.clone(), next.clone()),
            Tensor::u32(shape, positions),
        ];
        let got = run_kept(g, weights, &mut store, 0, &feeds).unwrap();
        let id = *got[0].indices().unwrap().last().unwrap() as u32;
        past += next.len();
        next = vec![id];
        tokens.push(id);
        out.push(id);
    }
    out
}

#[test]
fn engine_matches_interpreter() {
    let c = Config::toy(1, 16);
    let g = stepper(&c);
    let weights = weights(&g, &mut Noise(11));
    let settings = settings::read();
    let mut engine = Engine::open(stepper(&c), 16, Pins::default(), &settings).unwrap();
    engine.load(&weights).unwrap();
    let prompt = [3u32, 9, 4];
    let want = reference(&g, &weights, &prompt, 6);
    let mut events = Vec::new();
    let request = Request {
        tokens: prompt.to_vec(),
        max_tokens: 6,
        stop: Vec::new(),
    };
    engine.generate(&request, &mut |e| events.push(e)).unwrap();
    let got: Vec<u32> = events
        .iter()
        .filter_map(|e| match e {
            Event::Token(id) => Some(*id),
            Event::Done(_) => None,
        })
        .collect();
    assert_eq!(got, want);
    assert_eq!(events.last(), Some(&Event::Done(Reason::Length)));
    // A stop id ends the answer, left out.
    let stopped = Request {
        stop: vec![want[2]],
        ..request.clone()
    };
    let mut events = Vec::new();
    engine.generate(&stopped, &mut |e| events.push(e)).unwrap();
    let tokens = events
        .iter()
        .filter(|e| matches!(e, Event::Token(_)))
        .count();
    assert_eq!(
        (tokens, events.last()),
        (2, Some(&Event::Done(Reason::Stop)))
    );
    // The context fills before a long answer ends.
    let long = Request {
        max_tokens: 100,
        ..request
    };
    let mut last = None;
    engine.generate(&long, &mut |e| last = Some(e)).unwrap();
    assert_eq!(last, Some(Event::Done(Reason::Context)));
}
