use crate::p2p::{CROSSOVER, SLOT_BYTES, Schedule, ack_offset, area_bytes, pick};
use crate::proxy::sim::{Plan, run};
use crate::proxy::{Lane, Op, Pool};
use crate::{Backend, Options, flatten};

fn pool(depth: u32, one_way: bool, group: u32) -> Pool {
    Pool {
        depth,
        one_way,
        group,
        me: 0,
    }
}

fn lane(op: Op, pool: u32) -> Lane {
    Lane {
        op,
        pool,
        width: 8,
        block: 2,
    }
}

fn seeds(plan: &Plan) -> Result<(), String> {
    (1..400).try_for_each(|seed| run(plan, seed))
}

#[test]
fn two_way_pools_hold() {
    for (depth, base, resync) in [(2, 1, None), (4, 1, None), (4, 65_530, Some(12))] {
        let plan = Plan {
            pools: vec![pool(depth, false, 4)],
            groups: vec![vec![0, 1, 2, 3]],
            lanes: vec![lane(Op::Sum, 0), lane(Op::Gather, 0)],
            programs: vec![(0..24).map(|i| i % 2).collect(); 4],
            lag: vec![u64::from(depth / 2)],
            base,
            resync,
        };
        seeds(&plan).unwrap();
    }
}

#[test]
fn mixed_pools_hold() {
    let ops = [Op::AllToAll, Op::Send, Op::Recv, Op::Fetch, Op::Fetch];
    let lanes = ops.iter().zip([0, 1, 1, 2, 3]).map(|(&o, p)| lane(o, p));
    let plan = Plan {
        pools: vec![
            pool(2, false, 2),
            pool(2, true, 2),
            pool(2, false, 1),
            pool(2, false, 1),
        ],
        groups: vec![vec![0, 1], vec![0, 1], vec![0], vec![1]],
        lanes: lanes.collect(),
        programs: vec![
            (0..30).map(|i| [0, 1, 3][i % 3]).collect(),
            (0..30).map(|i| [0, 2, 4][i % 3]).collect(),
        ],
        lag: vec![1, 1, 2, 2],
        base: 65_534,
        resync: Some(15),
    };
    seeds(&plan).unwrap();
}

#[test]
fn broken_rule_caught() {
    let plan = Plan {
        pools: vec![pool(4, false, 2)],
        groups: vec![vec![0, 1]],
        lanes: vec![lane(Op::Sum, 0)],
        programs: vec![vec![0; 24]; 2],
        lag: vec![3],
        base: 1,
        resync: None,
    };
    assert!(seeds(&plan).unwrap_err().contains("overwrote"));
}

#[test]
fn wired_puts_and_credits() {
    use crate::proxy::wired::{CREDITS, FLAGS, RECV, SEND, Wired, regions};
    use crate::proxy::{Link, Part, Sent};
    use tessera_engine_comm::{Link as _, Memory, Span};
    let pools = [pool(2, false, 2), pool(2, true, 2)];
    let bytes = [(32, 32), (32, 32)];
    let regions = regions(&pools, &bytes);
    let wire = |link| Wired::new(link, &pools, &bytes, vec![vec![0, 1], vec![0, 1]]).unwrap();
    let mut links = Memory::ranks(2, &regions).into_iter().map(wire);
    let (mut a, mut b) = (links.next().unwrap(), links.next().unwrap());
    a.link().write(Span::new(SEND, 32, 16), &[7; 16]).unwrap();
    let part = Part {
        pool: 0,
        slot: 1,
        to: 1,
        ordinal: 9,
        count: 2,
        from: 0,
        bytes: 16,
    };
    a.put(part).unwrap();
    a.credit(1, 1, 5).unwrap();
    assert_eq!(
        a.sent().unwrap(),
        [Sent {
            pool: 0,
            ordinal: 9,
            to: 1
        }]
    );
    let b = b.link();
    assert_eq!(b.read(Span::new(RECV, 8 * 32, 16)).unwrap(), [7; 16]);
    let flag = |at| b.load(Span::new(FLAGS, at, 8)).unwrap();
    assert_eq!((flag(8 * 16), flag(8 * 16 + 8)), (9, 2));
    assert_eq!(b.load(Span::new(CREDITS, 64, 8)).unwrap(), 5);
    let stray = Part { slot: 2, ..part };
    assert!(a.put(stray).is_err());
}

/// Emulates one rank's device: sums `rounds` parts with its peer.
fn device_rounds(page: &[std::sync::atomic::AtomicU64], regions: [usize; 3], me: u32, rounds: u64) {
    use crate::layout::{DOORBELL_SENT, DOORBELL_SLOTS, MAX_GROUP};
    use std::sync::atomic::Ordering::{Acquire, Release};
    let [send, recv, flags] = regions.map(|r| r as *mut u8);
    let peer = 1 - me as usize;
    for o in 1..=rounds {
        let slot = (o % 2) as usize;
        while o > 2 && page[DOORBELL_SENT / 8].load(Acquire) < o - 2 {
            std::hint::spin_loop();
        }
        let part = [o as u32 * 10 + me; 4];
        unsafe { std::ptr::copy_nonoverlapping(part.as_ptr().cast(), send.add(slot * 16), 16) };
        let word = 1 << 63 | (o & 0xFFFF) << 32 | 4;
        page[DOORBELL_SLOTS / 8 + slot * 2].store(word, Release);
        page[0].store(o, Release);
        let at = slot * MAX_GROUP + peer;
        let flag = unsafe { std::sync::atomic::AtomicU64::from_ptr(flags.add(at * 16).cast()) };
        while flag.load(Acquire) != o {
            std::hint::spin_loop();
        }
        let mut got = [0u32; 4];
        unsafe { std::ptr::copy_nonoverlapping(recv.add(at * 16), got.as_mut_ptr().cast(), 16) };
        assert_eq!(got, [o as u32 * 10 + peer as u32; 4], "rank {me} round {o}");
    }
}

#[test]
fn page_over_shm() {
    use crate::proxy::page::Page;
    use crate::proxy::running::Running;
    use crate::proxy::wired::{FLAGS, RECV, SEND, Wired, regions};
    use crate::proxy::{Lane, Proxy};
    use std::sync::atomic::AtomicU64;
    use tessera_engine_comm::{Link as _, ShmLink};
    let pools = [pool(2, false, 2)];
    let bytes = [(16, 16)];
    let regions = regions(&pools, &bytes);
    let path = tessera_testkit::scratch(&format!("page-{}", std::process::id()));
    let file = std::fs::File::create_new(&path).unwrap();
    file.set_len(ShmLink::room(2, &regions) as u64).unwrap();
    let lanes = vec![Lane {
        width: 4,
        ..lane(Op::Sum, 0)
    }];
    let ranks: Vec<_> = (0..2u32)
        .map(|r| {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            let link = ShmLink::over(r, 2, file, &regions).unwrap();
            let mut wired = Wired::new(
                link,
                &[Pool { me: r, ..pools[0] }],
                &bytes,
                vec![vec![0, 1]],
            )
            .unwrap();
            let mut base = |region| wired.link().memory(region).unwrap().0 as usize;
            let spans = [base(SEND), base(RECV), base(FLAGS)];
            let page: Box<[AtomicU64]> = (0..512).map(|_| AtomicU64::new(0)).collect();
            let mut host =
                unsafe { Page::new(page.as_ptr().cast_mut().cast(), &mut wired).unwrap() };
            let pools = vec![Pool { me: r, ..pools[0] }];
            let proxy = Proxy::new(&mut host, pools, lanes.clone(), 1).unwrap();
            (Running::spawn(proxy, host, wired), page, spans)
        })
        .collect();
    std::thread::scope(|s| {
        for (r, (_, page, spans)) in ranks.iter().enumerate() {
            s.spawn(move || device_rounds(page, *spans, r as u32, 40));
        }
    });
    std::fs::remove_file(&path).unwrap();
    for (running, ..) in ranks {
        running.stop().1.unwrap();
    }
}

#[test]
fn p2p_kernel_flattens() {
    let source = include_str!("p2p/p2p.kernel");
    for backend in [Backend::Cuda, Backend::Rocm] {
        let options = Options {
            backend,
            remote_kv: false,
            marks: false,
        };
        let flat = flatten("p2p.kernel", source, &|_| None, &options).unwrap();
        assert!(flat.text.contains("tessera_p2p_sum"));
    }
}

#[test]
fn p2p_layout_and_schedule() {
    assert_eq!(ack_offset(4), 8 * SLOT_BYTES);
    assert_eq!(area_bytes(4), 8 * SLOT_BYTES + 4096);
    assert_eq!(pick(12 << 10, 4, Schedule::Auto), (0, 1));
    assert_eq!(pick(CROSSOVER, 4, Schedule::Auto), (1, 4));
    assert_eq!(pick(CROSSOVER, 3, Schedule::Auto), (1, 4));
    assert_eq!(pick(CROSSOVER, 2, Schedule::Auto), (0, 4));
    assert_eq!(pick(1 << 20, 4, Schedule::Fanout), (0, 4));
    assert_eq!(pick(1024, 4, Schedule::Rounds), (1, 1));
}
