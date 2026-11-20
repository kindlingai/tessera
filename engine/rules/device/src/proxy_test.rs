use crate::proxy::sim::{Plan, run};
use crate::proxy::{Lane, Op, Pool};

fn pool(depth: u32, one_way: bool, group: u32) -> Pool {
    Pool {
        depth,
        one_way,
        group,
        me: 0,
    }
}

fn lane(op: Op, pool: u32) -> Lane {
    Lane { op, pool }
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
