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
    for depth in [2, 4] {
        let lanes = vec![lane(Op::Sum, 0), lane(Op::Gather, 0)];
        let program: Vec<usize> = (0..24).map(|i| i % 2).collect();
        let groups = vec![vec![0, 1, 2, 3]];
        let pools = vec![pool(depth, false, 4)];
        let lag = vec![u64::from(depth / 2)];
        seeds(&Plan {
            pools,
            groups,
            lanes,
            programs: vec![program; 4],
            lag,
        })
        .unwrap();
    }
}

#[test]
fn mixed_pools_hold() {
    let pools = vec![
        pool(2, false, 2),
        pool(2, true, 2),
        pool(2, false, 1),
        pool(2, false, 1),
    ];
    let groups = vec![vec![0, 1], vec![0, 1], vec![0], vec![1]];
    let ops = [Op::AllToAll, Op::Send, Op::Recv, Op::Fetch, Op::Fetch];
    let lanes: Vec<Lane> = ops
        .iter()
        .zip([0, 1, 1, 2, 3])
        .map(|(&o, p)| lane(o, p))
        .collect();
    let a: Vec<usize> = (0..30).map(|i| [0, 1, 3][i % 3]).collect();
    let b: Vec<usize> = (0..30).map(|i| [0, 2, 4][i % 3]).collect();
    seeds(&Plan {
        pools,
        groups,
        lanes,
        programs: vec![a, b],
        lag: vec![1, 1, 2, 2],
    })
    .unwrap();
}

#[test]
fn broken_rule_caught() {
    let lanes = vec![lane(Op::Sum, 0)];
    let groups = vec![vec![0, 1]];
    let plan = Plan {
        pools: vec![pool(4, false, 2)],
        groups,
        lanes,
        programs: vec![vec![0; 24]; 2],
        lag: vec![3],
    };
    assert!(seeds(&plan).is_err());
}
