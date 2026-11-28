use tessera_device::proxy::{
    Error, Fetched, Fetcher, Host, Lane, Link, Op, Part, Pool, Proxy, Sent, Word,
};
use tessera_engine_comm::Error as CommError;

#[derive(Default)]
struct StubHost {
    epoch: u64,
    triggers: [u64; 4],
    sent: u64,
    aborted: bool,
}

impl Host for StubHost {
    fn load(&self, word: Word) -> u64 {
        match word {
            Word::Epoch => self.epoch,
            Word::Trigger { slot, .. } => self.triggers[slot as usize],
            _ => 0,
        }
    }
    fn store(&mut self, word: Word, value: u64) {
        match word {
            Word::Sent { .. } => self.sent = value,
            Word::Abort => self.aborted = value != 0,
            _ => {}
        }
    }
}

#[derive(Default)]
struct StubLink {
    puts: Vec<(u64, u32)>,
    ack: Vec<Sent>,
}

impl Link for StubLink {
    fn put(&mut self, part: Part) -> Result<(), CommError> {
        self.puts.push((part.ordinal, part.to));
        Ok(())
    }
    fn credit(&mut self, _: u32, _: u32, _: u64) -> Result<(), CommError> {
        Ok(())
    }
    fn sent(&mut self) -> Result<Vec<Sent>, CommError> {
        Ok(std::mem::take(&mut self.ack))
    }
    fn reset(&mut self) -> Result<(), CommError> {
        self.puts.clear();
        self.ack.clear();
        Ok(())
    }
}

struct NoFetch;
impl Fetcher for NoFetch {
    fn fetch(&mut self, _: u32, _: u32, _: u64, _: u32) -> Result<(), String> {
        Ok(())
    }
    fn fetched(&mut self) -> Vec<Fetched> {
        Vec::new()
    }
    fn reset(&mut self) {}
}

fn trigger_word(lane: u64, ordinal: u64) -> u64 {
    (1u64 << 63) | (lane << 48) | ((ordinal & 0xFFFF) << 32) | 4
}

#[test]
fn unordered_events() {
    let pool = Pool {
        depth: 4,
        one_way: false,
        group: 8,
        me: 3,
    };
    let lanes = vec![
        Lane {
            op: Op::Sum,
            pool: 0,
            width: 4,
            block: 1,
        },
        Lane {
            op: Op::Recv,
            pool: 0,
            width: 4,
            block: 1,
        },
    ];
    let mut host = StubHost::default();
    let mut link = StubLink::default();
    let mut fetch = NoFetch;
    let mut proxy = Proxy::new(&mut host, vec![pool], lanes, 10).unwrap();
    // Trigger ordinals 10 (Sum) and 11 (Recv) back-to-back.
    host.triggers[10 % 4] = trigger_word(0, 10);
    host.triggers[11 % 4] = trigger_word(1, 11);
    host.epoch = 1;
    assert!(proxy.step(&mut host, &mut link, &mut fetch).unwrap());
    assert_eq!(
        link.puts,
        [0, 1, 2, 4, 5, 6, 7]
            .into_iter()
            .map(|r| (10, r))
            .collect::<Vec<_>>()
    );
    // Ordinal 11 has zero targets, but stays blocked behind 10.
    assert_eq!(host.sent, 9);
    assert!(proxy.open());
    // Finishing ordinal 10 peers releases 10 and 11.
    link.ack = [7, 5, 2, 0, 6, 4, 1]
        .into_iter()
        .map(|to| Sent {
            pool: 0,
            ordinal: 10,
            to,
        })
        .collect();
    assert!(proxy.step(&mut host, &mut link, &mut fetch).unwrap());
    assert_eq!(host.sent, 11);
    assert!(!proxy.open());
    // Stray events (released ordinal, out-of-range rank, future ordinal) fail.
    for bad in [
        Sent {
            pool: 0,
            ordinal: 10,
            to: 0,
        },
        Sent {
            pool: 0,
            ordinal: 12,
            to: 0,
        },
        Sent {
            pool: 0,
            ordinal: 11,
            to: 64,
        },
    ] {
        host.aborted = false;
        link.ack = vec![bad];
        let err = proxy.step(&mut host, &mut link, &mut fetch).unwrap_err();
        assert_eq!(
            err,
            Error::Event {
                pool: bad.pool,
                ordinal: bad.ordinal,
            }
        );
        assert!(host.aborted);
    }
}
