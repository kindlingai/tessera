use super::pending::{Pending, PendingCmd, PendingEnc, PendingOps};
use crate::Error;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

struct Enc {
    id: usize,
    ends: Cell<usize>,
    barriers: Cell<usize>,
}

impl PendingEnc for Rc<Enc> {
    fn end_open(&self) {
        self.ends.set(self.ends.get() + 1);
    }
}

struct Cmd {
    id: usize,
    unretained: bool,
    commits: Cell<usize>,
    waits: Cell<usize>,
    times: Cell<(f64, f64)>,
    live: Rc<Cell<usize>>,
    log: Option<Rc<RefCell<Vec<&'static str>>>>,
}

impl PendingCmd for Rc<Cmd> {
    fn commit_open(&self) {
        self.commits.set(self.commits.get() + 1);
        self.live.set(self.live.get().saturating_sub(1));
        if let Some(log) = &self.log {
            log.borrow_mut().push("C");
        }
    }

    fn wait_done(&self) {
        self.waits.set(self.waits.get() + 1);
        if let Some(log) = &self.log {
            log.borrow_mut().push("W");
        }
    }
}

impl Drop for Cmd {
    fn drop(&mut self) {
        if self.commits.get() == 0 {
            self.live.set(self.live.get().saturating_sub(1));
        }
    }
}

struct Kept(Rc<RefCell<Vec<&'static str>>>);

impl Drop for Kept {
    fn drop(&mut self) {
        self.0.borrow_mut().push("K");
    }
}

struct Ops {
    next_cmd: usize,
    next_enc: usize,
    fail_cmd: bool,
    fail_open: bool,
    cap: usize,
    live: Rc<Cell<usize>>,
    next_times: (f64, f64),
    log: Option<Rc<RefCell<Vec<&'static str>>>>,
}

impl Default for Ops {
    fn default() -> Self {
        Self {
            next_cmd: 0,
            next_enc: 0,
            fail_cmd: false,
            fail_open: false,
            cap: usize::MAX,
            live: Rc::new(Cell::new(0)),
            next_times: (0.0, 0.0),
            log: None,
        }
    }
}

impl Ops {
    fn logged(log: &Rc<RefCell<Vec<&'static str>>>, times: (f64, f64), cap: usize) -> Self {
        Self {
            log: Some(log.clone()),
            next_times: times,
            cap,
            ..Self::default()
        }
    }
}

impl PendingOps for Ops {
    type Cmd = Rc<Cmd>;
    type Enc = Rc<Enc>;

    fn new_cmd(&mut self, unretained: bool) -> Result<Self::Cmd, Error> {
        if self.fail_cmd || self.live.get() >= self.cap {
            return Err(Error::Gpu("command buffer"));
        }
        self.next_cmd += 1;
        self.live.set(self.live.get() + 1);
        if let Some(log) = &self.log {
            log.borrow_mut().push("N");
        }
        Ok(Rc::new(Cmd {
            id: self.next_cmd,
            unretained,
            commits: Cell::new(0),
            waits: Cell::new(0),
            times: Cell::new(self.next_times),
            live: self.live.clone(),
            log: self.log.clone(),
        }))
    }

    fn open_enc(&mut self, _cmd: &Self::Cmd, _concurrent: bool) -> Result<Self::Enc, Error> {
        if self.fail_open {
            return Err(Error::Gpu("encoder"));
        }
        self.next_enc += 1;
        Ok(Rc::new(Enc {
            id: self.next_enc,
            ends: Cell::new(0),
            barriers: Cell::new(0),
        }))
    }

    fn end_enc(&mut self, enc: Self::Enc) {
        enc.end_open();
    }

    fn commit(&mut self, cmd: &Self::Cmd) {
        cmd.commit_open();
    }

    fn wait(&mut self, cmd: &Self::Cmd) -> (f64, f64) {
        cmd.wait_done();
        cmd.times.get()
    }
}

type MockPending<K = ()> = Pending<Rc<Cmd>, Rc<Enc>, K>;

fn cm(c: &Rc<Cmd>) -> usize {
    c.commits.get()
}

fn cmd_of<K>(p: &MockPending<K>) -> Rc<Cmd> {
    p.cmd.clone().unwrap()
}

#[test]
fn failed_switch_retry() {
    let (mut ops, mut slot): (Ops, Option<MockPending>) = (Ops::default(), None);
    let e1 = Pending::acquire(&mut slot, 1, false, &mut ops).unwrap();
    let c1 = cmd_of(slot.as_ref().unwrap());
    ops.fail_open = true;
    assert!(Pending::acquire(&mut slot, 1, true, &mut ops).is_err());
    assert_eq!((e1.ends.get(), cm(&c1)), (1, 0));
    assert!(Pending::acquire(&mut slot, 1, false, &mut ops).is_err());
    ops.fail_open = false;
    let e2 = Pending::acquire(&mut slot, 1, false, &mut ops).unwrap();
    assert_eq!(
        (e2.id, e2.ends.get(), c1.id),
        (2, 0, cmd_of(slot.as_ref().unwrap()).id)
    );
    slot.take().unwrap().finish(&mut ops);
    assert_eq!((e1.ends.get(), e2.ends.get(), cm(&c1)), (1, 1, 1));
    let e3 = Pending::acquire(&mut slot, 1, true, &mut ops).unwrap();
    let c2 = cmd_of(slot.as_ref().unwrap());
    let p = slot.as_mut().unwrap();
    p.dirty = true;
    p.fence(|e| e.barriers.set(e.barriers.get() + 1));
    p.fence(|e| e.barriers.set(e.barriers.get() + 1));
    assert_eq!(e3.barriers.get(), 1);
    ops.fail_open = true;
    assert!(Pending::acquire(&mut slot, 1, false, &mut ops).is_err());
    slot.take().unwrap().finish(&mut ops);
    assert_eq!((e3.ends.get(), cm(&c2)), (1, 1));
}

#[test]
fn rotate_and_drop() {
    let log = Rc::new(RefCell::new(Vec::new()));
    let mut ops = Ops::logged(&log, (10.0, 10.25), 1);
    let mut p: MockPending<Kept> = Pending::open(1, true, true, &mut ops).unwrap();
    let c1 = cmd_of(&p);
    p.mark_dispatch();
    ops.fail_cmd = true;
    assert!(p.rotate(true, true, &mut ops).is_err());
    assert_eq!((cm(&c1), p.flushed.len(), p.cmd.is_none()), (1, 1, true));
    ops.fail_cmd = false;
    ops.next_times = (10.30, 10.75);
    p.ensure_mode(true, true, &mut ops).unwrap();
    let c2 = cmd_of(&p);
    p.mark_dispatch();
    ops.next_times = (0.0, 0.0);
    p.rotate(true, true, &mut ops).unwrap();
    let c3 = cmd_of(&p);
    assert!((p.finish(&mut ops) - 0.75).abs() < 1e-9);
    assert_eq!([cm(&c1), cm(&c2), cm(&c3)], [1, 1, 0]);
    drop(c3);
    log.borrow_mut().clear();
    let mut slot: Option<MockPending<Kept>> = Some(Pending::open(1, true, true, &mut ops).unwrap());
    let p = slot.as_mut().unwrap();
    p.keep.push(Kept(log.clone()));
    for _ in 0..64 {
        p.mark_dispatch();
    }
    let pre = cmd_of(p);
    p.rotate(true, true, &mut ops).unwrap();
    let tail = cmd_of(p);
    assert!(pre.unretained && tail.unretained);
    p.mark_dispatch();
    drop(slot);
    assert_eq!(
        [cm(&pre), cm(&tail), pre.waits.get(), tail.waits.get()],
        [1, 1, 1, 1]
    );
    assert_eq!(*log.borrow(), ["N", "C", "N", "C", "W", "W", "K"]);
}
