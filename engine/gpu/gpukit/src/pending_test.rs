use super::pending::{Pending, PendingOps};
use crate::Error;
use std::cell::Cell;
use std::rc::Rc;

struct Enc {
    id: usize,
    ends: Cell<usize>,
    barriers: Cell<usize>,
}

struct Cmd {
    id: usize,
    commits: Cell<usize>,
}

#[derive(Default)]
struct Ops {
    next_cmd: usize,
    next_enc: usize,
    fail_open: bool,
}

impl PendingOps for Ops {
    type Cmd = Rc<Cmd>;
    type Enc = Rc<Enc>;

    fn new_cmd(&mut self) -> Result<Self::Cmd, Error> {
        self.next_cmd += 1;
        Ok(Rc::new(Cmd {
            id: self.next_cmd,
            commits: Cell::new(0),
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
        enc.ends.set(enc.ends.get() + 1);
    }

    fn commit(&mut self, cmd: Self::Cmd) -> f64 {
        cmd.commits.set(cmd.commits.get() + 1);
        0.0
    }
}

#[test]
fn failed_switch_retry() {
    let (mut ops, mut slot) = (Ops::default(), None);
    let e1 = Pending::acquire(&mut slot, 1, false, &mut ops).unwrap();
    let c1 = slot.as_ref().unwrap().cmd.clone();
    ops.fail_open = true;
    assert!(Pending::acquire(&mut slot, 1, true, &mut ops).is_err());
    assert_eq!((e1.ends.get(), c1.commits.get()), (1, 0));
    assert!(Pending::acquire(&mut slot, 1, false, &mut ops).is_err());
    ops.fail_open = false;
    let e2 = Pending::acquire(&mut slot, 1, false, &mut ops).unwrap();
    assert_eq!(
        (e2.id, e2.ends.get(), c1.id),
        (2, 0, slot.as_ref().unwrap().cmd.id)
    );
    slot.take().unwrap().finish(&mut ops);
    assert_eq!((e1.ends.get(), e2.ends.get(), c1.commits.get()), (1, 1, 1));
    // Direct finish after failure commits without double-ending.
    let e3 = Pending::acquire(&mut slot, 1, true, &mut ops).unwrap();
    let c2 = slot.as_ref().unwrap().cmd.clone();
    let p = slot.as_mut().unwrap();
    p.dirty = true;
    p.fence(|e| e.barriers.set(e.barriers.get() + 1));
    p.fence(|e| e.barriers.set(e.barriers.get() + 1));
    assert_eq!(e3.barriers.get(), 1);
    ops.fail_open = true;
    assert!(Pending::acquire(&mut slot, 1, false, &mut ops).is_err());
    slot.take().unwrap().finish(&mut ops);
    assert_eq!((e3.ends.get(), c2.commits.get()), (1, 1));
}
