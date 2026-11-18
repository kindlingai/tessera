use crate::ibv::{Context, Mr, PortAttr, Qp, QpAttr, QpInitAttr, SendWr, Sge, Wc};
use std::mem::{offset_of, size_of};

/// The sizes `infiniband/verbs.h` gives on LP64.
#[test]
fn header_layouts() {
    assert_eq!(size_of::<SendWr>(), 128);
    assert_eq!(size_of::<Sge>(), 16);
    assert_eq!(size_of::<Wc>(), 48);
    assert_eq!(size_of::<QpAttr>(), 144);
    assert_eq!(size_of::<QpInitAttr>(), 64);
    assert_eq!(size_of::<PortAttr>(), 64);
    assert_eq!(offset_of!(Context, ops), 8);
    assert_eq!(offset_of!(Qp, qp_num), 52);
    assert_eq!(offset_of!(Mr, lkey), 36);
    assert_eq!(offset_of!(Mr, rkey), 40);
    assert_eq!(offset_of!(SendWr, send_flags), 32);
    assert_eq!(offset_of!(SendWr, remote_addr), 40);
    assert_eq!(offset_of!(SendWr, rkey), 48);
    assert_eq!(offset_of!(QpAttr, ah_attr), 56);
    assert_eq!(offset_of!(QpAttr, max_rd_atomic), 126);
    assert_eq!(offset_of!(QpAttr, timeout), 130);
    assert_eq!(offset_of!(QpAttr, rate_limit), 136);
    assert_eq!(offset_of!(Wc, qp_num), 28);
}
