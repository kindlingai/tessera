use crate::teleport::{platform, quote};

#[test]
fn maps_uname() {
    let arm = platform("Linux aarch64\n");
    assert_eq!(arm, Some(("aarch64-linux", "tesserax-linux-arm64")));
    assert_eq!(platform("Linux riscv64"), None);
    assert_eq!(quote("a'b"), r"'a'\''b'");
}
