use crate::numa::{intersect_cpus, parse_list, parse_node, pci_bdf};

#[test]
fn parses_numa_sysfs() {
    assert_eq!(pci_bdf("00000000:C1:00.0").as_deref(), Some("0000:c1:00.0"));
    assert_eq!(pci_bdf("0000:01:00.1\n").as_deref(), Some("0000:01:00.1"));
    assert_eq!(
        pci_bdf("00010000:02:00.0").as_deref(),
        Some("10000:02:00.0")
    );
    assert_eq!(pci_bdf("bad"), None);

    assert_eq!(parse_node("-1\n"), None);
    assert_eq!(parse_node("2\n"), Some(2));
    assert_eq!(parse_list("0-3,8,10-11\n"), vec![0, 1, 2, 3, 8, 10, 11]);
    assert_eq!(parse_list("0\n"), vec![0]);

    let all = vec![0, 1, 2, 3, 4, 5, 6, 7];
    assert_eq!(intersect_cpus(&all, &[2, 3]), Some(vec![2, 3]));
    assert_eq!(intersect_cpus(&[2, 3], &[2, 3]), None);
    assert_eq!(intersect_cpus(&[0, 1], &[2, 3]), None);
}
