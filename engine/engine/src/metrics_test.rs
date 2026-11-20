use crate::metrics::text;

crate::metrics::registry!(TESTED = "tested");
crate::metrics::metric!(SEEN in TESTED = Counter "tessera:test_seen_total", "Calls seen.");

#[test]
fn text_lists_registries() {
    SEEN.add(2);
    SEEN.add(1);
    let out = text();
    let mine = "# HELP tessera:test_seen_total Calls seen.\n\
                # TYPE tessera:test_seen_total counter\n\
                tessera:test_seen_total 3\n";
    assert!(out.contains(mine));
    assert!(out.contains("# TYPE tessera:requests_running gauge"));
}
