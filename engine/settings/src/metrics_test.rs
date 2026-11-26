use crate::metrics::text;

crate::declare_registry!(TESTED = "tested");
crate::declare_metric!(SEEN in TESTED = Counter "tessera:test_seen_total", "Calls seen.");
crate::declare_metric!(OPEN in TESTED = Gauge "tessera:test_open", "Open calls.");

#[test]
fn text_lists_registries() {
    SEEN.add(2);
    SEEN.add(1);
    OPEN.set(4);
    let out = text();
    let mine = "# HELP tessera:test_seen_total Calls seen.\n\
                # TYPE tessera:test_seen_total counter\n\
                tessera:test_seen_total 3\n";
    assert!(out.contains(mine));
    assert!(out.contains("# TYPE tessera:test_open gauge\ntessera:test_open 4\n"));
}
