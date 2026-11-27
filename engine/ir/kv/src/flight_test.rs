use crate::flight::Flights;

#[test]
fn flights_hold_pages() {
    let mut flights = Flights::default();
    flights.depart(7, 1, 4);
    flights.depart(9, 2, 5);
    // A freed page holds both its pages until it lands.
    flights.free(9);
    let landed = flights.land();
    let fields: Vec<_> = landed
        .iter()
        .map(|l| (l.page, l.vram, l.host, l.live))
        .collect();
    assert_eq!(fields, [(7, 1, 4, true), (9, 2, 5, false)]);
    assert!(flights.is_empty());
}
