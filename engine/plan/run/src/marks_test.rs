use crate::marks::{ROOM, read};
use crate::profile::Marks;

#[test]
fn gaps_between_marks() {
    let mut records = vec![0u64; 2 * ROOM];
    records[..6].copy_from_slice(&[10, 100, 20, 400, 10, 1000]);
    let mut marks = Marks::default();
    read(&[ROOM as u32 + 2], &records, 1e9, &mut marks);
    let (first, second) = (&marks.regions[0], &marks.regions[1]);
    assert_eq!(
        (first.from, first.to, first.seconds.as_slice()),
        (10, 20, &[300e-9][..])
    );
    assert_eq!((second.from, second.to), (20, 10));
    assert_eq!(marks.lost, 2);
}
