use crate::platform::{TABLE, row_for, rows};

#[test]
fn files_are_rows() {
    let files = TABLE
        .files()
        .filter(|f| f.path().extension().is_some_and(|e| e == "json"))
        .count();
    assert!(files > 0);
    assert_eq!(rows().len(), files);
}

#[test]
fn table_resolves() {
    let tile = |id: &str| row_for(id).and_then(|r| r.tile);
    assert_eq!(tile("cuda sm_121"), Some([16, 8, 8]));
    assert_eq!(tile("cuda sm_90"), Some([16, 8, 8]));
    assert_eq!(tile("cuda sm_75"), None, "a row with no usable tile");
    assert!(row_for("cuda sm_75").is_some());
    assert!(row_for("cuda sm_61").is_none(), "no row at all");
    assert_eq!(tile("rocm gfx11_5"), Some([16, 16, 16]));
    assert_eq!(tile("metal applegpu_g15p"), Some([8, 8, 8]));
    assert_eq!(tile("metal apple9"), Some([8, 8, 8]));
    assert!(row_for("cuda sm_90").is_some_and(|r| r.pdl));
    assert!(row_for("cuda sm_89").is_some_and(|r| !r.pdl));
    for row in rows() {
        assert!(!row.matches.is_empty(), "{row:?}");
    }
}
