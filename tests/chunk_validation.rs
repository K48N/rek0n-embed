use rek0n_embed::{ChunkKind, IndexedChunk};

#[test]
fn valid_chunk_passes_validation() {
    let chunk = IndexedChunk {
        kind: ChunkKind::Struct,
        name: Some("User".into()),
        text: "pub struct User {}".into(),
        start_line: 1,
        end_line: 5,
        file_path: "src/models.rs".into(),
    };
    chunk.validate().expect("chunk should be valid");
}

#[test]
fn invalid_chunks_are_rejected() {
    let base = IndexedChunk {
        kind: ChunkKind::Function,
        name: None,
        text: "fn ok() {}".into(),
        start_line: 1,
        end_line: 1,
        file_path: "src/lib.rs".into(),
    };

    let mut empty_text = base.clone();
    empty_text.text = "".into();
    assert!(empty_text.validate().is_err());

    let mut empty_path = base.clone();
    empty_path.file_path = "  ".into();
    assert!(empty_path.validate().is_err());

    let mut bad_lines = base;
    bad_lines.end_line = 0;
    assert!(bad_lines.validate().is_err());
}
