use rek0n_embed::testing::record_batch_from_vectors;
use rek0n_embed::{ChunkKind, EmbedError, SemanticChunk, VectorStorage, EMBEDDING_DIM};

fn unit_vector(active: usize) -> Vec<f32> {
    let mut vector = vec![0.0_f32; EMBEDDING_DIM as usize];
    let index = active % vector.len();
    vector[index] = 1.0;
    vector
}

fn chunk(file_path: &str, text: &str, line: usize) -> SemanticChunk {
    SemanticChunk {
        kind: ChunkKind::Function,
        name: Some("demo".into()),
        text: text.into(),
        start_line: line,
        end_line: line,
        file_path: file_path.into(),
    }
}

#[tokio::test]
async fn indexes_searches_deletes_and_replaces_without_model() -> Result<(), EmbedError> {
    let temp = tempfile::tempdir().expect("tempdir");
    let db_dir = temp.path().join("lancedb");
    let storage = VectorStorage::initialize(db_dir.to_str().expect("utf8"), "itest").await?;

    assert!(!storage.table_exists().await?);
    assert_eq!(storage.count_rows().await?, 0);

    let auth_chunks = [
        chunk("src/auth.rs", "verify jwt token", 10),
        chunk("src/auth.rs", "hash password bcrypt", 20),
    ];
    let models_chunks = [chunk("src/models.rs", "struct User id email", 1)];
    let vectors = [unit_vector(0), unit_vector(1), unit_vector(2)];

    let batch = record_batch_from_vectors(
        &[
            auth_chunks[0].clone(),
            auth_chunks[1].clone(),
            models_chunks[0].clone(),
        ],
        &vectors,
    )?;
    storage.index_record_batch(batch).await?;

    assert!(storage.table_exists().await?);
    assert_eq!(storage.count_rows().await?, 3);
    assert_eq!(storage.count_rows_for_file("src/auth.rs").await?, 2);

    let hits = storage.search(&vectors[0], 2).await?;
    assert!(!hits.is_empty());
    assert_eq!(hits[0].chunk.file_path, "src/auth.rs");
    assert!(hits[0].chunk.text.contains("jwt"));

    let deleted = storage.delete_by_file_path("src/auth.rs").await?;
    assert_eq!(deleted, 2);
    assert_eq!(storage.count_rows().await?, 1);
    assert_eq!(storage.count_rows_for_file("src/auth.rs").await?, 0);

    let replacement = [chunk("src/auth.rs", "rotate session token", 40)];
    let replace_batch = record_batch_from_vectors(&replacement, &[unit_vector(40)])?;
    storage
        .replace_file_record_batch("src/auth.rs", replace_batch)
        .await?;

    assert_eq!(storage.count_rows().await?, 2);
    assert_eq!(storage.count_rows_for_file("src/auth.rs").await?, 1);

    let hits = storage.search(&unit_vector(40), 1).await?;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].chunk.start_line, 40);

    storage.reset_table().await?;
    assert!(!storage.table_exists().await?);
    assert!(storage.is_empty().await?);

    Ok(())
}
