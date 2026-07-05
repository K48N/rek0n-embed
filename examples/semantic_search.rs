use std::path::PathBuf;
use std::sync::Arc;

use rek0n_embed::{query_semantic_context, ChunkKind, IndexedChunk, LocalEmbedder, VectorStorage};

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/model")
}

fn ensure_weights_present(model_dir: &std::path::Path) -> Result<(), String> {
    let weights = model_dir.join("model.safetensors");
    if weights.is_file() {
        return Ok(());
    }
    Err(format!(
        "missing {}\nrun scripts/download_model.ps1 or scripts/download_model.sh",
        weights.display()
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = model_dir();
    ensure_weights_present(&model_dir)
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::NotFound, message))?;

    let embedder = Arc::new(LocalEmbedder::new(
        &model_dir.join("model.safetensors"),
        &model_dir.join("tokenizer.json"),
    )?);

    let auth_chunks = vec![
        IndexedChunk {
            kind: ChunkKind::Function,
            name: Some("authenticate".into()),
            text: "pub fn authenticate(token: &str) -> Result<User, AuthError> { verify_jwt(token) }".into(),
            start_line: 10,
            end_line: 12,
            file_path: "src/auth.rs".into(),
        },
        IndexedChunk {
            kind: ChunkKind::Function,
            name: Some("hash_password".into()),
            text: "pub fn hash_password(password: &str) -> String { bcrypt::hash(password, DEFAULT_COST) }".into(),
            start_line: 20,
            end_line: 22,
            file_path: "src/auth.rs".into(),
        },
    ];

    let models_chunks = vec![IndexedChunk {
        kind: ChunkKind::Struct,
        name: Some("User".into()),
        text: "pub struct User { pub id: Uuid, pub email: String }".into(),
        start_line: 1,
        end_line: 3,
        file_path: "src/models.rs".into(),
    }];

    let db_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/data/lancedb");
    let storage =
        VectorStorage::initialize(db_dir.to_str().ok_or("invalid db path")?, "demo").await?;

    println!("Indexing src/auth.rs …");
    storage
        .replace_file_chunks("src/auth.rs", &auth_chunks, Arc::clone(&embedder))
        .await?;

    println!("Indexing src/models.rs …");
    storage
        .replace_file_chunks("src/models.rs", &models_chunks, Arc::clone(&embedder))
        .await?;

    println!(
        "Table has {} rows ({} in auth.rs)\n",
        storage.count_rows().await?,
        storage.count_rows_for_file("src/auth.rs").await?
    );

    let query = "verify user login token";
    let results = query_semantic_context(&storage, embedder, query, 3).await?;

    println!("Query: {query}\n");
    for (rank, hit) in results.iter().enumerate() {
        println!(
            "#{rank} score={:.4} {}:{} {} {:?}",
            hit.score,
            hit.chunk.file_path,
            hit.chunk.line_range(),
            hit.chunk.kind.as_str(),
            hit.chunk.name
        );
        println!("   {}", hit.chunk.text);
        println!();
    }

    Ok(())
}
