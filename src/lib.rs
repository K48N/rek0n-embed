//! Local embeddings and LanceDB vector search for [rek0n](https://github.com/K48N/rek0n).

mod db;
mod index_config;
mod lock;
mod model;
mod types;

use std::sync::Arc;

pub use db::VectorStorage;
pub use index_config::{IvfPqConfig, VectorDistance, VectorIndexConfig, VectorIndexKind};
pub use lock::LockOptions;
pub use model::{generate_embedding_async, LocalEmbedder};
pub use types::{
    validate_embed_batch_size, validate_file_path, validate_input_text_length,
    validate_search_limit, validate_table_name, ChunkKind, EmbedError, SearchResult, SemanticChunk,
    DEFAULT_EMBED_BATCH_SIZE, EMBEDDING_DIM, MAX_INPUT_TEXT_LEN,
};

use tracing::instrument;

#[instrument(skip(storage, embedder, query_text), fields(limit))]
pub async fn query_semantic_context(
    storage: &VectorStorage,
    embedder: Arc<LocalEmbedder>,
    query_text: &str,
    limit: usize,
) -> Result<Vec<SearchResult>, EmbedError> {
    let query_vector = generate_embedding_async(embedder, query_text).await?;
    storage.search(&query_vector, limit).await
}

#[doc(hidden)]
pub mod testing {
    pub use crate::db::record_batch_from_vectors;
}
