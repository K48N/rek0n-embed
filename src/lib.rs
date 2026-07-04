//! Local embeddings and LanceDB vector search for [rek0n](https://github.com/K48N/rek0n).

mod db;
mod index_config;
mod lock;
mod model;
mod types;

use std::sync::Arc;

pub use db::VectorStorage;
pub use index_config::{IvfPqConfig, VectorDistance, VectorIndexConfig, VectorIndexKind};
pub use lock::{LockOptions, DEFAULT_LOCK_TIMEOUT};
pub use model::{generate_embedding_async, LocalEmbedder};
pub use types::{
    try_from_parser_chunk, try_from_parser_parts, validate_embed_batch_size, validate_file_path,
    validate_index_batch, validate_input_text_length, validate_query_text, validate_search_limit,
    validate_table_name, ChunkKind, EmbedError, SearchResult, SemanticChunk,
    DEFAULT_EMBED_BATCH_SIZE, EMBEDDING_DIM, MAX_EMBED_BATCH_SIZE, MAX_FILE_PATH_LEN,
    MAX_INDEX_BATCH_CHUNKS, MAX_INPUT_TEXT_LEN, MAX_QUERY_TEXT_LEN, MAX_SEARCH_LIMIT,
};

use tracing::instrument;

#[instrument(skip(storage, embedder, query_text), fields(limit))]
pub async fn query_semantic_context(
    storage: &VectorStorage,
    embedder: Arc<LocalEmbedder>,
    query_text: &str,
    limit: usize,
) -> Result<Vec<SearchResult>, EmbedError> {
    validate_query_text(query_text)?;
    let query_vector = generate_embedding_async(embedder, query_text).await?;
    storage.search(&query_vector, limit).await
}

#[doc(hidden)]
pub mod testing {
    pub use crate::db::record_batch_from_vectors;
}
