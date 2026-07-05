use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use rek0n_db::{
    AnnStrategy, ChunkRecord, DbError, DbLockOptions, Rek0nDb, SearchHit, SearchScope,
    DEFAULT_IVF_PROBE,
};
use tracing::{debug, instrument};

use crate::index_config::VectorIndexConfig;
use crate::lock::LockOptions;
use crate::model::LocalEmbedder;
use crate::types::{
    validate_embed_batch_size, validate_file_path, validate_index_batch, validate_search_limit,
    validate_table_name, ChunkKind, EmbedError, IndexedChunk, SearchResult,
    DEFAULT_EMBED_BATCH_SIZE, EMBEDDING_DIM,
};

const MIN_ROWS_FOR_IVF: usize = 256;

#[derive(Debug, Clone)]
pub struct IndexedBatch {
    pub chunks: Vec<IndexedChunk>,
    pub vectors: Vec<Vec<f32>>,
}

pub struct VectorStorage {
    db: Arc<Mutex<Rek0nDb>>,
    db_dir: String,
    repo_name: String,
    store_path: PathBuf,
    lock_options: LockOptions,
    vector_index_config: VectorIndexConfig,
}

impl VectorStorage {
    #[instrument(skip(db_dir), fields(repo = repo_name))]
    pub async fn initialize(db_dir: &str, repo_name: &str) -> Result<Self, EmbedError> {
        validate_table_name(repo_name)?;
        let store_path = PathBuf::from(db_dir).join(repo_name);
        let lock_options = LockOptions::default();
        let db = open_db(&store_path, lock_options).await?;
        debug!(db_dir, ?store_path, "opened rek0n-db store");
        Ok(Self {
            db: Arc::new(Mutex::new(db)),
            db_dir: db_dir.to_owned(),
            repo_name: repo_name.to_owned(),
            store_path,
            lock_options,
            vector_index_config: VectorIndexConfig::default(),
        })
    }

    pub fn with_lock_options(mut self, lock_options: LockOptions) -> Self {
        self.lock_options = lock_options;
        self
    }

    pub fn with_vector_index_config(mut self, vector_index_config: VectorIndexConfig) -> Self {
        self.vector_index_config = vector_index_config;
        self
    }

    pub fn lock_options(&self) -> LockOptions {
        self.lock_options
    }

    pub fn vector_index_config(&self) -> &VectorIndexConfig {
        &self.vector_index_config
    }

    pub fn db_dir(&self) -> &str {
        &self.db_dir
    }

    pub fn store_path(&self) -> &Path {
        &self.store_path
    }

    pub fn repo_name(&self) -> &str {
        &self.repo_name
    }

    #[instrument(skip(self, chunks, embedder), fields(repo = %self.repo_name, chunks = chunks.len()))]
    pub async fn index_chunks(
        &self,
        chunks: &[IndexedChunk],
        embedder: Arc<LocalEmbedder>,
    ) -> Result<(), EmbedError> {
        self.index_chunks_with_batch_size(chunks, embedder, DEFAULT_EMBED_BATCH_SIZE)
            .await
    }

    #[instrument(skip(self, chunks, embedder), fields(repo = %self.repo_name, chunks = chunks.len(), batch_size))]
    pub async fn index_chunks_with_batch_size(
        &self,
        chunks: &[IndexedChunk],
        embedder: Arc<LocalEmbedder>,
        batch_size: usize,
    ) -> Result<(), EmbedError> {
        if chunks.is_empty() {
            return Ok(());
        }

        validate_index_input(chunks, batch_size)?;

        let chunks = chunks.to_vec();
        let embedder = Arc::clone(&embedder);
        let batch = tokio::task::spawn_blocking(move || {
            build_indexed_batch(&chunks, &embedder, batch_size)
        })
        .await??;

        self.commit_batch(batch).await
    }

    pub async fn delete_by_file_path(&self, file_path: &str) -> Result<u64, EmbedError> {
        validate_file_path(file_path)?;
        let file_path = file_path.to_owned();
        let deleted = self
            .with_db_mut(move |db| db.delete_by_file_path(&file_path))
            .await?;
        debug!(deleted, "deleted rows by file_path");
        Ok(deleted as u64)
    }

    pub async fn reset_table(&self) -> Result<(), EmbedError> {
        self.with_db_mut(|db| db.reset()).await?;
        debug!("reset rek0n-db store");
        Ok(())
    }

    pub async fn replace_file_chunks(
        &self,
        file_path: &str,
        chunks: &[IndexedChunk],
        embedder: Arc<LocalEmbedder>,
    ) -> Result<(), EmbedError> {
        self.replace_file_chunks_with_batch_size(
            file_path,
            chunks,
            embedder,
            DEFAULT_EMBED_BATCH_SIZE,
        )
        .await
    }

    #[instrument(skip(self, chunks, embedder), fields(repo = %self.repo_name, file_path, chunks = chunks.len(), batch_size))]
    pub async fn replace_file_chunks_with_batch_size(
        &self,
        file_path: &str,
        chunks: &[IndexedChunk],
        embedder: Arc<LocalEmbedder>,
        batch_size: usize,
    ) -> Result<(), EmbedError> {
        validate_file_path(file_path)?;
        validate_chunks_for_file(file_path, chunks)?;
        if chunks.is_empty() {
            return self.delete_by_file_path(file_path).await.map(|_| ());
        }

        validate_index_input(chunks, batch_size)?;

        let chunks = chunks.to_vec();
        let embedder = Arc::clone(&embedder);
        let file_path = file_path.to_owned();
        let batch = tokio::task::spawn_blocking(move || {
            build_indexed_batch(&chunks, &embedder, batch_size)
        })
        .await??;

        self.replace_file_batch(&file_path, batch).await
    }

    pub async fn table_exists(&self) -> Result<bool, EmbedError> {
        let exists = self.with_db(|db| Ok(!db.is_empty())).await?;
        Ok(exists)
    }

    pub async fn count_rows(&self) -> Result<u64, EmbedError> {
        let count = self.with_db(|db| Ok(db.len())).await?;
        Ok(count as u64)
    }

    pub async fn count_rows_for_file(&self, file_path: &str) -> Result<u64, EmbedError> {
        validate_file_path(file_path)?;
        let file_path = file_path.to_owned();
        let count = self
            .with_db(move |db| Ok(db.count_rows_for_file(&file_path)))
            .await?;
        Ok(count as u64)
    }

    pub async fn is_empty(&self) -> Result<bool, EmbedError> {
        Ok(self.count_rows().await? == 0)
    }

    #[doc(hidden)]
    pub async fn index_record_batch(&self, batch: IndexedBatch) -> Result<(), EmbedError> {
        self.commit_batch(batch).await
    }

    #[doc(hidden)]
    pub async fn replace_file_record_batch(
        &self,
        file_path: &str,
        batch: IndexedBatch,
    ) -> Result<(), EmbedError> {
        validate_file_path(file_path)?;
        self.replace_file_batch(file_path, batch).await
    }

    #[instrument(skip(self, query_vector), fields(repo = %self.repo_name, limit))]
    pub async fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchResult>, EmbedError> {
        if query_vector.len() != EMBEDDING_DIM as usize {
            return Err(EmbedError::InvalidMetadata(format!(
                "query vector must be {EMBEDDING_DIM}-dimensional, got {}",
                query_vector.len()
            )));
        }

        let limit = validate_search_limit(limit)?;
        let query = query_vector.to_vec();
        let config = self.vector_index_config.clone();
        let hits = self
            .with_db(move |db| search_db(db, &query, limit, &config))
            .await?;

        let results: Vec<SearchResult> = hits.into_iter().map(hit_to_search_result).collect();
        debug!(hits = results.len(), "vector search complete");
        Ok(results)
    }

    async fn commit_batch(&self, batch: IndexedBatch) -> Result<(), EmbedError> {
        let config = self.vector_index_config.clone();
        self.with_db_mut(move |db| {
            for (vector, record) in batch.owned_pairs() {
                db.insert_persistent(&vector, &record)?;
            }
            maybe_build_ivf(db, &config)
        })
        .await
    }

    async fn replace_file_batch(
        &self,
        file_path: &str,
        batch: IndexedBatch,
    ) -> Result<(), EmbedError> {
        let file_path = file_path.to_owned();
        let config = self.vector_index_config.clone();
        let pairs = batch.owned_pairs();
        self.with_db_mut(move |db| {
            let refs: Vec<(&[f32], &ChunkRecord)> = pairs
                .iter()
                .map(|(vector, record)| (vector.as_slice(), record))
                .collect();
            db.replace_file(&file_path, &refs)?;
            maybe_build_ivf(db, &config)
        })
        .await
    }

    async fn with_db<F, T>(&self, f: F) -> Result<T, EmbedError>
    where
        F: FnOnce(&Rek0nDb) -> Result<T, DbError> + Send + 'static,
        T: Send + 'static,
    {
        let db = Arc::clone(&self.db);
        tokio::task::spawn_blocking(move || {
            let guard = lock_db(&db)?;
            f(&guard).map_err(EmbedError::from)
        })
        .await?
    }

    async fn with_db_mut<F, T>(&self, f: F) -> Result<T, EmbedError>
    where
        F: FnOnce(&mut Rek0nDb) -> Result<T, DbError> + Send + 'static,
        T: Send + 'static,
    {
        let db = Arc::clone(&self.db);
        tokio::task::spawn_blocking(move || {
            let mut guard = lock_db(&db)?;
            f(&mut guard).map_err(EmbedError::from)
        })
        .await?
    }
}

impl IndexedBatch {
    fn owned_pairs(&self) -> Vec<(Vec<f32>, ChunkRecord)> {
        self.chunks
            .iter()
            .zip(&self.vectors)
            .map(|(chunk, vector)| (vector.clone(), chunk_to_record(chunk)))
            .collect()
    }
}

async fn open_db(store_path: &Path, lock_options: LockOptions) -> Result<Rek0nDb, EmbedError> {
    let store_path = store_path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        Rek0nDb::open_with_options(store_path, to_db_lock_options(lock_options))
    })
    .await?
    .map_err(EmbedError::from)
}

fn lock_db(db: &Arc<Mutex<Rek0nDb>>) -> Result<MutexGuard<'_, Rek0nDb>, EmbedError> {
    db.lock()
        .map_err(|_| EmbedError::InvalidMetadata("rek0n-db lock poisoned".into()))
}

fn to_db_lock_options(options: LockOptions) -> DbLockOptions {
    if options == LockOptions::try_once() {
        DbLockOptions::try_exclusive_once()
    } else if options == LockOptions::blocking() {
        DbLockOptions::exclusive(crate::lock::DEFAULT_LOCK_TIMEOUT)
    } else {
        DbLockOptions::default()
    }
}

fn search_db(
    db: &Rek0nDb,
    query: &[f32],
    limit: usize,
    config: &VectorIndexConfig,
) -> Result<Vec<SearchHit>, DbError> {
    let live = db.live_persistent_count();
    let strategy = if db.has_ivf_index() {
        config.ann_strategy(live)
    } else {
        AnnStrategy::Exact
    };
    db.search_scoped(query, limit, SearchScope::all(), strategy)
}

fn maybe_build_ivf(db: &mut Rek0nDb, config: &VectorIndexConfig) -> Result<(), DbError> {
    let live = db.live_persistent_count();
    if live < MIN_ROWS_FOR_IVF || db.has_ivf_index() {
        return Ok(());
    }
    db.build_ivf_index(config.ivf_bucket_count(), DEFAULT_IVF_PROBE)
}

fn chunk_to_record(chunk: &IndexedChunk) -> ChunkRecord {
    ChunkRecord {
        kind: chunk.kind.as_str().to_string(),
        name: chunk.name.clone(),
        text: chunk.text.clone(),
        file_path: chunk.file_path.clone(),
        start_line: chunk.start_line as u64,
        end_line: chunk.end_line as u64,
    }
}

pub fn build_indexed_batch(
    chunks: &[IndexedChunk],
    embedder: &LocalEmbedder,
    batch_size: usize,
) -> Result<IndexedBatch, EmbedError> {
    validate_index_input(chunks, batch_size)?;

    let mut vectors = Vec::with_capacity(chunks.len());
    for chunk_batch in chunks.chunks(batch_size) {
        let batch_texts: Vec<&str> = chunk_batch
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect();
        vectors.extend(embedder.generate_embeddings(&batch_texts)?);
    }

    assemble_indexed_batch(chunks, &vectors)
}

#[doc(hidden)]
pub fn record_batch_from_vectors(
    chunks: &[IndexedChunk],
    vectors: &[Vec<f32>],
) -> Result<IndexedBatch, EmbedError> {
    if chunks.len() != vectors.len() {
        return Err(EmbedError::InvalidMetadata(format!(
            "chunk count {} does not match vector count {}",
            chunks.len(),
            vectors.len()
        )));
    }

    for chunk in chunks {
        chunk.validate().map_err(EmbedError::from)?;
    }

    for (index, vector) in vectors.iter().enumerate() {
        if vector.len() != EMBEDDING_DIM as usize {
            return Err(EmbedError::InvalidMetadata(format!(
                "vector at index {index} must be {EMBEDDING_DIM}-dimensional, got {}",
                vector.len()
            )));
        }
    }

    assemble_indexed_batch(chunks, vectors)
}

fn assemble_indexed_batch(
    chunks: &[IndexedChunk],
    vectors: &[Vec<f32>],
) -> Result<IndexedBatch, EmbedError> {
    Ok(IndexedBatch {
        chunks: chunks.to_vec(),
        vectors: vectors.to_vec(),
    })
}

fn hit_to_search_result(hit: SearchHit) -> SearchResult {
    SearchResult {
        score: hit.score,
        chunk: IndexedChunk {
            kind: ChunkKind::from_parser_kind(&hit.record.kind),
            name: hit.record.name,
            text: hit.record.text,
            start_line: hit.record.start_line as usize,
            end_line: hit.record.end_line as usize,
            file_path: hit.record.file_path,
        },
    }
}

fn validate_index_input(chunks: &[IndexedChunk], batch_size: usize) -> Result<(), EmbedError> {
    validate_index_batch(chunks)?;
    validate_embed_batch_size(batch_size)?;
    for chunk in chunks {
        chunk.validate().map_err(EmbedError::from)?;
    }
    Ok(())
}

fn validate_chunks_for_file(file_path: &str, chunks: &[IndexedChunk]) -> Result<(), EmbedError> {
    for chunk in chunks {
        if chunk.file_path != file_path {
            return Err(EmbedError::InvalidChunk(format!(
                "chunk file_path {:?} does not match target {file_path:?}",
                chunk.file_path
            )));
        }
    }
    Ok(())
}

impl From<DbError> for EmbedError {
    fn from(error: DbError) -> Self {
        match error {
            DbError::LockTimeout { path } => EmbedError::LockTimeout { path },
            DbError::Io { path, source } => EmbedError::io_path(path, source),
            DbError::Json(error) => EmbedError::Json(error),
            other => EmbedError::Database(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChunkKind;

    #[test]
    fn validate_chunks_for_file_rejects_mismatch() {
        let chunks = vec![IndexedChunk {
            kind: ChunkKind::Function,
            name: Some("foo".into()),
            text: "fn foo() {}".into(),
            start_line: 1,
            end_line: 1,
            file_path: "src/other.rs".into(),
        }];

        let err = validate_chunks_for_file("src/a.rs", &chunks).expect_err("mismatch");
        assert!(matches!(err, EmbedError::InvalidChunk(_)));
    }

    #[test]
    fn validate_index_input_rejects_late_invalid_chunk() {
        let chunks = vec![
            IndexedChunk {
                kind: ChunkKind::Function,
                name: Some("ok".into()),
                text: "fn ok() {}".into(),
                start_line: 1,
                end_line: 1,
                file_path: "src/a.rs".into(),
            },
            IndexedChunk {
                kind: ChunkKind::Function,
                name: None,
                text: "   ".into(),
                start_line: 2,
                end_line: 2,
                file_path: "src/a.rs".into(),
            },
        ];

        let err = validate_index_input(&chunks, DEFAULT_EMBED_BATCH_SIZE).expect_err("invalid");
        assert!(matches!(err, EmbedError::InvalidChunk(_)));
    }
}
