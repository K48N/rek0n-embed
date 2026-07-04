use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures_util::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::Connection;
use tracing::{debug, instrument, warn};

use crate::index_config::{to_lancedb_index, VectorIndexConfig};
use crate::lock::{acquire_exclusive, acquire_shared, lock_path, LockOptions};
use crate::model::LocalEmbedder;
use crate::types::{
    validate_embed_batch_size, validate_file_path, validate_index_batch, validate_search_limit,
    validate_table_name, EmbedError, SearchResult, SemanticChunk, DEFAULT_EMBED_BATCH_SIZE,
    EMBEDDING_DIM,
};
use lancedb::table::{OptimizeAction, OptimizeOptions};
use lancedb::DistanceType;

const MIN_ROWS_FOR_VECTOR_INDEX: usize = 256;

pub struct VectorStorage {
    connection: Connection,
    db_dir: String,
    repo_name: String,
    lock_options: LockOptions,
    vector_index_config: VectorIndexConfig,
}

impl VectorStorage {
    #[instrument(skip(db_dir), fields(repo = repo_name))]
    pub async fn initialize(db_dir: &str, repo_name: &str) -> Result<Self, EmbedError> {
        validate_table_name(repo_name)?;
        let connection = lancedb::connect(db_dir).execute().await?;
        debug!(db_dir, "connected to LanceDB");
        Ok(Self {
            connection,
            db_dir: db_dir.to_owned(),
            repo_name: repo_name.to_owned(),
            lock_options: LockOptions::default(),
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

    pub fn connection(&self) -> &Connection {
        &self.connection
    }

    pub fn repo_name(&self) -> &str {
        &self.repo_name
    }

    #[instrument(skip(self, chunks, embedder), fields(repo = %self.repo_name, chunks = chunks.len()))]
    pub async fn index_chunks(
        &self,
        chunks: &[SemanticChunk],
        embedder: Arc<LocalEmbedder>,
    ) -> Result<(), EmbedError> {
        self.index_chunks_with_batch_size(chunks, embedder, DEFAULT_EMBED_BATCH_SIZE)
            .await
    }

    #[instrument(skip(self, chunks, embedder), fields(repo = %self.repo_name, chunks = chunks.len(), batch_size))]
    pub async fn index_chunks_with_batch_size(
        &self,
        chunks: &[SemanticChunk],
        embedder: Arc<LocalEmbedder>,
        batch_size: usize,
    ) -> Result<(), EmbedError> {
        if chunks.is_empty() {
            return Ok(());
        }

        validate_index_input(chunks, batch_size)?;

        let chunks = chunks.to_vec();
        let embedder = Arc::clone(&embedder);
        let batch =
            tokio::task::spawn_blocking(move || build_record_batch(&chunks, &embedder, batch_size))
                .await??;

        self.commit_batch(batch).await
    }

    pub async fn delete_by_file_path(&self, file_path: &str) -> Result<u64, EmbedError> {
        validate_file_path(file_path)?;

        let _lock =
            acquire_exclusive(lock_path(&self.db_dir, &self.repo_name), self.lock_options).await?;

        let Some(table) = self.try_open_table().await? else {
            return Ok(0);
        };

        let result = table.delete(&file_path_predicate(file_path)).await?;
        if result.num_deleted_rows > 0 {
            refresh_vector_index(&table, &self.vector_index_config, IndexRefreshMode::Retrain)
                .await?;
        }
        debug!(
            deleted = result.num_deleted_rows,
            "deleted rows by file_path"
        );
        Ok(result.num_deleted_rows)
    }

    pub async fn reset_table(&self) -> Result<(), EmbedError> {
        let _lock =
            acquire_exclusive(lock_path(&self.db_dir, &self.repo_name), self.lock_options).await?;

        match self.connection.drop_table(&self.repo_name, &[]).await {
            Ok(()) => {
                debug!("dropped LanceDB table");
                Ok(())
            }
            Err(error) if is_table_not_found(&error) => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    pub async fn replace_file_chunks(
        &self,
        file_path: &str,
        chunks: &[SemanticChunk],
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
        chunks: &[SemanticChunk],
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
        let batch =
            tokio::task::spawn_blocking(move || build_record_batch(&chunks, &embedder, batch_size))
                .await??;

        self.replace_file_batch(file_path, batch).await
    }

    pub async fn table_exists(&self) -> Result<bool, EmbedError> {
        let _lock =
            acquire_shared(lock_path(&self.db_dir, &self.repo_name), self.lock_options).await?;
        Ok(self.try_open_table().await?.is_some())
    }

    pub async fn count_rows(&self) -> Result<u64, EmbedError> {
        let _lock =
            acquire_shared(lock_path(&self.db_dir, &self.repo_name), self.lock_options).await?;

        let Some(table) = self.try_open_table().await? else {
            return Ok(0);
        };

        Ok(table.count_rows(None).await? as u64)
    }

    pub async fn count_rows_for_file(&self, file_path: &str) -> Result<u64, EmbedError> {
        validate_file_path(file_path)?;

        let _lock =
            acquire_shared(lock_path(&self.db_dir, &self.repo_name), self.lock_options).await?;

        let Some(table) = self.try_open_table().await? else {
            return Ok(0);
        };

        Ok(table
            .count_rows(Some(file_path_predicate(file_path)))
            .await? as u64)
    }

    pub async fn is_empty(&self) -> Result<bool, EmbedError> {
        Ok(self.count_rows().await? == 0)
    }

    #[doc(hidden)]
    pub async fn index_record_batch(&self, batch: RecordBatch) -> Result<(), EmbedError> {
        self.commit_batch(batch).await
    }

    #[doc(hidden)]
    pub async fn replace_file_record_batch(
        &self,
        file_path: &str,
        batch: RecordBatch,
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
        let _lock =
            acquire_shared(lock_path(&self.db_dir, &self.repo_name), self.lock_options).await?;

        let Some(table) = self.try_open_table().await? else {
            return Ok(Vec::new());
        };

        let distance: DistanceType = self.vector_index_config.search_distance().into();
        let stream = table
            .query()
            .nearest_to(query_vector)?
            .distance_type(distance)
            .limit(limit)
            .execute()
            .await?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        let mut results = Vec::new();

        for batch in batches {
            results.extend(parse_search_batch(&batch)?);
        }

        debug!(hits = results.len(), "vector search complete");
        Ok(results)
    }

    async fn commit_batch(&self, batch: RecordBatch) -> Result<(), EmbedError> {
        let _lock =
            acquire_exclusive(lock_path(&self.db_dir, &self.repo_name), self.lock_options).await?;
        let table = self.open_or_create_table(batch).await?;
        refresh_vector_index(&table, &self.vector_index_config, IndexRefreshMode::Append).await
    }

    async fn replace_file_batch(
        &self,
        file_path: &str,
        batch: RecordBatch,
    ) -> Result<(), EmbedError> {
        let _lock =
            acquire_exclusive(lock_path(&self.db_dir, &self.repo_name), self.lock_options).await?;

        if let Some(table) = self.try_open_table().await? {
            table.delete(&file_path_predicate(file_path)).await?;
        }

        let table = self.open_or_create_table(batch).await?;
        refresh_vector_index(&table, &self.vector_index_config, IndexRefreshMode::Retrain).await
    }

    async fn open_or_create_table(&self, batch: RecordBatch) -> Result<lancedb::Table, EmbedError> {
        match self
            .connection
            .open_table(self.repo_name.clone())
            .execute()
            .await
        {
            Ok(table) => {
                table.add(batch).execute().await?;
                Ok(table)
            }
            Err(error) if is_table_not_found(&error) => {
                match self
                    .connection
                    .create_table(self.repo_name.clone(), batch.clone())
                    .execute()
                    .await
                {
                    Ok(table) => Ok(table),
                    Err(error) if is_table_already_exists(&error) => {
                        let table = self
                            .connection
                            .open_table(self.repo_name.clone())
                            .execute()
                            .await?;
                        table.add(batch).execute().await?;
                        Ok(table)
                    }
                    Err(error) => Err(error.into()),
                }
            }
            Err(error) => Err(error.into()),
        }
    }

    async fn try_open_table(&self) -> Result<Option<lancedb::Table>, EmbedError> {
        match self
            .connection
            .open_table(self.repo_name.clone())
            .execute()
            .await
        {
            Ok(table) => Ok(Some(table)),
            Err(error) if is_table_not_found(&error) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }
}

enum IndexRefreshMode {
    Append,
    Retrain,
}

async fn refresh_vector_index(
    table: &lancedb::Table,
    vector_index_config: &VectorIndexConfig,
    mode: IndexRefreshMode,
) -> Result<(), EmbedError> {
    let row_count = table.count_rows(None).await?;
    let has_index = vector_index_exists(table).await?;

    if row_count < MIN_ROWS_FOR_VECTOR_INDEX {
        if has_index {
            drop_vector_indices(table).await?;
        }
        return Ok(());
    }

    if !has_index {
        return create_vector_index_if_needed(table, vector_index_config).await;
    }

    let options = match mode {
        IndexRefreshMode::Append => OptimizeOptions::merge(0),
        IndexRefreshMode::Retrain => OptimizeOptions::retrain(),
    };

    match table.optimize(OptimizeAction::Index(options)).await {
        Ok(_) => Ok(()),
        Err(error) if is_skippable_index_error(&error) => {
            warn!(error = %error, "skipped vector index refresh");
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

async fn create_vector_index_if_needed(
    table: &lancedb::Table,
    vector_index_config: &VectorIndexConfig,
) -> Result<(), EmbedError> {
    if vector_index_exists(table).await? {
        return Ok(());
    }

    let row_count = table.count_rows(None).await?;
    if row_count < MIN_ROWS_FOR_VECTOR_INDEX {
        debug!(
            row_count,
            min_rows = MIN_ROWS_FOR_VECTOR_INDEX,
            "skipping vector index creation for small table"
        );
        return Ok(());
    }

    let index = to_lancedb_index(vector_index_config);
    match table.create_index(&["vector"], index).execute().await {
        Ok(()) => {
            debug!("created vector index on `vector` column");
            Ok(())
        }
        Err(error) if is_skippable_index_error(&error) => {
            warn!(error = %error, "skipped vector index creation");
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

async fn drop_vector_indices(table: &lancedb::Table) -> Result<(), EmbedError> {
    for index in table.list_indices().await? {
        if index.columns.iter().any(|column| column == "vector") {
            table.drop_index(&index.name).await?;
        }
    }
    Ok(())
}

async fn vector_index_exists(table: &lancedb::Table) -> Result<bool, EmbedError> {
    let indices = table.list_indices().await?;
    Ok(indices
        .iter()
        .any(|index| index.columns.iter().any(|column| column == "vector")))
}

fn is_table_not_found(error: &lancedb::Error) -> bool {
    matches!(error, lancedb::Error::TableNotFound { .. })
}

fn is_table_already_exists(error: &lancedb::Error) -> bool {
    matches!(error, lancedb::Error::TableAlreadyExists { .. })
}

fn is_skippable_index_error(error: &lancedb::Error) -> bool {
    match error {
        lancedb::Error::InvalidInput { message } | lancedb::Error::Runtime { message } => {
            let lowered = message.to_ascii_lowercase();
            lowered.contains("not enough")
                || lowered.contains("insufficient")
                || lowered.contains("too few")
                || lowered.contains("minimum")
        }
        _ => false,
    }
}

fn vector_item_field() -> Arc<Field> {
    Arc::new(Field::new("item", DataType::Float32, false))
}

pub fn chunk_table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(vector_item_field(), EMBEDDING_DIM),
            false,
        ),
        Field::new("text", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("start_line", DataType::Int64, false),
        Field::new("end_line", DataType::Int64, false),
    ]))
}

pub fn build_record_batch(
    chunks: &[SemanticChunk],
    embedder: &LocalEmbedder,
    batch_size: usize,
) -> Result<RecordBatch, EmbedError> {
    validate_index_input(chunks, batch_size)?;

    let mut vectors = Vec::with_capacity(chunks.len());
    for chunk_batch in chunks.chunks(batch_size) {
        let batch_texts: Vec<&str> = chunk_batch
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect();
        vectors.extend(embedder.generate_embeddings(&batch_texts)?);
    }

    assemble_record_batch(chunks, &vectors)
}

#[doc(hidden)]
pub fn record_batch_from_vectors(
    chunks: &[SemanticChunk],
    vectors: &[Vec<f32>],
) -> Result<RecordBatch, EmbedError> {
    if chunks.len() != vectors.len() {
        return Err(EmbedError::InvalidMetadata(format!(
            "chunk count {} does not match vector count {}",
            chunks.len(),
            vectors.len()
        )));
    }

    for chunk in chunks {
        chunk.validate()?;
    }

    for (index, vector) in vectors.iter().enumerate() {
        if vector.len() != EMBEDDING_DIM as usize {
            return Err(EmbedError::InvalidMetadata(format!(
                "vector at index {index} must be {EMBEDDING_DIM}-dimensional, got {}",
                vector.len()
            )));
        }
    }

    assemble_record_batch(chunks, vectors)
}

fn assemble_record_batch(
    chunks: &[SemanticChunk],
    vectors: &[Vec<f32>],
) -> Result<RecordBatch, EmbedError> {
    let schema = chunk_table_schema();
    let row_count = chunks.len();

    let mut flat_vectors = Vec::with_capacity(row_count * EMBEDDING_DIM as usize);
    let mut texts = Vec::with_capacity(row_count);
    let mut kinds = Vec::with_capacity(row_count);
    let mut names: Vec<Option<String>> = Vec::with_capacity(row_count);
    let mut file_paths = Vec::with_capacity(row_count);
    let mut start_lines = Vec::with_capacity(row_count);
    let mut end_lines = Vec::with_capacity(row_count);

    for (chunk, embedding) in chunks.iter().zip(vectors) {
        flat_vectors.extend_from_slice(embedding);
        texts.push(chunk.text.as_str());
        kinds.push(chunk.kind.as_str());
        names.push(chunk.name.clone());
        file_paths.push(chunk.file_path.as_str());
        start_lines.push(chunk.start_line as i64);
        end_lines.push(chunk.end_line as i64);
    }

    let vector_values = Arc::new(Float32Array::from(flat_vectors));
    let vector_array = Arc::new(FixedSizeListArray::new(
        vector_item_field(),
        EMBEDDING_DIM,
        vector_values,
        None,
    ));

    let text_array = Arc::new(StringArray::from(texts));
    let kind_array = Arc::new(StringArray::from(kinds));
    let name_array = Arc::new(StringArray::from(names));
    let file_path_array = Arc::new(StringArray::from(file_paths));
    let start_line_array = Arc::new(Int64Array::from(start_lines));
    let end_line_array = Arc::new(Int64Array::from(end_lines));

    RecordBatch::try_new(
        schema,
        vec![
            vector_array,
            text_array,
            kind_array,
            name_array,
            file_path_array,
            start_line_array,
            end_line_array,
        ],
    )
    .map_err(EmbedError::from)
}

fn validate_index_input(chunks: &[SemanticChunk], batch_size: usize) -> Result<(), EmbedError> {
    validate_index_batch(chunks)?;
    validate_embed_batch_size(batch_size)?;
    for chunk in chunks {
        chunk.validate()?;
    }
    Ok(())
}

fn validate_chunks_for_file(file_path: &str, chunks: &[SemanticChunk]) -> Result<(), EmbedError> {
    for chunk in chunks {
        if chunk.file_path != file_path {
            return Err(EmbedError::InvalidChunk(format!(
                "expected file_path `{file_path}`, got `{}`",
                chunk.file_path
            )));
        }
    }
    Ok(())
}

fn escape_sql_string(value: &str) -> String {
    value.replace('\'', "''")
}

fn file_path_predicate(file_path: &str) -> String {
    format!("file_path = '{}'", escape_sql_string(file_path))
}

fn parse_search_batch(batch: &RecordBatch) -> Result<Vec<SearchResult>, EmbedError> {
    let row_count = batch.num_rows();
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let text_array = column_as_string(batch, "text")?;
    let kind_array = column_as_string(batch, "kind")?;
    let name_array = column_as_string(batch, "name")?;
    let file_path_array = column_as_string(batch, "file_path")?;
    let start_line_array = column_as_i64(batch, "start_line")?;
    let end_line_array = column_as_i64(batch, "end_line")?;
    let distance_scores = batch
        .schema()
        .fields()
        .iter()
        .position(|field| field.name() == "_distance")
        .map(|index| {
            let array = batch.column(index);
            let distances = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    EmbedError::InvalidMetadata("_distance column is not Float32".to_owned())
                })?;
            Ok::<&Float32Array, EmbedError>(distances)
        })
        .transpose()?;

    let mut results = Vec::with_capacity(row_count);

    for row in 0..row_count {
        let start_line = usize::try_from(start_line_array.value(row)).map_err(|_| {
            EmbedError::InvalidMetadata(format!(
                "negative start_line at row {row}: {}",
                start_line_array.value(row)
            ))
        })?;
        let end_line = usize::try_from(end_line_array.value(row)).map_err(|_| {
            EmbedError::InvalidMetadata(format!(
                "negative end_line at row {row}: {}",
                end_line_array.value(row)
            ))
        })?;

        let name = if name_array.is_null(row) {
            None
        } else {
            let value = name_array.value(row);
            if value.is_empty() {
                None
            } else {
                Some(value.to_owned())
            }
        };

        let score = if let Some(distances) = distance_scores {
            l2_distance_to_cosine_score(distances.value(row))
        } else {
            0.0
        };

        results.push(SearchResult {
            chunk: SemanticChunk {
                kind: match kind_array.value(row).parse() {
                    Ok(kind) => kind,
                    Err(error) => match error {},
                },
                name,
                text: text_array.value(row).to_owned(),
                start_line,
                end_line,
                file_path: file_path_array.value(row).to_owned(),
            },
            score,
        });
    }

    Ok(results)
}

fn column_as_string<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, EmbedError> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| EmbedError::InvalidMetadata(format!("missing `{name}` column")))?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| EmbedError::InvalidMetadata(format!("`{name}` column is not Utf8")))
}

fn column_as_i64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, EmbedError> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| EmbedError::InvalidMetadata(format!("missing `{name}` column")))?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| EmbedError::InvalidMetadata(format!("`{name}` column is not Int64")))
}

/// L2 distance between unit vectors → cosine similarity.
pub(crate) fn l2_distance_to_cosine_score(distance: f32) -> f32 {
    let similarity = 1.0 - (distance * distance) / 2.0;
    similarity.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChunkKind;
    use std::sync::Arc;

    #[test]
    fn schema_uses_native_line_columns() {
        let schema = chunk_table_schema();
        assert!(schema.field_with_name("start_line").is_ok());
        assert!(schema.field_with_name("end_line").is_ok());
        assert!(schema.field_with_name("line_range").is_err());
        assert_eq!(
            schema.field_with_name("start_line").unwrap().data_type(),
            &DataType::Int64
        );
    }

    #[test]
    fn parses_search_batch_from_integer_line_columns() {
        let schema = chunk_table_schema();
        let vector_values = Arc::new(Float32Array::from(vec![1.0_f32; EMBEDDING_DIM as usize]));
        let vector = Arc::new(FixedSizeListArray::new(
            vector_item_field(),
            EMBEDDING_DIM,
            vector_values,
            None,
        ));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                vector,
                Arc::new(StringArray::from(vec!["fn main() {}"])),
                Arc::new(StringArray::from(vec!["Function"])),
                Arc::new(StringArray::from(vec![Some("main")])),
                Arc::new(StringArray::from(vec!["src/main.rs"])),
                Arc::new(Int64Array::from(vec![10_i64])),
                Arc::new(Int64Array::from(vec![12_i64])),
            ],
        )
        .expect("record batch");

        let results = parse_search_batch(&batch).expect("parse batch");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.start_line, 10);
        assert_eq!(results[0].chunk.end_line, 12);
    }

    #[test]
    fn file_path_predicate_escapes_quotes() {
        assert_eq!(
            file_path_predicate("src/a'b.rs"),
            "file_path = 'src/a''b.rs'"
        );
    }

    #[test]
    fn validate_chunks_for_file_rejects_mismatch() {
        let chunks = vec![SemanticChunk {
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
            SemanticChunk {
                kind: ChunkKind::Function,
                name: Some("ok".into()),
                text: "fn ok() {}".into(),
                start_line: 1,
                end_line: 1,
                file_path: "src/a.rs".into(),
            },
            SemanticChunk {
                kind: ChunkKind::Function,
                name: Some("bad".into()),
                text: "fn bad() {}".into(),
                start_line: 5,
                end_line: 2,
                file_path: "src/b.rs".into(),
            },
        ];

        let err = validate_index_input(&chunks, DEFAULT_EMBED_BATCH_SIZE).expect_err("invalid");
        assert!(matches!(err, EmbedError::InvalidChunk(_)));
    }

    #[test]
    fn cosine_score_conversion() {
        assert!((l2_distance_to_cosine_score(0.0) - 1.0).abs() < f32::EPSILON);
        assert!(l2_distance_to_cosine_score(2.0) < 0.0);
    }
}
