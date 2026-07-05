use rek0n_chunk::{
    validate_file_path as chunk_validate_file_path,
    validate_input_text_length as chunk_validate_input_text_length, ChunkError,
};

pub use rek0n_chunk::{ChunkKind, IndexedChunk, MAX_FILE_PATH_LEN, MAX_INPUT_TEXT_LEN};

pub const MAX_QUERY_TEXT_LEN: usize = 8_192;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: IndexedChunk,
    pub score: f32,
}

pub const EMBEDDING_DIM: i32 = 384;

pub const DEFAULT_EMBED_BATCH_SIZE: usize = 32;
pub const MAX_EMBED_BATCH_SIZE: usize = 256;
pub const MAX_INDEX_BATCH_CHUNKS: usize = 10_000;
pub const MAX_SEARCH_LIMIT: usize = 10_000;

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("LanceDB error: {0}")]
    LanceDb(#[from] lancedb::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Candle tensor error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),

    #[error("Invalid semantic chunk: {0}")]
    InvalidChunk(String),

    #[error("model load error: {0}")]
    ModelConfig(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("background task failed")]
    BackgroundTask(#[from] tokio::task::JoinError),

    #[error("missing required file: {0}")]
    MissingFile(String),

    #[error("invalid table name `{name}`: {reason}")]
    InvalidTableName { name: String, reason: String },

    #[error("file lock timed out at {path}")]
    LockTimeout { path: String },
}

impl EmbedError {
    pub fn io_path(path: impl AsRef<std::path::Path>, source: std::io::Error) -> Self {
        EmbedError::Io {
            path: path.as_ref().display().to_string(),
            source,
        }
    }

    pub fn inference(message: impl Into<String>) -> Self {
        EmbedError::Inference(message.into())
    }

    fn from_chunk_error(error: ChunkError) -> Self {
        error.into()
    }
}

impl From<ChunkError> for EmbedError {
    fn from(error: ChunkError) -> Self {
        match error {
            ChunkError::EmptyText => EmbedError::InvalidChunk("text must not be empty".into()),
            ChunkError::EmptyFilePath => {
                EmbedError::InvalidChunk("file_path must not be empty".into())
            }
            ChunkError::InputTooLong { len, max } => EmbedError::Tokenizer(format!(
                "input length {len} exceeds limit of {max} characters"
            )),
            ChunkError::FilePathTooLong { len, max } => EmbedError::InvalidMetadata(format!(
                "file_path length {len} exceeds limit of {max}"
            )),
            ChunkError::FilePathControlChars => {
                EmbedError::InvalidMetadata("file_path must not contain control characters".into())
            }
            ChunkError::InvertedLineRange {
                start_line,
                end_line,
            } => EmbedError::InvalidChunk(format!(
                "end_line ({end_line}) must be >= start_line ({start_line})"
            )),
            ChunkError::HasSyntaxErrors => {
                EmbedError::InvalidChunk("parser flagged chunk with syntax errors".into())
            }
        }
    }
}

pub fn validate_input_text_length(text: &str) -> Result<(), EmbedError> {
    chunk_validate_input_text_length(text).map_err(EmbedError::from_chunk_error)
}

pub fn validate_search_limit(limit: usize) -> Result<usize, EmbedError> {
    if limit == 0 {
        return Err(EmbedError::InvalidMetadata(
            "search limit must be > 0".to_owned(),
        ));
    }
    if limit > MAX_SEARCH_LIMIT {
        return Err(EmbedError::InvalidMetadata(format!(
            "search limit {limit} exceeds maximum of {MAX_SEARCH_LIMIT}"
        )));
    }
    u32::try_from(limit).map_err(|_| {
        EmbedError::InvalidMetadata(format!("search limit {limit} exceeds u32::MAX"))
    })?;
    Ok(limit)
}

pub fn validate_embed_batch_size(batch_size: usize) -> Result<(), EmbedError> {
    if batch_size == 0 {
        return Err(EmbedError::InvalidMetadata(
            "embed batch size must be > 0".to_owned(),
        ));
    }
    if batch_size > MAX_EMBED_BATCH_SIZE {
        return Err(EmbedError::InvalidMetadata(format!(
            "embed batch size {batch_size} exceeds maximum of {MAX_EMBED_BATCH_SIZE}"
        )));
    }
    Ok(())
}

pub fn validate_index_batch(chunks: &[IndexedChunk]) -> Result<(), EmbedError> {
    if chunks.is_empty() {
        return Err(EmbedError::InvalidMetadata(
            "index batch must contain at least one chunk".to_owned(),
        ));
    }
    if chunks.len() > MAX_INDEX_BATCH_CHUNKS {
        return Err(EmbedError::InvalidMetadata(format!(
            "index batch contains {} chunks, exceeding maximum of {MAX_INDEX_BATCH_CHUNKS}",
            chunks.len()
        )));
    }
    Ok(())
}

pub fn validate_query_text(text: &str) -> Result<(), EmbedError> {
    if text.trim().is_empty() {
        return Err(EmbedError::InvalidMetadata(
            "query text must not be empty".to_owned(),
        ));
    }
    if text.len() > MAX_QUERY_TEXT_LEN {
        return Err(EmbedError::InvalidMetadata(format!(
            "query text length {} exceeds limit of {MAX_QUERY_TEXT_LEN}",
            text.len()
        )));
    }
    Ok(())
}

pub fn validate_file_path(file_path: &str) -> Result<(), EmbedError> {
    chunk_validate_file_path(file_path).map_err(EmbedError::from_chunk_error)
}

pub fn try_from_parser_chunk(
    file_path: impl Into<String>,
    parsed: &rek0n_chunk::ParsedChunk,
) -> Result<IndexedChunk, EmbedError> {
    IndexedChunk::from_parsed(file_path, parsed).map_err(EmbedError::from_chunk_error)
}

pub fn try_from_parser_parts(
    file_path: impl Into<String>,
    parser_kind: &str,
    name: Option<String>,
    text: impl Into<String>,
    start_line: usize,
    end_line: usize,
    has_error: bool,
) -> Result<IndexedChunk, EmbedError> {
    IndexedChunk::try_from_parser_parts(
        file_path,
        parser_kind,
        name,
        text,
        start_line,
        end_line,
        has_error,
    )
    .map_err(EmbedError::from_chunk_error)
}

pub fn validate_table_name(name: &str) -> Result<(), EmbedError> {
    const MAX_LEN: usize = 255;

    if name.is_empty() {
        return Err(EmbedError::InvalidTableName {
            name: name.to_owned(),
            reason: "must not be empty".to_owned(),
        });
    }

    if name.len() > MAX_LEN {
        return Err(EmbedError::InvalidTableName {
            name: name.to_owned(),
            reason: format!("must be at most {MAX_LEN} bytes"),
        });
    }

    if name == "." || name == ".." {
        return Err(EmbedError::InvalidTableName {
            name: name.to_owned(),
            reason: "reserved path component".to_owned(),
        });
    }

    if name.contains('/') || name.contains('\\') {
        return Err(EmbedError::InvalidTableName {
            name: name.to_owned(),
            reason: "must not contain path separators".to_owned(),
        });
    }

    if name.contains("..") {
        return Err(EmbedError::InvalidTableName {
            name: name.to_owned(),
            reason: "must not contain traversal sequences".to_owned(),
        });
    }

    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
    {
        return Err(EmbedError::InvalidTableName {
            name: name.to_owned(),
            reason: "only ASCII letters, digits, '_' and '-' are allowed".to_owned(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunk() -> IndexedChunk {
        IndexedChunk {
            kind: ChunkKind::Function,
            name: Some("foo".into()),
            text: "fn foo() {}".into(),
            start_line: 1,
            end_line: 3,
            file_path: "src/lib.rs".into(),
        }
    }

    #[test]
    fn maps_parser_kind_strings() {
        assert_eq!(
            ChunkKind::from_parser_kind("type_alias"),
            ChunkKind::TypeAlias
        );
        assert_eq!(ChunkKind::from_parser_kind("Function"), ChunkKind::Function);
    }

    #[test]
    fn from_parser_parts_builds_chunk() {
        let chunk = IndexedChunk::from_parser_parts(
            "src/lib.rs",
            "function",
            Some("main".into()),
            "fn main() {}",
            1,
            1,
        );
        assert_eq!(chunk.kind, ChunkKind::Function);
        assert_eq!(chunk.file_path, "src/lib.rs");
        assert!(chunk.validate().is_ok());
    }

    #[test]
    fn rejects_empty_file_path() {
        assert!(validate_file_path("").is_err());
        assert!(validate_file_path("   ").is_err());
    }

    #[test]
    fn rejects_unsafe_table_names() {
        for name in ["", "..", "../secrets", "foo/bar", "bad name", "a/b"] {
            assert!(
                validate_table_name(name).is_err(),
                "{name} should be rejected"
            );
        }
    }

    #[test]
    fn accepts_safe_table_names() {
        for name in ["demo", "my-repo", "repo_1"] {
            assert!(
                validate_table_name(name).is_ok(),
                "{name} should be accepted"
            );
        }
    }

    #[test]
    fn rejects_empty_chunk_text() {
        let mut chunk = sample_chunk();
        chunk.text = "   ".into();
        assert!(matches!(chunk.validate(), Err(ChunkError::EmptyText)));
    }

    #[test]
    fn rejects_empty_chunk_file_path() {
        let mut chunk = sample_chunk();
        chunk.file_path = "".into();
        assert!(matches!(chunk.validate(), Err(ChunkError::EmptyFilePath)));
    }

    #[test]
    fn rejects_inverted_line_range() {
        let mut chunk = sample_chunk();
        chunk.start_line = 10;
        chunk.end_line = 2;
        assert!(matches!(
            chunk.validate(),
            Err(ChunkError::InvertedLineRange { .. })
        ));
    }

    #[test]
    fn rejects_oversized_input_text() {
        let text = "a".repeat(MAX_INPUT_TEXT_LEN + 1);
        assert!(validate_input_text_length(&text).is_err());
    }

    #[test]
    fn rejects_search_limit_overflow() {
        assert!(validate_search_limit(usize::MAX).is_err());
    }

    #[test]
    fn rejects_zero_embed_batch_size() {
        assert!(validate_embed_batch_size(0).is_err());
    }

    #[test]
    fn rejects_oversized_index_batch() {
        let chunks = vec![sample_chunk(); MAX_INDEX_BATCH_CHUNKS + 1];
        assert!(validate_index_batch(&chunks).is_err());
    }

    #[test]
    fn rejects_zero_search_limit() {
        assert!(validate_search_limit(0).is_err());
    }

    #[test]
    fn rejects_empty_query_text() {
        assert!(validate_query_text("   ").is_err());
    }

    #[test]
    fn rejects_control_characters_in_file_path() {
        assert!(validate_file_path("src/a\nb.rs").is_err());
    }

    #[test]
    fn rejects_parser_error_chunks() {
        let err = try_from_parser_parts(
            "src/a.rs",
            "function",
            Some("broken".into()),
            "fn broken() {}",
            1,
            1,
            true,
        )
        .expect_err("has_error");
        assert!(matches!(err, EmbedError::InvalidChunk(_)));
    }
}
