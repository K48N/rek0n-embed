use serde::{Deserialize, Serialize};

pub const MAX_INPUT_TEXT_LEN: usize = 262_144;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChunkKind {
    Function,
    Struct,
    Impl,
    Unknown,
}

impl ChunkKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChunkKind::Function => "Function",
            ChunkKind::Struct => "Struct",
            ChunkKind::Impl => "Impl",
            ChunkKind::Unknown => "Unknown",
        }
    }
}

impl std::str::FromStr for ChunkKind {
    type Err = std::convert::Infallible;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(match value {
            "Function" => ChunkKind::Function,
            "Struct" => ChunkKind::Struct,
            "Impl" => ChunkKind::Impl,
            _ => ChunkKind::Unknown,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SemanticChunk {
    pub kind: ChunkKind,
    pub name: Option<String>,
    pub text: String,
    pub start_line: usize,
    pub end_line: usize,
    pub file_path: String,
}

impl SemanticChunk {
    pub fn line_range(&self) -> String {
        format!("{}:{}", self.start_line, self.end_line)
    }

    pub fn validate(&self) -> Result<(), EmbedError> {
        if self.text.trim().is_empty() {
            return Err(EmbedError::InvalidChunk(
                "text must not be empty".to_owned(),
            ));
        }

        if self.file_path.trim().is_empty() {
            return Err(EmbedError::InvalidChunk(
                "file_path must not be empty".to_owned(),
            ));
        }

        if self.end_line < self.start_line {
            return Err(EmbedError::InvalidChunk(format!(
                "end_line ({}) must be >= start_line ({})",
                self.end_line, self.start_line
            )));
        }

        validate_input_text_length(&self.text)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: SemanticChunk,
    pub score: f32,
}

pub const EMBEDDING_DIM: i32 = 384;

pub const DEFAULT_EMBED_BATCH_SIZE: usize = 32;

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
}

pub fn validate_input_text_length(text: &str) -> Result<(), EmbedError> {
    if text.len() > MAX_INPUT_TEXT_LEN {
        return Err(EmbedError::Tokenizer(format!(
            "input length {} exceeds limit of {MAX_INPUT_TEXT_LEN} characters",
            text.len()
        )));
    }
    Ok(())
}

pub fn validate_search_limit(limit: usize) -> Result<usize, EmbedError> {
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
    Ok(())
}

pub fn validate_file_path(file_path: &str) -> Result<(), EmbedError> {
    if file_path.trim().is_empty() {
        return Err(EmbedError::InvalidMetadata(
            "file_path must not be empty".to_owned(),
        ));
    }
    Ok(())
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

    fn sample_chunk() -> SemanticChunk {
        SemanticChunk {
            kind: ChunkKind::Function,
            name: Some("foo".into()),
            text: "fn foo() {}".into(),
            start_line: 1,
            end_line: 3,
            file_path: "src/lib.rs".into(),
        }
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
        assert!(matches!(chunk.validate(), Err(EmbedError::InvalidChunk(_))));
    }

    #[test]
    fn rejects_empty_chunk_file_path() {
        let mut chunk = sample_chunk();
        chunk.file_path = "".into();
        assert!(matches!(chunk.validate(), Err(EmbedError::InvalidChunk(_))));
    }

    #[test]
    fn rejects_inverted_line_range() {
        let mut chunk = sample_chunk();
        chunk.start_line = 10;
        chunk.end_line = 2;
        assert!(matches!(chunk.validate(), Err(EmbedError::InvalidChunk(_))));
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
}
