use std::path::Path;
use std::sync::Arc;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tracing::{debug, instrument};

use crate::types::{validate_input_text_length, EmbedError, EMBEDDING_DIM};

pub struct LocalEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    max_position_embeddings: usize,
}

impl LocalEmbedder {
    pub fn new(model_weights_path: &Path, tokenizer_path: &Path) -> Result<Self, EmbedError> {
        Self::load(model_weights_path, tokenizer_path, Device::Cpu)
    }

    pub fn with_device(
        model_weights_path: &Path,
        tokenizer_path: &Path,
        device: Device,
    ) -> Result<Self, EmbedError> {
        Self::load(model_weights_path, tokenizer_path, device)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn load(
        model_weights_path: &Path,
        tokenizer_path: &Path,
        device: Device,
    ) -> Result<Self, EmbedError> {
        let config_path = model_weights_path
            .parent()
            .map(|parent| parent.join("config.json"))
            .ok_or_else(|| {
                EmbedError::ModelConfig(format!(
                    "cannot resolve parent directory for {}",
                    model_weights_path.display()
                ))
            })?;

        if !config_path.is_file() {
            return Err(EmbedError::MissingFile(config_path.display().to_string()));
        }

        let config_data = std::fs::read_to_string(&config_path)
            .map_err(|source| EmbedError::io_path(&config_path, source))?;
        let config: Config = serde_json::from_str(&config_data)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| EmbedError::Tokenizer(error.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_position_embeddings,
                ..TruncationParams::default()
            }))
            .map_err(|error| EmbedError::Tokenizer(error.to_string()))?;
        // BatchLongest: don't pad every batch to tokenizer.json's fixed width (e.g. 128).
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        }));

        let weights_path = model_weights_path.to_path_buf();
        // SAFETY: standard Candle mmap path — only load verified weights.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device).map_err(
                |error| EmbedError::ModelConfig(format!("failed to mmap safetensors: {error}")),
            )?
        };

        let model = BertModel::load(vb, &config)?;
        let max_position_embeddings = config.max_position_embeddings;

        debug!(
            max_position_embeddings,
            path = %model_weights_path.display(),
            "loaded embedding model"
        );

        Ok(Self {
            model,
            tokenizer,
            device,
            max_position_embeddings,
        })
    }

    #[instrument(skip(self, text), fields(input_len = text.len()))]
    pub fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut batch = self.generate_embeddings(&[text])?;
        batch
            .pop()
            .ok_or_else(|| EmbedError::inference("empty embedding batch"))
    }

    #[instrument(skip(self, texts), fields(batch = texts.len()))]
    pub fn generate_embeddings(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        for text in texts {
            validate_input_text_length(text)?;
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|error| EmbedError::Tokenizer(error.to_string()))?;

        for encoding in &encodings {
            if encoding.get_ids().len() > self.max_position_embeddings {
                return Err(EmbedError::Tokenizer(format!(
                    "encoded sequence length {} exceeds model limit {}",
                    encoding.get_ids().len(),
                    self.max_position_embeddings
                )));
            }
            validate_attention_mask(encoding.get_attention_mask())?;
        }

        let token_ids =
            stack_token_tensors(&encodings, |encoding| encoding.get_ids(), &self.device)?;
        let attention_mask = stack_token_tensors(
            &encodings,
            |encoding| encoding.get_attention_mask(),
            &self.device,
        )?;
        let token_type_ids = token_ids.zeros_like()?;

        let token_embeddings =
            self.model
                .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        let pooled = mean_pool(&token_embeddings, &attention_mask)?;
        let normalized = l2_normalize(&pooled)?;

        let batch_size = texts.len();
        let mut embeddings = Vec::with_capacity(batch_size);
        for index in 0..batch_size {
            let vector = normalized.get(index)?.to_vec1::<f32>()?;
            if vector.len() != EMBEDDING_DIM as usize {
                return Err(EmbedError::inference(format!(
                    "expected {EMBEDDING_DIM}-dimensional embedding, got {}",
                    vector.len()
                )));
            }
            validate_finite_embedding(&vector)?;
            embeddings.push(vector);
        }

        Ok(embeddings)
    }
}

pub async fn generate_embedding_async(
    embedder: Arc<LocalEmbedder>,
    text: impl Into<String>,
) -> Result<Vec<f32>, EmbedError> {
    let text = text.into();
    tokio::task::spawn_blocking(move || embedder.generate_embedding(&text)).await?
}

fn stack_token_tensors<F>(
    encodings: &[tokenizers::Encoding],
    select: F,
    device: &Device,
) -> Result<Tensor, EmbedError>
where
    F: Fn(&tokenizers::Encoding) -> &[u32],
{
    let tensors: Result<Vec<_>, EmbedError> = encodings
        .iter()
        .map(|encoding| {
            let values = select(encoding);
            Ok(Tensor::new(values, device)?)
        })
        .collect();
    Ok(Tensor::stack(&tensors?, 0)?)
}

fn validate_attention_mask(mask: &[u32]) -> Result<(), EmbedError> {
    let active_tokens: u32 = mask.iter().copied().sum();
    if active_tokens == 0 {
        return Err(EmbedError::Tokenizer(
            "tokenization produced an empty attention mask".to_owned(),
        ));
    }
    Ok(())
}

fn validate_finite_embedding(values: &[f32]) -> Result<(), EmbedError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(EmbedError::inference(
            "embedding contains NaN or Inf values",
        ));
    }
    Ok(())
}

fn mean_pool(embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor, EmbedError> {
    let mask = attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;
    let masked = embeddings.broadcast_mul(&mask)?;
    let summed = masked.sum(1)?;
    let counts = mask.sum(1)?;

    let counts_vec = counts.flatten_all()?.to_vec1::<f32>()?;
    if counts_vec
        .iter()
        .any(|count| !count.is_finite() || *count <= f32::EPSILON)
    {
        return Err(EmbedError::Tokenizer(
            "cannot mean-pool with zero active tokens".to_owned(),
        ));
    }

    Ok(summed.broadcast_div(&counts)?)
}

fn l2_normalize(embeddings: &Tensor) -> Result<Tensor, EmbedError> {
    let norms = embeddings.sqr()?.sum_keepdim(1)?.sqrt()?;
    let norms_vec = norms.flatten_all()?.to_vec1::<f32>()?;
    if norms_vec
        .iter()
        .any(|norm| !norm.is_finite() || *norm <= f32::EPSILON)
    {
        return Err(EmbedError::inference(
            "cannot L2-normalize a zero or non-finite embedding",
        ));
    }
    Ok(embeddings.broadcast_div(&norms)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MAX_INPUT_TEXT_LEN;

    #[test]
    fn rejects_input_before_tokenization() {
        let text = "x".repeat(MAX_INPUT_TEXT_LEN + 1);
        let err = validate_input_text_length(&text).expect_err("expected length error");
        assert!(matches!(err, EmbedError::Tokenizer(_)));
    }
}
