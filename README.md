# rek0n-embed

Part of [rek0n](https://github.com/K48N/rek0n). Embeds parser chunks locally and stores them in LanceDB for semantic search.

## Overview

This crate takes `SemanticChunk` values (indexed chunks from [rek0n-chunk](https://github.com/K48N/rek0n-chunk)), runs all-MiniLM-L6-v2 through Candle on CPU, and writes 384-dimensional L2-normalized vectors into LanceDB. No cloud embedding APIs and no separate vector service.

Local inference is slower than a hosted API. The tradeoff is privacy and control: rek0n is meant to read a whole repository without sending source code off the machine.

## How it works

1. `LocalEmbedder` loads `model.safetensors`, tokenizes text, runs BERT, mean-pools with the attention mask, and L2-normalizes to 384 floats.
2. `VectorStorage::replace_file_chunks` embeds on a blocking thread pool, then takes a file lock only for the LanceDB write. Rows for that `file_path` are deleted and replaced.
3. Vectors are stored as Arrow fixed-size float lists in a per-repo table.
4. `query_semantic_context` embeds the query off-thread, opens a shared lock, and runs vector search.
5. ANN indexes are created once a table reaches 256 rows. Appends call LanceDB `optimize` with incremental merge. File replace and delete paths retrain the existing index instead of dropping and rebuilding it from scratch.

## Design

**Embed before lock.** Tokenization and forward passes stay off the async executor. The advisory lock covers LanceDB I/O only.

**LanceDB on disk.** Embedded Arrow-native storage without running a separate server. Good fit for a single-machine indexer.

**Incremental index maintenance.** Full index drops were too expensive on frequent file saves. `OptimizeOptions::merge(0)` handles appends; `OptimizeOptions::retrain()` handles replace and delete churn.

**Shared chunk types.** rek0n-chunk keeps parser and embed aligned on kinds, limits, and validation rules.

## Usage

```rust
use std::sync::Arc;
use rek0n_embed::{try_from_parser_chunk, LocalEmbedder, VectorStorage, query_semantic_context};

let embedder = Arc::new(LocalEmbedder::new(
    "model/model.safetensors".as_ref(),
    "model/tokenizer.json".as_ref(),
)?);
let storage = VectorStorage::initialize("./lancedb", "my-repo").await?;

for parsed in parsed_chunks {
    let chunk = try_from_parser_chunk("src/lib.rs", &parsed)?;
    // batch and index...
}

let hits = query_semantic_context(&storage, embedder, "how does auth work?", 10).await?;
```

Example:

```sh
./scripts/download_model.sh
cargo run --example semantic_search
```

Requires `protoc` for LanceDB build scripts. CI installs it automatically.

## Known gaps

- CPU inference dominates latency on large repos. `with_device` exists but there is no tuned GPU pipeline yet.
- Advisory file locks coordinate one machine, not a fleet of indexers.
- Model swap is manual: tokenizer and architecture must stay BERT-compatible with Candle.
- Tables under 256 rows use flat search by design.

## License

MIT
