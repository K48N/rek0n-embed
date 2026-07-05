# rek0n-embed

Embeds parser chunks locally and stores them in [rek0n-db](https://github.com/K48N/rek0n-db) for semantic search.

> **Storage:** Vector persistence goes through `rek0n-db` (mmap + exact/IVF-lite search). LanceDB is no longer a dependency — no `protoc` or Arrow toolchain required.

## Overview

This crate takes [`IndexedChunk`](https://github.com/K48N/rek0n-chunk) values from [rek0n-chunk](https://github.com/K48N/rek0n-chunk), runs all-MiniLM-L6-v2 through Candle on CPU, and writes 384-dimensional L2-normalized vectors into a per-repo `rek0n-db` store. Build them from parser [`ParsedChunk`](https://github.com/K48N/rek0n-chunk) output with `try_from_parser_chunk`. No cloud embedding APIs and no separate vector service.

Local inference is slower than a hosted API. The tradeoff is privacy and control: the pipeline is meant to read a whole repository without sending source code off the machine.

## How it works

1. `LocalEmbedder` loads `model.safetensors`, tokenizes text, runs BERT, mean-pools with the attention mask, and L2-normalizes to 384 floats.
2. `VectorStorage::replace_file_chunks` embeds on a blocking thread pool, then writes through `rek0n-db`. Rows for that `file_path` are deleted and replaced.
3. Each repo gets a directory under `{db_dir}/{repo_name}` with `manifest.json` and mmap'd `vectors.bin`.
4. `query_semantic_context` embeds the query off-thread and runs vector search.
5. IVF-lite indexes are built once a store reaches 256 rows. Until then, search is exact dot product.

## Design

**Embed before lock.** Tokenization and forward passes stay off the async executor. `rek0n-db` handles advisory locking at open time.

**rek0n-db on disk.** Lightweight mmap storage without LanceDB churn or protobuf build scripts. Good fit for a single-machine indexer and MCTS branch workloads.

**Incremental index maintenance.** IVF is built lazily on writes; search uses IVF only when the index already exists.

**Shared chunk types.** `ParsedChunk` and `IndexedChunk` live in rek0n-chunk; parser and embed re-export them under the same names.

## Usage

```rust
use std::sync::Arc;
use rek0n_embed::{
    try_from_parser_chunk, IndexedChunk, LocalEmbedder, ParsedChunk, VectorStorage,
    query_semantic_context,
};
use rek0n_parser::parse_file;

let parsed: Vec<ParsedChunk> = parse_file(source, "rust")?;
let chunks: Vec<IndexedChunk> = parsed
    .iter()
    .filter(|chunk| !chunk.has_error)
    .map(|chunk| try_from_parser_chunk("src/lib.rs", chunk))
    .collect::<Result<_, _>>()?;

let embedder = Arc::new(LocalEmbedder::new(
    "model/model.safetensors".as_ref(),
    "model/tokenizer.json".as_ref(),
)?);
let storage = VectorStorage::initialize("./vectors", "my-repo").await?;
storage.replace_file_chunks("src/lib.rs", &chunks, Arc::clone(&embedder)).await?;

let hits = query_semantic_context(&storage, embedder, "how does auth work?", 10).await?;
```

Example:

```sh
./scripts/download_model.sh
cargo run --example semantic_search
```

## Publishing to crates.io

Publish dependencies first, then this crate:

1. [`rek0n-chunk`](https://crates.io/crates/rek0n-chunk) — already on crates.io
2. [`rek0n-db`](https://github.com/K48N/rek0n-db) — publish before embed
3. **rek0n-embed** — set `rek0n-chunk = "0.1.0"` and `rek0n-db = "0.1.0"` (no `path =` in `Cargo.toml`)

From this repo root:

```sh
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
cargo publish --dry-run
cargo publish
```

`cargo login` is one-time. Bump `version` for each release; crates.io does not allow republishing the same version.

## Known gaps

- CPU inference dominates latency on large repos. `with_device` exists but there is no tuned GPU pipeline yet.
- Advisory file locks coordinate one machine, not a fleet of indexers.
- Model swap is manual: tokenizer and architecture must stay BERT-compatible with Candle.
- Stores under 256 rows use flat search by design.

## License

MIT
