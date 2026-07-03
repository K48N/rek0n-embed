# rek0n-embed

Part of my personal project rek0n. Embeds parser chunks locally and stores them in LanceDB for semantic search.

## What it is

A Rust library that takes `SemanticChunk` values from rek0n-parser, runs all-MiniLM-L6-v2 via Candle, and persists 384-d L2-normalized vectors in LanceDB for nearest-neighbor search. No cloud embedding APIs, no hosted vector DB here; just embed, store, and query.

Most local RAG setups reach for a hosted embedding API, and that is a reasonable choice for a lot of teams. rek0n does not, because sending code off-device is not something I am willing to do for a tool meant to read an entire repository. That decision means CPU inference through Candle, which is slower than a network call. `with_device` exists as an escape hatch for that, though a tuned GPU pipeline is not built yet.

## How it works

1. `LocalEmbedder::new` mmap's `model.safetensors` beside `config.json` and `tokenizer.json`. Tokenize → BERT forward → mean pool (mask-aware) → L2 normalize. Bulk indexing batches texts (default 32) per forward pass.
2. `VectorStorage::replace_file_chunks` validates chunks, embeds on `spawn_blocking`, then takes an exclusive file lock only for the LanceDB write. Existing rows for that `file_path` are deleted before the new batch is appended.
3. Vectors land as Arrow `FixedSizeList<Float32, 384>` record batches in a per-repo LanceDB table.
4. `query_semantic_context` embeds the query off-thread, takes a shared lock, and runs vector search. Results map back to `SemanticChunk` + score.
5. ANN index creation waits until ≥256 rows and is rebuilt after row deletes. Cross-process writers are serialized via `.rek0n-{repo}.lock` (advisory, one machine).

## Why it's built this way

**Local inference.** RAG over a codebase should not phone home. Candle runs the transformer on CPU with weights I fetch and checksum myself.

**Hard embed boundary.** rek0n-parser handles parse. This crate answers one question only: what are the vectors, and which chunks match? No hidden side effects.

**Mean pool, not CLS.** Matches sentence-transformers pooling. L2-normalized vectors let LanceDB L2 distance map to cosine similarity.

**Embed before lock.** Tokenization and forward passes are CPU-heavy. They stay on the blocking thread pool; the file lock covers only the LanceDB I/O.

**LanceDB on disk.** Embedded, Arrow-native storage without running a separate vector service. Enough for a single-machine indexer.

## Shortcomings

- CPU inference is the bottleneck on large repos; `with_device` exists but there is no tuned GPU pipeline yet.
- File locks are advisory and local: fine for one indexer per checkout, not a multi-machine coordination layer.
- ANN index creation waits until ≥256 rows; smaller tables use flat search.
- Model swap is not plug-and-play: tokenizer and architecture must stay BERT-compatible with Candle.
- `SemanticChunk` is embed-local (includes `file_path`); keep it aligned with parser output at the call site.

## Usage

```rust
use std::sync::Arc;
use rek0n_embed::{LocalEmbedder, VectorStorage, query_semantic_context};

let embedder = Arc::new(LocalEmbedder::new(
    "model/model.safetensors".as_ref(),
    "model/tokenizer.json".as_ref(),
)?);
let storage = VectorStorage::initialize("./lancedb", "my-repo").await?;
storage
    .replace_file_chunks("src/lib.rs", &chunks, Arc::clone(&embedder))
    .await?;
let hits = query_semantic_context(&storage, embedder, "how does auth work?", 10).await?;
```

See `examples/semantic_search.rs` for a full embed-and-search walkthrough:

```sh
./scripts/download_model.sh
cargo run --example semantic_search
```

## License

MIT
