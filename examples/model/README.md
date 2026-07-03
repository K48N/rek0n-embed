# Example model weights

`config.json` and `tokenizer.json` are committed. Fetch `model.safetensors` from the repo root:

```sh
./scripts/download_model.sh      # Unix
./scripts/download_model.ps1     # Windows
```

Checksum is pinned in `model.sha256`. Then: `cargo run --example semantic_search`
