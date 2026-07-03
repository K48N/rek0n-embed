#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT}/examples/model"
URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors"
TARGET="${MODEL_DIR}/model.safetensors"
EXPECTED_SHA256="53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
EXPECTED_SIZE=90868376

verify_checksum() {
  [[ -f "${TARGET}" ]] || return 1
  [[ "$(wc -c < "${TARGET}" | tr -d ' ')" -eq "${EXPECTED_SIZE}" ]] || return 1
  if command -v sha256sum >/dev/null 2>&1; then
    echo "${EXPECTED_SHA256}  ${TARGET}" | sha256sum -c --status
  elif command -v shasum >/dev/null 2>&1; then
    [[ "$(shasum -a 256 "${TARGET}" | awk '{print $1}')" == "${EXPECTED_SHA256}" ]]
  else
    echo "Install sha256sum or shasum to verify model weights." >&2
    return 1
  fi
}

mkdir -p "${MODEL_DIR}"

if [[ -f "${TARGET}" ]]; then
  if verify_checksum; then
    echo "Already present and verified: ${TARGET}"
    exit 0
  fi
  echo "Removing corrupt or outdated weights at ${TARGET}"
  rm -f "${TARGET}"
fi

echo "Downloading model.safetensors (~87 MB) to ${TARGET}"
if command -v curl >/dev/null 2>&1; then
  curl -L "${URL}" -o "${TARGET}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${TARGET}" "${URL}"
else
  echo "Install curl or wget to download the model weights." >&2
  exit 1
fi

if ! verify_checksum; then
  rm -f "${TARGET}"
  echo "Downloaded model.safetensors failed checksum verification" >&2
  exit 1
fi

echo "Download complete and verified."
