$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ModelDir = Join-Path $Root "examples\model"
$Url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors"
$Target = Join-Path $ModelDir "model.safetensors"
$ExpectedHash = "53AA51172D142C89D9012CCE15AE4D6CC0CA6895895114379CACB4FAB128D9DB"
$ExpectedSize = 90868376

function Test-ModelChecksum {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $false }
    $info = Get-Item $Path
    if ($info.Length -ne $ExpectedSize) {
        Write-Warning "Unexpected file size: $($info.Length) (expected $ExpectedSize)"
        return $false
    }
    $hash = (Get-FileHash -Path $Path -Algorithm SHA256).Hash
    if ($hash -ne $ExpectedHash) {
        Write-Warning "Checksum mismatch: $hash"
        return $false
    }
    return $true
}

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

if (Test-Path $Target) {
    if (Test-ModelChecksum -Path $Target) {
        Write-Host "Already present and verified: $Target"
        exit 0
    }
    Write-Host "Removing corrupt or outdated weights at $Target"
    Remove-Item -Force $Target
}

Write-Host "Downloading model.safetensors (~87 MB) to $Target"
Invoke-WebRequest -Uri $Url -OutFile $Target

if (-not (Test-ModelChecksum -Path $Target)) {
    Remove-Item -Force $Target
    throw "Downloaded model.safetensors failed checksum verification"
}

Write-Host "Download complete and verified."
