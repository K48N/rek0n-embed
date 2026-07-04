use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fs4::fs_std::FileExt;

use crate::types::EmbedError;

const LOCK_POLL_INTERVAL: Duration = Duration::from_millis(50);
pub const DEFAULT_LOCK_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LockOptions {
    timeout: Option<Duration>,
}

impl LockOptions {
    pub fn blocking() -> Self {
        Self { timeout: None }
    }

    pub fn try_once() -> Self {
        Self {
            timeout: Some(Duration::ZERO),
        }
    }

    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            timeout: Some(timeout),
        }
    }
}

impl Default for LockOptions {
    fn default() -> Self {
        Self::with_timeout(DEFAULT_LOCK_TIMEOUT)
    }
}

pub(crate) fn lock_path(db_dir: &str, repo_name: &str) -> PathBuf {
    PathBuf::from(db_dir).join(format!(".rek0n-{repo_name}.lock"))
}

#[derive(Debug)]
pub(crate) struct SharedLock {
    file: File,
}

#[derive(Debug)]
pub(crate) struct ExclusiveLock {
    file: File,
}

impl SharedLock {
    pub(crate) fn acquire(path: PathBuf, options: LockOptions) -> Result<Self, EmbedError> {
        let file = acquire_lock_file(path, options, LockMode::Shared)?;
        Ok(Self { file })
    }
}

impl ExclusiveLock {
    pub(crate) fn acquire(path: PathBuf, options: LockOptions) -> Result<Self, EmbedError> {
        let file = acquire_lock_file(path, options, LockMode::Exclusive)?;
        Ok(Self { file })
    }
}

impl Drop for SharedLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

impl Drop for ExclusiveLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

enum LockMode {
    Shared,
    Exclusive,
}

fn acquire_lock_file(
    path: PathBuf,
    options: LockOptions,
    mode: LockMode,
) -> Result<File, EmbedError> {
    let deadline = options.timeout.map(|timeout| Instant::now() + timeout);

    loop {
        let file = open_lock_file(&path)?;
        let result = match mode {
            LockMode::Shared => FileExt::try_lock_shared(&file),
            LockMode::Exclusive => FileExt::try_lock_exclusive(&file),
        };

        match result {
            Ok(()) => return Ok(file),
            Err(source) if is_lock_contended(&source) => {
                if let Some(deadline) = deadline {
                    if Instant::now() >= deadline {
                        return Err(EmbedError::LockTimeout {
                            path: path.display().to_string(),
                        });
                    }
                }
                std::thread::sleep(LOCK_POLL_INTERVAL);
            }
            Err(source) => return Err(EmbedError::io_path(&path, source)),
        }
    }
}

fn is_lock_contended(source: &std::io::Error) -> bool {
    source.kind() == ErrorKind::WouldBlock
        || source.raw_os_error() == fs4::lock_contended_error().raw_os_error()
}

fn open_lock_file(path: &Path) -> Result<File, EmbedError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| EmbedError::io_path(parent, source))?;
    }

    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|source| EmbedError::io_path(path, source))
}

pub(crate) async fn acquire_shared(
    path: PathBuf,
    options: LockOptions,
) -> Result<SharedLock, EmbedError> {
    tokio::task::spawn_blocking(move || SharedLock::acquire(path, options)).await?
}

pub(crate) async fn acquire_exclusive(
    path: PathBuf,
    options: LockOptions,
) -> Result<ExclusiveLock, EmbedError> {
    tokio::task::spawn_blocking(move || ExclusiveLock::acquire(path, options)).await?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_parent_directory_for_lock_file() {
        let base = tempfile::tempdir().expect("tempdir");
        let nested = base.path().join("nested/lancedb");
        let lock = lock_path(nested.to_str().expect("utf8"), "demo");
        open_lock_file(&lock).expect("open lock file");
        assert!(lock.parent().expect("parent").is_dir());
    }

    #[test]
    fn try_once_fails_when_lock_held() {
        let base = tempfile::tempdir().expect("tempdir");
        let lock = lock_path(base.path().to_str().expect("utf8"), "demo");

        let _held = ExclusiveLock::acquire(lock.clone(), LockOptions::blocking()).expect("lock");

        let err = ExclusiveLock::acquire(lock, LockOptions::try_once()).expect_err("busy");
        assert!(matches!(err, EmbedError::LockTimeout { .. }));
    }
}
