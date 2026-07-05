use std::time::Duration;

pub const DEFAULT_LOCK_TIMEOUT: Duration = Duration::from_secs(30);

/// Options for acquiring an exclusive lock on a rek0n-db store directory.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_uses_standard_timeout() {
        assert_eq!(
            LockOptions::default(),
            LockOptions::with_timeout(DEFAULT_LOCK_TIMEOUT)
        );
    }
}
