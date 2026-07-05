use rek0n_db::{AnnStrategy, DEFAULT_IVF_BUCKETS, DEFAULT_IVF_PROBE};

const MIN_ROWS_FOR_IVF: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorDistance {
    #[default]
    L2,
    Cosine,
    Dot,
}

#[derive(Debug, Clone, Default)]
pub struct IvfPqConfig {
    pub distance: VectorDistance,
    pub num_partitions: Option<u32>,
    pub num_sub_vectors: Option<u32>,
    pub sample_rate: Option<u32>,
    pub max_iterations: Option<u32>,
}

#[derive(Debug, Clone, Default)]
pub enum VectorIndexKind {
    #[default]
    Auto,
    IvfPq(IvfPqConfig),
}

#[derive(Debug, Clone, Default)]
pub struct VectorIndexConfig {
    pub kind: VectorIndexKind,
}

impl VectorIndexConfig {
    pub fn ivf_pq(config: IvfPqConfig) -> Self {
        Self {
            kind: VectorIndexKind::IvfPq(config),
        }
    }

    pub fn search_distance(&self) -> VectorDistance {
        match &self.kind {
            VectorIndexKind::Auto => VectorDistance::L2,
            VectorIndexKind::IvfPq(cfg) => cfg.distance,
        }
    }

    pub(crate) fn ann_strategy(&self, live_vectors: usize) -> AnnStrategy {
        match &self.kind {
            VectorIndexKind::Auto => {
                if live_vectors >= MIN_ROWS_FOR_IVF {
                    AnnStrategy::Ivf {
                        probe_buckets: DEFAULT_IVF_PROBE,
                    }
                } else {
                    AnnStrategy::Exact
                }
            }
            VectorIndexKind::IvfPq(cfg) => AnnStrategy::Ivf {
                probe_buckets: cfg
                    .num_partitions
                    .map(|value| value as usize)
                    .unwrap_or(DEFAULT_IVF_PROBE),
            },
        }
    }

    pub(crate) fn ivf_bucket_count(&self) -> usize {
        match &self.kind {
            VectorIndexKind::Auto => DEFAULT_IVF_BUCKETS,
            VectorIndexKind::IvfPq(cfg) => cfg
                .num_partitions
                .map(|value| value as usize)
                .unwrap_or(DEFAULT_IVF_BUCKETS),
        }
    }
}
