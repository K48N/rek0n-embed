use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::index::Index;
use lancedb::DistanceType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorDistance {
    #[default]
    L2,
    Cosine,
    Dot,
}

impl From<VectorDistance> for DistanceType {
    fn from(value: VectorDistance) -> Self {
        match value {
            VectorDistance::L2 => DistanceType::L2,
            VectorDistance::Cosine => DistanceType::Cosine,
            VectorDistance::Dot => DistanceType::Dot,
        }
    }
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
}

pub(crate) fn to_lancedb_index(config: &VectorIndexConfig) -> Index {
    match &config.kind {
        VectorIndexKind::Auto => Index::Auto,
        VectorIndexKind::IvfPq(cfg) => {
            let mut builder = IvfPqIndexBuilder::default().distance_type(cfg.distance.into());
            if let Some(num_partitions) = cfg.num_partitions {
                builder = builder.num_partitions(num_partitions);
            }
            if let Some(num_sub_vectors) = cfg.num_sub_vectors {
                builder = builder.num_sub_vectors(num_sub_vectors);
            }
            if let Some(sample_rate) = cfg.sample_rate {
                builder = builder.sample_rate(sample_rate);
            }
            if let Some(max_iterations) = cfg.max_iterations {
                builder = builder.max_iterations(max_iterations);
            }
            Index::IvfPq(builder)
        }
    }
}
