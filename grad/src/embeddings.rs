use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use crate::Tensor;

#[derive(Clone)]
#[derive(Serialize, Deserialize)]
pub struct SavedEmbeddings {
    embedding_dim: usize,
    vectors: Vec<Vec<f64>>,
}


pub struct Embeddings {
    embedding_dim: usize,
    vectors: Vec<Vec<Tensor>>,
}

impl Embeddings {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::rng();

        let normal = Normal::new(
            0.0,
            (1.0 / embedding_dim as f64).sqrt(),
        )
            .unwrap();

        let vectors = (0..vocab_size)
            .map(|_| {
                (0..embedding_dim)
                    .map(|_| Tensor::new(normal.sample(&mut rng)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self {
            embedding_dim,
            vectors,
        }
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn vocab_size(&self) -> usize {
        self.vectors.len()
    }

    pub fn parameter_count(&self) -> usize {
        self.embedding_dim * self.vectors.len()
    }

    pub fn encode(&self, ids: &[usize]) -> Vec<Tensor> {
        let mut out =
            Vec::with_capacity(ids.len() * self.embedding_dim);

        for &id in ids {
            out.extend(
                self.vectors[id]
                    .iter()
                    .cloned()
            );
        }
        out
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.vectors
            .iter()
            .flat_map(|row| row.iter().cloned())
            .collect()
    }

    pub fn save(&self) -> SavedEmbeddings {
        SavedEmbeddings {
            embedding_dim: self.embedding_dim,
            vectors: self.vectors
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|t| t.data())
                        .collect()
                })
                .collect(),
        }
    }

    pub fn load(saved: SavedEmbeddings) -> Self {
        Embeddings {
            embedding_dim: saved.embedding_dim,
            vectors: saved.vectors
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(Tensor::new)
                        .collect()
                })
                .collect(),
        }
    }
}
