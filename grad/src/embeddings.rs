use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use crate::Tensor;

#[derive(Clone)]
#[derive(Serialize, Deserialize)]
pub struct SavedEmbeddings {
    embedding_dim: usize,
    vectors: Vec<Vec<f32>>,
}

#[derive(Clone)]
pub struct Embeddings {
    embedding_dim: usize,
    vectors: Vec<Vec<Tensor>>,
}

impl Embeddings {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::rng();

        let normal = Normal::new(
            0.0,
            (1.0 / embedding_dim as f32).sqrt(),
        )
            .unwrap();

        let mut vectors = (0..vocab_size)
            .map(|_| {
                (0..embedding_dim)
                    .map(|_| Tensor::new(normal.sample(&mut rng)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        vectors[1] = (0..embedding_dim)
            .map(|_| Tensor::new(0.0))
            .collect();

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

    pub(crate) fn encode_batch(
        &self,
        ids: &[usize],
        batch_size: usize,
        context_len: usize,
    ) -> Vec<f32> {
        let input_size =
            context_len * self.embedding_dim;

        let mut output =
            vec![0.0f32; batch_size * input_size];

        for b in 0..batch_size {
            let sample =
                &ids[b * context_len..(b + 1) * context_len];

            let dst =
                &mut output[
                    b * input_size
                        ..(b + 1) * input_size
                    ];

            for (position, &id) in sample.iter().enumerate() {
                let src = &self.vectors[id];

                let offset =
                    position * self.embedding_dim;

                for i in 0..self.embedding_dim {
                    dst[offset + i] = src[i].data();
                }
            }
        }
        output
    }

    pub(crate) fn accumulate_batch_grads(
        &self,
        ids: &[usize],
        input_grads: &[f32],
        batch_size: usize,
        context_len: usize,
    ) {
        let input_size =
            context_len * self.embedding_dim;

        let mut grads = Vec::new();

        for b in 0..batch_size {
            for position in 0..context_len {
                let token = ids[
                    b * context_len + position
                    ];

                let input_offset =
                    b * input_size
                        + position * self.embedding_dim;

                let embedding =
                    &self.vectors[token];

                for i in 0..self.embedding_dim {
                    grads.push((
                        embedding[i].handle,
                        input_grads[input_offset + i], // here
                    ));
                }
            }
        }

        crate::add_handle_grads(&grads);
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

    pub fn find_clusters(&self, threshold: f32, vocab: Vec<String>) {
        let n = self.vectors.len();

        let mut norms = vec![0.0; n];

        for (norm, vector) in norms.iter_mut().zip(&self.vectors).take(n) {
            *norm = vector
                .iter()
                .map(|t| {
                    let x = t.data();
                    x * x
                })
                .sum::<f32>()
                .sqrt();
        }

        let cosine = |a: usize, b: usize| -> f32 {
            let dot = self.vectors[a]
                .iter()
                .zip(self.vectors[b].iter())
                .map(|(x, y)| x.data() * y.data())
                .sum::<f32>();

            dot / (norms[a] * norms[b] + 1e-12)
        };

        let mut graph = vec![Vec::<usize>::new(); n];

        let mut max_sim = -1.0;
        let mut max_pair = (0, 0);

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine(i, j);

                if sim > max_sim {
                    max_sim = sim;
                    max_pair = (i, j);
                }

                if sim >= threshold {
                    graph[i].push(j);
                    graph[j].push(i);
                }
            }
        }

        println!(
            "Highest similarity: {:.5} between {} and {}",
            max_sim,
            max_pair.0,
            max_pair.1
        );

        let mut visited = vec![false; n];
        let mut clusters = 0;

        for start in 0..n {
            if visited[start] {
                continue;
            }

            let mut stack = vec![start];
            let mut cluster = Vec::new();

            visited[start] = true;

            while let Some(node) = stack.pop() {
                cluster.push(node);

                for &next in &graph[node] {
                    if !visited[next] {
                        visited[next] = true;
                        stack.push(next);
                    }
                }
            }

            if cluster.len() > 1 {
                clusters += 1;

                println!("\nCluster {} ({} tokens):", clusters, cluster.len());

                for id in cluster {
                    println!("  token {} ({})", id, vocab[id]);
                }
            }
        }

        println!("\nTotal clusters: {}", clusters);
    }
}

#[derive(Deserialize)]
pub struct OldSavedEmbeddings {
    embedding_dim: usize,
    vectors: Vec<Vec<f64>>,
}

impl From<OldSavedEmbeddings> for SavedEmbeddings {
    fn from(old: OldSavedEmbeddings) -> Self {
        Self {
            embedding_dim: old.embedding_dim,
            vectors: old.vectors
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|x| x as f32)
                        .collect()
                })
                .collect(),
        }
    }
}
