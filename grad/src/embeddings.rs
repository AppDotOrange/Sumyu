use std::cell::RefCell;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use crate::{Node, Tensor, TensorHandle, TAPE};

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
    flat_values: RefCell<Vec<f32>>,
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
        vectors[0] = (0..embedding_dim)
            .map(|_| Tensor::new(0.0))
            .collect();

        let flat_values = vectors
            .iter()
            .flat_map(|row| row.iter().map(|t| t.data()))
            .collect();

        Self {
            embedding_dim,
            vectors,
            flat_values: RefCell::new(flat_values),
        }
    }

    #[inline]
    pub(crate) fn flat_values(&self) -> std::cell::Ref<'_, [f32]> {
        std::cell::Ref::map(
            self.flat_values.borrow(),
            |values| values.as_slice(),
        )
    }

    #[inline]
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vectors.len()
    }

    #[inline]
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
        let input_size = context_len * self.embedding_dim;
        let mut output = vec![0.0f32; batch_size * input_size];

        for b in 0..batch_size {
            let sample_start = b * context_len;
            let dst = &mut output[b * input_size..(b + 1) * input_size];

            for position in 0..context_len {
                let src = &self.vectors[ids[sample_start + position]];
                let dst = &mut dst[
                    position * self.embedding_dim
                        ..(position + 1) * self.embedding_dim
                    ];

                for (dst, src) in dst.iter_mut().zip(src.iter()) {
                    *dst = src.data();
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
        let input_size = context_len * self.embedding_dim;

        TAPE.with(|t| {
            let mut tape = t.borrow_mut();

            for b in 0..batch_size {
                let sample = &ids[
                    b * context_len..(b + 1) * context_len
                    ];

                let batch_grads = &input_grads[
                    b * input_size..(b + 1) * input_size
                    ];

                for (position, &token) in sample.iter().enumerate() {
                    let embedding = &self.vectors[token];

                    let start = position * self.embedding_dim;
                    let input = &batch_grads[
                        start..start + self.embedding_dim
                        ];

                    for (embedding, &grad) in
                        embedding.iter().zip(input.iter())
                    {
                        match &mut tape.nodes[embedding.handle.node] {
                            Node::Scalar(node) => {
                                node.grad += grad;
                            }

                            Node::FusedLayer(node) => {
                                node.grads[embedding.handle.index] += grad;
                            }
                        }
                    }
                }
            }
        });
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.vectors
            .iter()
            .flat_map(|row| row.iter().cloned())
            .collect()
    }

    pub fn parameter_handles(&self) -> Vec<TensorHandle> {
        self.vectors
            .iter()
            .flat_map(|row| row.iter().map(|t| t.handle))
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
        let embedding_dim = saved.embedding_dim;

        let vectors = saved.vectors
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(Tensor::new)
                    .collect()
            })
            .collect::<Vec<Vec<Tensor>>>();

        let flat_values = vectors
            .iter()
            .flat_map(|row| row.iter().map(|t| t.data()))
            .collect();

        Embeddings {
            embedding_dim,
            vectors,
            flat_values: RefCell::new(flat_values),
        }
    }

    pub(crate) fn sync_flat_values(&self) {
        let mut flat_values = self.flat_values.borrow_mut();

        debug_assert_eq!(
            flat_values.len(),
            self.parameter_count()
        );

        let mut offset = 0;

        for row in &self.vectors {
            for tensor in row {
                flat_values[offset] = tensor.data();
                offset += 1;
            }
        }
    }

    pub(crate) fn accumulate_flat_grads(&self, grads: &[f32]) {
        assert_eq!(
            grads.len(),
            self.parameter_count(),
            "Invalid embedding gradient length"
        );

        TAPE.with(|t| {
            let mut tape = t.borrow_mut();

            let mut offset = 0;

            for row in &self.vectors {
                for embedding in row {
                    let grad = grads[offset];

                    match &mut tape.nodes[embedding.handle.node] {
                        Node::Scalar(node) => {
                            node.grad += grad;
                        }

                        Node::FusedLayer(node) => {
                            node.grads[embedding.handle.index] += grad;
                        }
                    }

                    offset += 1;
                }
            }
        });
    }

    pub fn find_clusters(&self, threshold: f32, vocab: Vec<String>) {
        let n = self.vectors.len();
        if n < 2 {
            println!("Highest similarity: N/A");
            println!("\nTotal clusters: 0");
            return;
        }
        let dim = self.embedding_dim;
        // Flatten and normalize embeddings once.
        // After this, cosine similarity is just a dot product.
        let mut normalized = vec![0.0f32; n * dim];
        for id in 0..n {
            let src = &self.vectors[id];
            let dst = &mut normalized[id * dim..(id + 1) * dim];
            let mut norm_sq = 0.0f32;
            for i in 0..dim {
                let x = src[i].data();
                dst[i] = x;
                norm_sq += x * x;
            }
            if norm_sq > 0.0 {
                let inv_norm = norm_sq.sqrt().recip();
                for x in dst {
                    *x *= inv_norm;
                }
            }
        }
        // Union-find / disjoint-set structure.
        let mut parent: Vec<usize> = (0..n).collect();
        let mut size = vec![1usize; n];
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        fn union(
            parent: &mut [usize],
            size: &mut [usize],
            a: usize,
            b: usize,
        ) {
            let mut root_a = find(parent, a);
            let mut root_b = find(parent, b);
            if root_a == root_b { return; }
            // Union by size.
            if size[root_a] < size[root_b] {
                std::mem::swap(&mut root_a, &mut root_b);
            }
            parent[root_b] = root_a;
            size[root_a] += size[root_b];
        }
        let mut max_sim = -1.0f32;
        let mut max_pair = (0usize, 0usize);
        // Exact O(n² * dim) cosine search, but with:
        // - no repeated Tensor::data() calls
        // - no sqrt/division per comparison
        // - no graph allocation
        // - contiguous f32 memory
        for i in 0..n {
            let a = &normalized[i * dim..(i + 1) * dim];

            for j in (i + 1)..n {
                let b = &normalized[j * dim..(j + 1) * dim];

                let mut dot = 0.0f32;

                for k in 0..dim {
                    dot += a[k] * b[k];
                }

                let sim = dot;

                if sim > max_sim {
                    max_sim = sim;
                    max_pair = (i, j);
                }
                if sim >= threshold {
                    union(&mut parent, &mut size, i, j);
                }
            }
        }
        println!(
            "Highest similarity: {:.5} between {} and {}",
            max_sim,
            max_pair.0,
            max_pair.1
        );
        // Collect connected components.
        let mut cluster_members: Vec<Vec<usize>> = Vec::new();
        let mut cluster_index = vec![usize::MAX; n];
        for id in 0..n {
            let root = find(&mut parent, id);
            if cluster_index[root] == usize::MAX {
                cluster_index[root] = cluster_members.len();
                cluster_members.push(Vec::new());
            }
            cluster_members[cluster_index[root]].push(id);
        }
        let mut clusters = 0;
        for cluster in cluster_members {
            if cluster.len() <= 1 { continue }
            clusters += 1;
            println!(
                "\nCluster {} ({} tokens):",
                clusters,
                cluster.len()
            );
            for id in cluster {
                println!("  token {} ({})", id, vocab[id]);
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
