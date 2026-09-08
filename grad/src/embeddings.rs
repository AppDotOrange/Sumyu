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

    // Flat: [token0_dim0 ... token0_dimD, token1_dim0 ...]
    vectors: Vec<Tensor>,

    // Contiguous values used by BLAS / batch encoding.
    flat_values: RefCell<Vec<f32>>,

    // First embedding parameter node in the tape.
    node_start: usize,
}

impl Embeddings {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::rng();

        let normal = Normal::new(
            0.0,
            (1.0 / embedding_dim as f32).sqrt(),
        )
            .unwrap();

        let node_start = crate::tape_len();

        let mut vectors =
            Vec::with_capacity(vocab_size * embedding_dim);

        for token in 0..vocab_size {
            for _ in 0..embedding_dim {
                let value = if token == 0 {
                    0.0
                } else {
                    normal.sample(&mut rng)
                };

                vectors.push(Tensor::new(value));
            }
        }

        let flat_values =
            vectors
                .iter()
                .map(|t| t.data())
                .collect::<Vec<f32>>();

        Self {
            embedding_dim,
            vectors,
            flat_values: RefCell::new(flat_values),
            node_start,
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
        self.vectors.len() / self.embedding_dim
    }

    #[inline]
    pub fn parameter_count(&self) -> usize {
        self.vectors.len()
    }

    #[inline]
    pub fn encode(&self, ids: &[u16]) -> Vec<Tensor> {
        let mut out =
            Vec::with_capacity(ids.len() * self.embedding_dim);

        for &id in ids {
            let start = id as usize * self.embedding_dim;

            let end = start + self.embedding_dim;

            out.extend_from_slice(
                &self.vectors[start..end]
            );
        }

        out
    }

    pub(crate) fn encode_batch_into(
        &self,
        ids: &[u16],
        batch_size: usize,
        context_len: usize,
        output: &mut Vec<f32>,
    ) {
        let input_size =
            context_len * self.embedding_dim;

        let required_len =
            batch_size * input_size;

        if output.len() != required_len {
            output.resize(required_len, 0.0);
        }

        let flat_values =
            self.flat_values.borrow();

        for b in 0..batch_size {
            let sample_start =
                b * context_len;

            let dst_batch_start =
                b * input_size;

            let dst_batch =
                &mut output[
                    dst_batch_start
                        ..dst_batch_start + input_size
                    ];

            for position in 0..context_len {
                let token =
                    ids[sample_start + position]
                        as usize;

                let src_start =
                    token * self.embedding_dim;

                let src =
                    &flat_values[
                        src_start
                            ..src_start + self.embedding_dim
                        ];

                let dst_start =
                    position * self.embedding_dim;

                let dst =
                    &mut dst_batch[
                        dst_start
                            ..dst_start + self.embedding_dim
                        ];

                dst.copy_from_slice(src);
            }
        }
    }

    pub(crate) fn encode_batch(
        &self,
        ids: &[u16],
        batch_size: usize,
        context_len: usize,
    ) -> Vec<f32> {
        let mut output = Vec::new();

        self.encode_batch_into(
            ids,
            batch_size,
            context_len,
            &mut output,
        );

        output
    }

    pub(crate) fn accumulate_batch_grads(
        &self,
        ids: &[u16],
        input_grads: &[f32],
        batch_size: usize,
        context_len: usize,
    ) {
        let input_size =
            context_len * self.embedding_dim;

        debug_assert_eq!(
            ids.len(),
            batch_size * context_len
        );

        debug_assert_eq!(
            input_grads.len(),
            batch_size * input_size
        );

        TAPE.with(|t| {
            let mut tape = t.borrow_mut();

            for b in 0..batch_size {
                let sample_start =
                    b * context_len;

                let grad_start =
                    b * input_size;

                for position in 0..context_len {
                    let token = ids[sample_start + position] as usize;

                    let embedding_start =
                        token * self.embedding_dim;

                    let grad_start =
                        grad_start
                            + position * self.embedding_dim;

                    for i in 0..self.embedding_dim {
                        let node_id =
                            self.node_start
                                + embedding_start
                                + i;

                        let grad =
                            input_grads[grad_start + i];

                        if let Node::Scalar(node) =
                            &mut tape.nodes[node_id]
                        {
                            node.grad += grad;
                        } else {
                            unreachable!(
                                "Embedding node is not Scalar"
                            );
                        }
                    }
                }
            }
        });
    }

    #[inline]
    pub fn parameters(&self) -> Vec<Tensor> {
        self.vectors.clone()
    }

    #[inline]
    pub fn parameter_handles(&self) -> Vec<TensorHandle> {
        (0..self.vectors.len())
            .map(|i| TensorHandle {
                node: self.node_start + i,
                index: 0,
            })
            .collect()
    }

    pub fn save(&self) -> SavedEmbeddings {
        let vectors =
            self.vectors
                .chunks_exact(self.embedding_dim)
                .map(|row| {
                    row.iter()
                        .map(|t| t.data())
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<Vec<f32>>>();

        SavedEmbeddings {
            embedding_dim: self.embedding_dim,
            vectors,
        }
    }

    pub fn load(saved: SavedEmbeddings) -> Self {
        let embedding_dim =
            saved.embedding_dim;

        let node_start =
            crate::tape_len();

        let total =
            saved.vectors.len()
                * embedding_dim;

        let mut vectors =
            Vec::with_capacity(total);

        for row in saved.vectors {
            debug_assert_eq!(
                row.len(),
                embedding_dim
            );

            for value in row {
                vectors.push(Tensor::new(value));
            }
        }

        let flat_values =
            vectors
                .iter()
                .map(|t| t.data())
                .collect::<Vec<f32>>();

        Self {
            embedding_dim,
            vectors,
            flat_values: RefCell::new(flat_values),
            node_start,
        }
    }

    #[inline]
    pub(crate) fn node_start(&self) -> usize {
        self.node_start
    }

    #[inline]
    pub(crate) fn flat_values_mut(
        &self,
    ) -> std::cell::RefMut<'_, Vec<f32>> {
        self.flat_values.borrow_mut()
    }

    #[inline]
    pub(crate) fn accumulate_flat_grads(&self, grads: &[f32]) {
        assert_eq!(
            grads.len(),
            self.parameter_count(),
            "Invalid embedding gradient length"
        );

        let start = self.node_start;

        TAPE.with(|t| {
            let mut tape = t.borrow_mut();

            for (i, &grad) in grads.iter().enumerate() {
                let node_id = start + i;

                match &mut tape.nodes[node_id] {
                    Node::Scalar(node) => {
                        node.grad += grad;
                    }

                    Node::FusedLayer(_) => {
                        unreachable!(
                            "Embedding parameter is a fused-layer output"
                        );
                    }
                }
            }
        });
    }

    pub fn find_clusters(&self, threshold: f32, vocab: Vec<String>) {
        let n =
            self.vectors.len()
                / self.embedding_dim;

        if n < 2 {
            println!("Highest similarity: N/A");
            println!("\nTotal clusters: 0");
            return;
        }

        let dim =
            self.embedding_dim;

        let mut normalized =
            vec![0.0f32; n * dim];

        for id in 0..n {
            let start =
                id * dim;

            let src =
                &self.vectors[start..start + dim];

            let dst =
                &mut normalized[start..start + dim];

            let mut norm_sq =
                0.0f32;

            for i in 0..dim {
                let x =
                    src[i].data();

                dst[i] = x;
                norm_sq += x * x;
            }

            if norm_sq > 0.0 {
                let inv_norm =
                    norm_sq.sqrt().recip();

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
