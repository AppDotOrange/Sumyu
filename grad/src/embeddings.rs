use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::parameters::{
    ParamRange,
    ParameterStore,
};

#[derive(Clone, Serialize, Deserialize)]
pub struct SavedEmbeddings {
    embedding_dim: usize,
    vectors: Vec<Vec<f32>>,
}

#[derive(Clone)]
pub struct Embeddings {
    embedding_dim: usize,

    // Contiguous embedding parameters inside the model's
    // global ParameterStore.
    //
    // Layout:
    //
    // [token0_dim0 ... token0_dimD,
    //  token1_dim0 ... token1_dimD, ...]
    parameter_range: ParamRange,
}

impl Embeddings {
    pub fn new(
        params: &mut ParameterStore,
        vocab_size: usize,
        embedding_dim: usize,
    ) -> Self {
        assert!(
            vocab_size > 0,
            "Embedding vocabulary size must be > 0"
        );

        assert!(
            embedding_dim > 0,
            "Embedding dimension must be > 0"
        );

        let mut rng = rand::rng();

        let normal = Normal::new(
            0.0,
            (1.0 / embedding_dim as f32).sqrt(),
        )
            .unwrap();

        let parameter_range =
            params.alloc_many(
                (0..vocab_size * embedding_dim)
                    .map(|index| {
                        let token =
                            index / embedding_dim;

                        if token == 0 {
                            0.0
                        } else {
                            normal.sample(&mut rng)
                        }
                    }),
            );

        Self {
            embedding_dim,
            parameter_range,
        }
    }

    pub fn load_into(
        params: &mut ParameterStore,
        saved: SavedEmbeddings,
    ) -> Self {
        let embedding_dim =
            saved.embedding_dim;

        assert!(
            embedding_dim > 0,
            "Saved embedding dimension must be > 0"
        );

        let vocab_size =
            saved.vectors.len();

        assert!(
            vocab_size > 0,
            "Saved embedding vocabulary must not be empty"
        );

        let mut flat =
            Vec::with_capacity(
                vocab_size
                    * embedding_dim
            );

        for row in saved.vectors {
            assert_eq!(
                row.len(),
                embedding_dim,
                "Invalid saved embedding row length"
            );

            flat.extend_from_slice(&row);
        }

        let parameter_range =
            params.alloc_many(
                flat.into_iter()
            );

        Self {
            embedding_dim,
            parameter_range,
        }
    }

    #[inline(always)]
    pub(crate) fn parameter_range(
        &self,
    ) -> ParamRange {
        self.parameter_range
    }

    #[inline(always)]
    pub fn embedding_dim(
        &self,
    ) -> usize {
        self.embedding_dim
    }

    #[inline(always)]
    pub fn vocab_size(
        &self,
    ) -> usize {
        self.parameter_range.len
            / self.embedding_dim
    }

    #[inline(always)]
    pub fn parameter_count(
        &self,
    ) -> usize {
        self.parameter_range.len
    }

    // ---------------------------------------------------------------------
    // Non-autograd single-sequence encoding.
    //
    // This now returns ordinary f32 values.
    // ---------------------------------------------------------------------

    #[inline]
    pub fn encode(
        &self,
        params: &ParameterStore,
        ids: &[u16],
    ) -> Vec<f32> {
        let values =
            params.values(
                self.parameter_range
            );

        let mut out =
            Vec::with_capacity(
                ids.len()
                    * self.embedding_dim
            );

        for &id in ids {
            let start =
                id as usize
                    * self.embedding_dim;

            let end =
                start + self.embedding_dim;

            assert!(
                end <= values.len(),
                "Embedding token id out of bounds"
            );

            out.extend_from_slice(
                &values[start..end]
            );
        }

        out
    }

    #[inline]
    pub(crate) fn encode_batch_into(
        &self,
        params: &ParameterStore,
        ids: &[u16],
        batch_size: usize,
        context_len: usize,
        output: &mut Vec<f32>,
    ) {
        let input_size =
            context_len
                * self.embedding_dim;

        let required_len =
            batch_size
                * input_size;

        assert_eq!(
            ids.len(),
            batch_size
                * context_len,
            "Embedding batch id count mismatch"
        );

        if output.len() != required_len {
            output.resize(
                required_len,
                0.0
            );
        }

        let values =
            params.values(
                self.parameter_range
            );

        for b in 0..batch_size {
            let sample_start =
                b * context_len;

            let dst_batch_start =
                b * input_size;

            let dst_batch =
                &mut output[
                    dst_batch_start
                        ..dst_batch_start
                        + input_size
                    ];

            for position in 0..context_len {
                let token =
                    ids[
                        sample_start
                            + position
                        ] as usize;

                let src_start =
                    token
                        * self.embedding_dim;

                let src_end =
                    src_start
                        + self.embedding_dim;

                assert!(
                    src_end <= values.len(),
                    "Embedding token id out of bounds"
                );

                let src =
                    &values[
                        src_start..src_end
                        ];

                let dst_start =
                    position
                        * self.embedding_dim;

                let dst =
                    &mut dst_batch[
                        dst_start
                            ..dst_start
                            + self.embedding_dim
                        ];

                dst.copy_from_slice(
                    src
                );
            }
        }
    }

    #[inline]
    pub(crate) fn encode_batch(
        &self,
        params: &ParameterStore,
        ids: &[u16],
        batch_size: usize,
        context_len: usize,
    ) -> Vec<f32> {
        let mut output =
            Vec::new();

        self.encode_batch_into(
            params,
            ids,
            batch_size,
            context_len,
            &mut output,
        );

        output
    }

    // ---------------------------------------------------------------------
    // Embedding gradient accumulation.
    //
    // No tape.
    // No handles.
    // Directly accumulates into ParameterStore.grads.
    // ---------------------------------------------------------------------

    pub(crate) fn accumulate_batch_grads(
        &self,
        params: &mut ParameterStore,
        ids: &[u16],
        input_grads: &[f32],
        batch_size: usize,
        context_len: usize,
    ) {
        let input_size =
            context_len
                * self.embedding_dim;

        debug_assert_eq!(
            ids.len(),
            batch_size
                * context_len
        );

        debug_assert_eq!(
            input_grads.len(),
            batch_size
                * input_size
        );

        let grads =
            params.grads_mut(
                self.parameter_range
            );

        for b in 0..batch_size {
            let sample_start =
                b * context_len;

            let batch_grad_start =
                b * input_size;

            for position in 0..context_len {
                let token =
                    ids[
                        sample_start
                            + position
                        ] as usize;

                let embedding_start =
                    token
                        * self.embedding_dim;

                let grad_start =
                    batch_grad_start
                        + position
                        * self.embedding_dim;

                for i in 0..self.embedding_dim {
                    grads[
                        embedding_start + i
                        ] += input_grads[
                        grad_start + i
                        ];
                }
            }
        }
    }

    pub fn save(
        &self,
        params: &ParameterStore,
    ) -> SavedEmbeddings {
        let values =
            params.values(
                self.parameter_range
            );

        let vectors =
            values
                .chunks_exact(
                    self.embedding_dim
                )
                .map(|row| row.to_vec())
                .collect();

        SavedEmbeddings {
            embedding_dim:
            self.embedding_dim,
            vectors,
        }
    }

    pub fn find_clusters(
        &self,
        params: &ParameterStore,
        threshold: f32,
        vocab: Vec<String>,
    ) {
        let n =
            self.vocab_size();

        if n < 2 {
            println!(
                "Highest similarity: N/A"
            );

            println!(
                "\nTotal clusters: 0"
            );

            return;
        }

        assert!(
            vocab.len() >= n,
            "Vocabulary is smaller than the embedding vocabulary"
        );

        let dim =
            self.embedding_dim;

        let values =
            params.values(
                self.parameter_range
            );

        let mut normalized =
            vec![
                0.0f32;
                n * dim
            ];

        for id in 0..n {
            let start =
                id * dim;

            let src =
                &values[
                    start..start + dim
                    ];

            let dst =
                &mut normalized[
                    start..start + dim
                    ];

            let mut norm_sq =
                0.0f32;

            for i in 0..dim {
                let x =
                    src[i];

                dst[i] =
                    x;

                norm_sq +=
                    x * x;
            }

            if norm_sq > 0.0 {
                let inv_norm =
                    norm_sq
                        .sqrt()
                        .recip();

                for x in dst {
                    *x *= inv_norm;
                }
            }
        }

        let mut parent:
            Vec<usize> =
            (0..n).collect();

        let mut size =
            vec![1usize; n];

        fn find(
            parent: &mut [usize],
            mut x: usize,
        ) -> usize {
            while parent[x] != x {
                parent[x] =
                    parent[parent[x]];

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
            let mut root_a =
                find(parent, a);

            let mut root_b =
                find(parent, b);

            if root_a == root_b {
                return;
            }

            if size[root_a]
                < size[root_b]
            {
                std::mem::swap(
                    &mut root_a,
                    &mut root_b,
                );
            }

            parent[root_b] =
                root_a;

            size[root_a] +=
                size[root_b];
        }

        let mut max_sim =
            -1.0f32;

        let mut max_pair =
            (0usize, 0usize);

        for i in 0..n {
            let a =
                &normalized[
                    i * dim
                        ..(i + 1) * dim
                    ];

            for j in (i + 1)..n {
                let b =
                    &normalized[
                        j * dim
                            ..(j + 1) * dim
                        ];

                let mut dot =
                    0.0f32;

                for k in 0..dim {
                    dot +=
                        a[k] * b[k];
                }

                let sim =
                    dot;

                if sim > max_sim {
                    max_sim =
                        sim;

                    max_pair =
                        (i, j);
                }

                if sim >= threshold {
                    union(
                        &mut parent,
                        &mut size,
                        i,
                        j,
                    );
                }
            }
        }

        println!(
            "Highest similarity: {:.5} between {} and {}",
            max_sim,
            max_pair.0,
            max_pair.1
        );

        let mut cluster_members:
            Vec<Vec<usize>> =
            Vec::new();

        let mut cluster_index =
            vec![usize::MAX; n];

        for id in 0..n {
            let root =
                find(
                    &mut parent,
                    id,
                );

            if cluster_index[root]
                == usize::MAX
            {
                cluster_index[root] =
                    cluster_members.len();

                cluster_members
                    .push(Vec::new());
            }

            cluster_members[
                cluster_index[root]
                ]
                .push(id);
        }

        let mut clusters =
            0;

        for cluster
        in cluster_members
        {
            if cluster.len() <= 1 {
                continue;
            }

            clusters += 1;

            println!(
                "\nCluster {} ({} tokens):",
                clusters,
                cluster.len()
            );

            for id in cluster {
                println!(
                    "  token {} ({})",
                    id,
                    vocab[id]
                );
            }
        }

        println!(
            "\nTotal clusters: {}",
            clusters
        );
    }
}
