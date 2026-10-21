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
        // Number of nearest neighbours each token is allowed to nominate.
        //
        // 8 is a good default for a ~2k vocabulary:
        // large enough to find semantic neighbourhoods, small enough
        // to prevent giant "everything is vaguely related" clusters.
        const K: usize = 8;

        // Don't print hundreds of members for a giant cluster.
        const MAX_PRINTED_MEMBERS: usize = 24;

        let n = self.vocab_size();

        if n < 2 {
            println!("Highest similarity: N/A");
            println!("\nTotal clusters: 0");
            return;
        }

        assert!(
            vocab.len() >= n,
            "Vocabulary is smaller than the embedding vocabulary"
        );

        assert!(
            threshold >= -1.0 && threshold <= 1.0,
            "Cosine similarity threshold must be in [-1, 1]"
        );

        let dim = self.embedding_dim;

        let values = params.values(self.parameter_range);

        // -------------------------------------------------------------
        // Normalize every embedding.
        // -------------------------------------------------------------

        let mut normalized = vec![0.0f32; n * dim];

        for id in 0..n {
            let start = id * dim;

            let src = &values[start..start + dim];
            let dst = &mut normalized[start..start + dim];

            let mut norm_sq = 0.0f32;

            for i in 0..dim {
                let x = src[i];
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

        // -------------------------------------------------------------
        // Cosine similarity helper.
        //
        // Embeddings are normalized, so cosine similarity is just dot().
        // -------------------------------------------------------------

        #[inline]
        fn similarity(
            normalized: &[f32],
            dim: usize,
            a: usize,
            b: usize,
        ) -> f32 {
            let a_start = a * dim;
            let b_start = b * dim;

            let a_vec = &normalized[
                a_start..a_start + dim
                ];

            let b_vec = &normalized[
                b_start..b_start + dim
                ];

            let mut dot = 0.0f32;

            for k in 0..dim {
                dot += a_vec[k] * b_vec[k];
            }

            dot
        }

        // -------------------------------------------------------------
        // Find the globally closest pair.
        // -------------------------------------------------------------

        let mut max_sim = -1.0f32;
        let mut max_pair = (0usize, 0usize);

        for i in 0..n {
            for j in (i + 1)..n {
                let sim =
                    similarity(
                        &normalized,
                        dim,
                        i,
                        j,
                    );

                if sim > max_sim {
                    max_sim = sim;
                    max_pair = (i, j);
                }
            }
        }

        println!(
            "Highest similarity: {:.5} between {} ({}) and {} ({})",
            max_sim,
            max_pair.0,
            vocab[max_pair.0],
            max_pair.1,
            vocab[max_pair.1],
        );

        // -------------------------------------------------------------
        // Find top-K neighbours for every token.
        //
        // Instead of keeping every pair above the threshold, we only
        // keep the strongest local relationships.
        // -------------------------------------------------------------

        let k = K.min(n - 1);

        // neighbours[id] = [(other_id, similarity), ...]
        //
        // Sorted strongest -> weakest.
        let mut neighbours =
            vec![Vec::<(usize, f32)>::new(); n];

        for i in 0..n {
            let mut best =
                Vec::<(usize, f32)>::with_capacity(k + 1);

            for j in 0..n {
                if i == j {
                    continue;
                }

                let sim =
                    similarity(
                        &normalized,
                        dim,
                        i,
                        j,
                    );

                // Insert into the small sorted top-K list.
                //
                // Since K is only 8, this is cheaper and simpler than
                // introducing a heap.
                let pos = best
                    .binary_search_by(
                        |&(_, existing)| {
                            existing
                                .partial_cmp(&sim)
                                .unwrap_or(
                                    std::cmp::Ordering::Equal
                                )
                                .reverse()
                        }
                    )
                    .unwrap_or_else(|p| p);

                best.insert(pos, (j, sim));

                if best.len() > k {
                    best.pop();
                }
            }

            neighbours[i] = best;
        }

        // -------------------------------------------------------------
        // Mutual-KNN graph.
        //
        // An edge exists only when:
        //
        //   1. similarity >= threshold
        //   2. A considers B one of its top-K neighbours
        //   3. B considers A one of its top-K neighbours
        //
        // This kills most of the giant transitive chains from the
        // previous implementation.
        // -------------------------------------------------------------

        let mut edges =
            vec![Vec::<(usize, f32)>::new(); n];

        for i in 0..n {
            for &(j, sim) in &neighbours[i] {
                if sim < threshold {
                    continue;
                }

                let mutual = neighbours[j]
                    .iter()
                    .any(|&(other, _)| other == i);

                if mutual {
                    edges[i].push((j, sim));
                }
            }
        }

        // -------------------------------------------------------------
        // Union-find over the mutual-KNN graph.
        // -------------------------------------------------------------

        let mut parent: Vec<usize> =
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

            if size[root_a] < size[root_b] {
                std::mem::swap(
                    &mut root_a,
                    &mut root_b,
                );
            }

            parent[root_b] = root_a;
            size[root_a] += size[root_b];
        }

        for i in 0..n {
            for &(j, _) in &edges[i] {
                union(
                    &mut parent,
                    &mut size,
                    i,
                    j,
                );
            }
        }

        // -------------------------------------------------------------
        // Build clusters.
        // -------------------------------------------------------------

        let mut cluster_members =
            vec![Vec::<usize>::new(); n];

        for id in 0..n {
            let root =
                find(&mut parent, id);

            cluster_members[root].push(id);
        }

        let mut clusters =
            Vec::<Vec<usize>>::new();

        for members in cluster_members {
            if members.len() >= 2 {
                clusters.push(members);
            }
        }

        // Largest clusters first.
        clusters.sort_by_key(
            |cluster| std::cmp::Reverse(cluster.len())
        );

        // -------------------------------------------------------------
        // Print clusters.
        // -------------------------------------------------------------

        let mut cluster_number = 0usize;

        for cluster in &clusters {
            cluster_number += 1;

            // Find the cluster's "core" token:
            // the member having the most strong mutual-KNN edges.
            //
            // If tied, use average similarity as the tiebreaker.
            let mut core = cluster[0];
            let mut best_degree = 0usize;
            let mut best_avg = -1.0f32;

            for &id in cluster {
                let mut degree = 0usize;
                let mut sum = 0.0f32;

                for &(other, sim) in &edges[id] {
                    if cluster.contains(&other) {
                        degree += 1;
                        sum += sim;
                    }
                }

                let avg =
                    if degree > 0 {
                        sum / degree as f32
                    } else {
                        0.0
                    };

                if degree > best_degree
                    || (
                    degree == best_degree
                        && avg > best_avg
                )
                {
                    core = id;
                    best_degree = degree;
                    best_avg = avg;
                }
            }

            println!(
                "\nCluster {} ({} tokens)",
                cluster_number,
                cluster.len()
            );

            println!(
                "  Core: {} ({})",
                core,
                vocab[core]
            );

            println!(
                "  Core connections: {}",
                best_degree
            );

            if best_degree > 0 {
                println!(
                    "  Core average similarity: {:.4}",
                    best_avg
                );
            }

            // ---------------------------------------------------------
            // Sort members by similarity to the core.
            // This makes the printed cluster much more useful than
            // simply printing vocabulary-ID order.
            // ---------------------------------------------------------

            let mut ranked =
                cluster
                    .iter()
                    .copied()
                    .filter(|&id| id != core)
                    .map(|id| {
                        let sim =
                            similarity(
                                &normalized,
                                dim,
                                core,
                                id,
                            );

                        (id, sim)
                    })
                    .collect::<Vec<_>>();

            ranked.sort_by(
                |a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(
                            std::cmp::Ordering::Equal
                        )
                }
            );

            let printed =
                ranked
                    .len()
                    .min(MAX_PRINTED_MEMBERS);

            for &(id, sim) in
                ranked.iter().take(printed)
            {
                println!(
                    "    {:.4}  {} ({})",
                    sim,
                    id,
                    vocab[id]
                );
            }

            if ranked.len() > printed {
                println!(
                    "    ... {} more tokens",
                    ranked.len() - printed
                );
            }
        }

        // -------------------------------------------------------------
        // Summary.
        // -------------------------------------------------------------

        let mutual_edges: usize =
            edges
                .iter()
                .map(|e| e.len())
                .sum::<usize>()
                / 2;

        println!(
            "\nClusters: {}",
            clusters.len()
        );

        println!(
            "Mutual-KNN edges: {}",
            mutual_edges
        );

        println!(
            "Threshold: {:.3}",
            threshold
        );

        println!(
            "K: {}",
            k
        );
    }
}
