use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::fs::File;
use std::io::{BufWriter, Write};
pub use crate::vocabs::*;
pub use crate::model_configs::*;
pub use crate::datasets::*;

pub fn dump_vocab_as_rust(
    vocab: &[String],
    path: &str,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "vec![")?;

    for token in vocab {
        // {:?} produces a valid escaped Rust string literal.
        writeln!(w, "    {:?},", token)?;
    }

    writeln!(w, "]")?;

    Ok(())
}

const NONE: u32 = u32::MAX;
const REMOVED: u32 = u32::MAX - 1;

// ------------------------------------------------------------
// Fast integer hasher
//
// The input here is trusted/local data, so we don't need
// HashMap's DoS-resistant SipHash. This is substantially faster
// for integer-heavy tables.
//
// ------------------------------------------------------------

#[derive(Default)]
struct FastHasher(u64);

impl FastHasher {
    #[inline]
    fn mix(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Fallback for things that don't use write_uXX().
        let mut h = 0x9e3779b97f4a7c15u64;

        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }

        self.0 = Self::mix(h);
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.0 = Self::mix(value as u64);
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = Self::mix(value);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.0 = Self::mix(value as u64);
    }
}

type FastHashMap<K, V> =
HashMap<K, V, BuildHasherDefault<FastHasher>>;

type FastHashSet<T> =
HashSet<T, BuildHasherDefault<FastHasher>>;

// ------------------------------------------------------------
// Byte-aware trie tokenizer
//
// Vocabulary entries can be:
//   - normal UTF-8 strings: "hello", "é", " world"
//   - single byte fallback: "<0xE9>"
//   - byte-encoded learned tokens:
//         "<0xE2><0x82>"
//         "<0xC3><0xA9>"
//     (used when the merged byte sequence is not itself valid
//      UTF-8 yet)
//
// Special tokens such as "<EOT>" remain literal strings.
//
// The trie therefore always matches the decoded byte sequence,
// never the literal "<0xXX>" representation.
// ------------------------------------------------------------

#[inline]
fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Parse one exact `<0xXX>` token.
#[inline]
fn parse_byte_token(token: &str) -> Option<u8> {
    let b = token.as_bytes();

    if b.len() != 6 ||
        b[0] != b'<' ||
        b[1] != b'0' ||
        b[2] != b'x' ||
        b[5] != b'>'
    {
        return None;
    }

    let hi = hex_value(b[3])?;
    let lo = hex_value(b[4])?;

    Some((hi << 4) | lo)
}

/// Returns true if the entire token is made only from
/// `<0xXX>` byte markers.
///
/// This is how we represent learned byte sequences that are
/// not valid UTF-8 yet.
#[inline]
fn is_byte_encoded_token(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }

    let bytes = token.as_bytes();
    let mut pos = 0;

    while pos < bytes.len() {
        if pos + 6 > bytes.len() {
            return false;
        }

        if parse_byte_token(
            unsafe { std::str::from_utf8_unchecked(&bytes[pos..pos + 6]) }
        ).is_none() {
            return false;
        }

        pos += 6;
    }

    true
}

/// Append the decoded byte representation of a vocabulary token.
#[inline]
fn append_token_bytes(token: &str, out: &mut Vec<u8>) {
    if is_byte_encoded_token(token) {
        let bytes = token.as_bytes();
        let mut pos = 0;

        while pos < bytes.len() {
            let byte =
                parse_byte_token(
                    unsafe {
                        std::str::from_utf8_unchecked(
                            &bytes[pos..pos + 6]
                        )
                    }
                )
                    .expect("validated byte token");

            out.push(byte);
            pos += 6;
        }
    } else {
        out.extend_from_slice(token.as_bytes());
    }
}

#[inline]
fn decoded_token_len(token: &str) -> usize {
    if is_byte_encoded_token(token) {
        token.len() / 6
    } else if let Some(_) = parse_byte_token(token) {
        1
    } else {
        token.len()
    }
}

pub struct TrieNode {
    children: FastHashMap<u8, u32>,
    id: u32,
}

pub(crate) struct Trie {
    pub(crate) nodes: Vec<TrieNode>,
    byte_fallback_base: u32,
}

impl Trie {
    pub(crate) fn from_vocab(vocab: &[String]) -> Self {
        let total_bytes: usize =
            vocab
                .iter()
                .map(|token| decoded_token_len(token))
                .sum();

        let mut nodes =
            Vec::with_capacity(total_bytes + 1);

        nodes.push(TrieNode {
            children: FastHashMap::default(),
            id: NONE,
        });

        for (id, token) in vocab.iter().enumerate() {
            let id = id as u32;

            let mut current = 0u32;

            if is_byte_encoded_token(token) {
                let bytes = token.as_bytes();
                let mut pos = 0;

                while pos < bytes.len() {
                    let byte =
                        parse_byte_token(
                            unsafe {
                                std::str::from_utf8_unchecked(
                                    &bytes[pos..pos + 6]
                                )
                            }
                        )
                            .expect("validated byte-encoded token");

                    let current_idx = current as usize;

                    if let Some(&next) =
                        nodes[current_idx]
                            .children
                            .get(&byte)
                    {
                        current = next;
                    } else {
                        let next =
                            nodes.len() as u32;

                        nodes.push(TrieNode {
                            children:
                            FastHashMap::default(),
                            id: NONE,
                        });

                        nodes[current_idx]
                            .children
                            .insert(byte, next);

                        current = next;
                    }

                    pos += 6;
                }
            } else if let Some(byte) = parse_byte_token(token) {
                let current_idx =
                    current as usize;

                if let Some(&next) =
                    nodes[current_idx]
                        .children
                        .get(&byte)
                {
                    current = next;
                } else {
                    let next =
                        nodes.len() as u32;

                    nodes.push(TrieNode {
                        children:
                        FastHashMap::default(),
                        id: NONE,
                    });

                    nodes[current_idx]
                        .children
                        .insert(byte, next);

                    current = next;
                }
            } else {
                for &byte in token.as_bytes() {
                    let current_idx =
                        current as usize;

                    if let Some(&next) =
                        nodes[current_idx]
                            .children
                            .get(&byte)
                    {
                        current = next;
                    } else {
                        let next =
                            nodes.len() as u32;

                        nodes.push(TrieNode {
                            children:
                            FastHashMap::default(),
                            id: NONE,
                        });

                        nodes[current_idx]
                            .children
                            .insert(byte, next);

                        current = next;
                    }
                }
            }

            // Empty reserved tokens are not actual trie tokens.
            if !token.is_empty() {
                nodes[current as usize].id = id;
            }
        }

        let byte_fallback_base = vocab
            .iter()
            .position(|token| token == "<0x00>")
            .expect("Vocabulary is missing <0x00>") as u32;

        assert!(
            byte_fallback_base as usize + 256 <= vocab.len(),
            "Byte fallback block is incomplete"
        );

        #[cfg(debug_assertions)]
        for byte in 0u16..=255 {
            let id =
                byte_fallback_base as usize +
                    byte as usize;

            assert_eq!(
                vocab[id],
                format!("<0x{:02X}>", byte),
            );
        }

        Self {
            nodes,
            byte_fallback_base,
        }
    }

    #[inline]
    pub(crate) fn tokenize_u32(
        &self,
        text: &str,
    ) -> Vec<u32> {
        self.tokenize_bytes(text.as_bytes())
    }

    #[inline]
    pub(crate) fn tokenize_bytes(
        &self,
        bytes: &[u8],
    ) -> Vec<u32> {
        let mut result =
            Vec::with_capacity(bytes.len());

        let mut pos = 0usize;

        while pos < bytes.len() {
            let mut current = 0u32;
            let mut best_id = NONE;
            let mut best_len = 0usize;

            let mut j = pos;

            while j < bytes.len() {
                let byte = bytes[j];

                let Some(&next) =
                    self.nodes[current as usize]
                        .children
                        .get(&byte)
                else {
                    break;
                };

                current = next;
                j += 1;

                let id =
                    self.nodes[current as usize].id;

                if id != NONE {
                    best_id = id;
                    best_len = j - pos;
                }
            }

            debug_assert!(
                best_id != NONE,
                "Byte fallback vocabulary should match every input byte"
            );

            if best_id == NONE {
                result.push(
                    self.byte_fallback_base +
                        bytes[pos] as u32,
                );

                pos += 1;
            } else {
                result.push(best_id);
                pos += best_len;
            }
        }

        result
    }
}

pub(crate) struct IncrementalTokenizer<'a> {
    trie: &'a Trie,

    // Tokens whose boundaries can no longer be affected by
    // future bytes.
    pub(crate) stable_tokens: Vec<u32>,

    // Raw bytes at the end of the stream whose greedy tokenization
    // might still change when more bytes arrive.
    pub(crate) pending: Vec<u8>,

    // Number of stable tokens we actually need to retain.
    max_tokens: usize,
}

impl<'a> IncrementalTokenizer<'a> {
    pub(crate) fn new(
        trie: &'a Trie,
        max_tokens: usize,
    ) -> Self {
        Self {
            trie,
            stable_tokens: Vec::with_capacity(max_tokens),
            pending: Vec::new(),
            max_tokens,
        }
    }

    /// Feed one vocabulary token into the stream.
    ///
    /// The token is decoded into its actual byte representation first,
    /// so byte-encoded vocabulary tokens behave exactly like normal
    /// input bytes.
    pub(crate) fn push_token(
        &mut self,
        token: &str,
    ) {
        append_token_bytes(token, &mut self.pending);
        self.resolve_stable();
    }

    /// Move everything that is provably finalized from `pending`
    /// into `stable_tokens`.
    ///
    /// The final trie path is intentionally kept pending when it can
    /// still be extended by future bytes.
    fn resolve_stable(&mut self) {
        let bytes = &self.pending;
        let mut pos = 0usize;

        while pos < bytes.len() {
            let mut current = 0u32;
            let mut best_id = NONE;
            let mut best_len = 0usize;
            let mut j = pos;

            while j < bytes.len() {
                let byte = bytes[j];

                let Some(&next) =
                    self.trie.nodes[current as usize]
                        .children
                        .get(&byte)
                else {
                    break;
                };

                current = next;
                j += 1;

                let id =
                    self.trie.nodes[current as usize].id;

                if id != NONE {
                    best_id = id;
                    best_len = j - pos;
                }
            }

            // If the trie path reached the end of the current
            // stream and can still be extended, we cannot finalize
            // this suffix yet.
            if j == bytes.len()
                && !self.trie.nodes[current as usize]
                .children
                .is_empty()
            {
                break;
            }

            debug_assert!(
                best_id != NONE,
                "Byte fallback vocabulary should match every input byte"
            );

            if best_id == NONE {
                self.stable_tokens.push(
                    self.trie.byte_fallback_base +
                        bytes[pos] as u32,
                );

                pos += 1;
            } else {
                self.stable_tokens.push(best_id);
                pos += best_len;
            }
        }

        if pos > 0 {
            self.pending = self.pending[pos..].to_vec();
        }

        // We only care about the recent context.
        if self.stable_tokens.len() > self.max_tokens {
            let remove =
                self.stable_tokens.len() -
                    self.max_tokens;

            self.stable_tokens.drain(..remove);
        }
    }

    /// Return the exact tokenization of the stream *as it stands now*,
    /// treating the current end of the stream as EOF.
    ///
    /// This may temporarily finalize `pending`. We deliberately do
    /// not mutate the incremental state because a future generated
    /// token may cause that suffix to merge differently.
    pub(crate) fn current_ids(
        &self,
        context_len: usize,
        pad_id: usize,
    ) -> Vec<usize> {
        let pending_ids =
            self.trie.tokenize_bytes(&self.pending); // here

        let mut ids =
            Vec::with_capacity(
                self.stable_tokens.len() +
                    pending_ids.len(),
            );

        ids.extend(
            self.stable_tokens
                .iter()
                .map(|&x| x as usize)
        );

        ids.extend(
            pending_ids
                .into_iter()
                .map(|x| x as usize)
        );

        if ids.len() > context_len {
            ids =
                ids[ids.len() - context_len..]
                    .to_vec();
        } else if ids.len() < context_len {
            let mut padded =
                vec![pad_id;
                     context_len - ids.len()];

            padded.extend(ids);
            ids = padded;
        }

        ids
    }

    pub(crate) fn push_raw_bytes(
        &mut self,
        bytes: &[u8],
    ) {
        self.pending.extend_from_slice(bytes);
        self.resolve_stable();
    }
}

// ------------------------------------------------------------
// Token metadata
// ------------------------------------------------------------

#[derive(Clone, Copy)]
struct TokenMeta {
    byte_len: u32,
    word_count: u32,
    starts_non_ws: bool,
    ends_non_ws: bool,

    // Special/reserved tokens are barriers and must not be merged.
    mergeable: bool,
}

impl TokenMeta {
    fn from_token(
        token: &str,
        mergeable: bool,
    ) -> Self {
        // Normal UTF-8 token.
        if !is_byte_encoded_token(token) {
            let mut word_count = 0u32;
            let mut in_word = false;
            let mut starts_non_ws = false;
            let mut ends_non_ws = false;
            let mut first = true;

            for ch in token.chars() {
                if ch.is_whitespace() {
                    in_word = false;
                    ends_non_ws = false;
                } else {
                    if first {
                        starts_non_ws = true;
                    }

                    if !in_word {
                        word_count += 1;
                        in_word = true;
                    }

                    ends_non_ws = true;
                }

                first = false;
            }

            return Self {
                byte_len: token.len() as u32,
                word_count,
                starts_non_ws,
                ends_non_ws,
                mergeable,
            };
        }

        // Byte-encoded token.
        let bytes = token.as_bytes();

        let mut word_count = 0u32;
        let mut in_word = false;
        let mut starts_non_ws = false;
        let mut ends_non_ws = false;
        let mut first = true;

        let mut pos = 0usize;

        while pos < bytes.len() {
            let byte =
                parse_byte_token(
                    unsafe {
                        std::str::from_utf8_unchecked(
                            &bytes[pos..pos + 6]
                        )
                    }
                )
                    .expect("validated byte token");

            if byte.is_ascii_whitespace() {
                in_word = false;
                ends_non_ws = false;
            } else {
                if first {
                    starts_non_ws = true;
                }

                if !in_word {
                    word_count += 1;
                    in_word = true;
                }

                ends_non_ws = true;
            }

            first = false;
            pos += 6;
        }

        Self {
            byte_len: (bytes.len() / 6) as u32,
            word_count,
            starts_non_ws,
            ends_non_ws,
            mergeable,
        }
    }

    #[inline]
    fn fuse(a: Self, b: Self) -> Self {
        let merges_words =
            a.ends_non_ws &&
                b.starts_non_ws &&
                a.word_count > 0 &&
                b.word_count > 0;

        Self {
            byte_len:
            a.byte_len + b.byte_len,

            word_count:
            a.word_count +
                b.word_count -
                merges_words as u32,

            starts_non_ws:
            if a.byte_len != 0 {
                a.starts_non_ws
            } else {
                b.starts_non_ws
            },

            ends_non_ws:
            if b.byte_len != 0 {
                b.ends_non_ws
            } else {
                a.ends_non_ws
            },

            mergeable:
            a.mergeable &&
                b.mergeable,
        }
    }

    #[inline]
    fn valid_fusion(
        a: Self,
        b: Self,
    ) -> bool {
        a.mergeable &&
            b.mergeable &&
            Self::fuse(a, b).word_count <= 1
    }
}

// ------------------------------------------------------------
// Pair helpers
// ------------------------------------------------------------

#[inline]
fn pair_key(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

#[inline]
fn pair_ids(key: u64) -> (u32, u32) {
    (
        (key >> 32) as u32,
        key as u32,
    )
}

#[inline]
fn pair_score(
    key: u64,
    count: u32,
    meta: &[TokenMeta],
) -> u64 {
    let (a, b) = pair_ids(key);

    let len =
        meta[a as usize].byte_len +
            meta[b as usize].byte_len;

    // score = count * sqrt(len)
    //
    // sqrt is monotonic, but we still want length to matter.
    // Keeping the calculation in f64 is fine here because this
    // happens only when heap entries are refreshed, not once per
    // token occurrence.
    ((count as f64) * (len as f64).sqrt()) as u64
}

// ------------------------------------------------------------
// Pair index
//
// counts:
//     current number of occurrences of each pair
//
// occurrences:
//     node IDs that have represented that pair
//
// heap:
//     candidates for the highest-scoring pair
//
// The occurrence vectors deliberately allow stale entries.
// When a pair is selected, they are validated against the
// current linked sequence. This avoids expensive deletion from
// occurrence sets on every local change.
// ------------------------------------------------------------

struct PairIndex {
    counts: FastHashMap<u64, u32>,

    occurrences:
        FastHashMap<u64, Vec<u32>>,

    heap:
        BinaryHeap<(u64, u64, u32)>,

    dirty:
        FastHashSet<u64>,
}

impl PairIndex {
    fn build(
        tokens: &[u32],
        meta: &[TokenMeta],
    ) -> Self {
        let mut counts =
            FastHashMap::default();

        let mut occurrences =
            FastHashMap::default();

        if tokens.len() >= 2 {
            for i in 0..tokens.len() - 1 {
                let a = tokens[i] as usize;
                let b = tokens[i + 1] as usize;

                let ma = meta[a];
                let mb = meta[b];

                if !TokenMeta::valid_fusion(ma, mb) {
                    continue;
                }

                let key =
                    pair_key(
                        tokens[i],
                        tokens[i + 1],
                    );

                *counts
                    .entry(key)
                    .or_insert(0) += 1;

                occurrences
                    .entry(key)
                    .or_insert_with(Vec::new)
                    .push(i as u32);
            }
        }

        let mut index = Self {
            counts,
            occurrences,
            heap: BinaryHeap::new(),
            dirty: FastHashSet::default(),
        };

        for (&key, &count) in index.counts.iter() {
            if count >= 2 {
                index.heap.push((
                    pair_score(key, count, meta),
                    key,
                    count,
                ));
            }
        }

        index
    }

    #[inline]
    fn add_edge(
        &mut self,
        node: u32,
        a: u32,
        b: u32,
        meta: &[TokenMeta],
    ) {
        let ma = meta[a as usize];
        let mb = meta[b as usize];

        if !TokenMeta::valid_fusion(ma, mb) {
            return;
        }

        let key = pair_key(a, b);

        *self.counts.entry(key).or_insert(0) += 1;

        self.occurrences
            .entry(key)
            .or_insert_with(Vec::new)
            .push(node);

        self.dirty.insert(key);
    }

    #[inline]
    fn remove_edge(
        &mut self,
        a: u32,
        b: u32,
        meta: &[TokenMeta],
    ) {
        let ma = meta[a as usize];
        let mb = meta[b as usize];

        if !TokenMeta::valid_fusion(ma, mb) {
            return;
        }

        let key = pair_key(a, b);

        let Some(count) = self.counts.get_mut(&key)
        else {
            return;
        };

        *count -= 1;

        if *count == 0 {
            self.counts.remove(&key);

            // The old occurrence vector is now useless.
            self.occurrences.remove(&key);
        }

        self.dirty.insert(key);
    }

    fn flush_heap(&mut self, meta: &[TokenMeta]) {
        for key in self.dirty.drain() {
            if let Some(&count) = self.counts.get(&key) {
                if count >= 2 {
                    self.heap.push((
                        pair_score(key, count, meta),
                        key,
                        count,
                    ));
                }
            }
        }
    }

    fn best_pair(
        &mut self,
        _meta: &[TokenMeta],
    ) -> Option<(u64, u32)> {
        while let Some((
                           _score,
                           key,
                           snapshot_count,
                       )) = self.heap.pop()
        {
            match self.counts.get(&key) {
                Some(&current_count)
                if current_count == snapshot_count &&
                    current_count >= 2 =>
                    {
                        return Some((key, current_count));
                    }

                _ => {
                    // Stale heap entry.
                }
            }
        }

        None
    }
}

// ------------------------------------------------------------
// Incremental pair merging
//
// Sequence is represented as a doubly linked list using u32s.
// We never physically remove elements from the Vec.
//
// This is dramatically cheaper than rebuilding the whole
// token vector after every vocabulary addition.
// ------------------------------------------------------------

fn merge_pair(
    key: u64,
    new_id: u32,

    tokens: &mut [u32],
    prev: &mut [u32],
    next: &mut [u32],

    pair_index: &mut PairIndex,
    meta: &[TokenMeta],
) -> usize {
    // Take the candidate occurrences out of the index.
    //
    // They may contain stale entries, so they are validated below.
    let mut candidates =
        pair_index
            .occurrences
            .remove(&key)
            .unwrap_or_default();

    // Node IDs preserve original text order, because merges only
    // remove the right-hand node of a pair.
    candidates.sort_unstable();

    let (wanted_a, wanted_b) =
        pair_ids(key);

    let mut merged = 0usize;

    for cur_u32 in candidates {
        let cur = cur_u32 as usize;

        // Node was already removed by an overlapping merge.
        if next[cur] == REMOVED {
            continue;
        }

        let right_u32 = next[cur];

        if right_u32 == NONE {
            continue;
        }

        let right = right_u32 as usize;

        // It may no longer be the same pair because a previous
        // merge modified this region.
        if tokens[cur] != wanted_a ||
            tokens[right] != wanted_b
        {
            continue;
        }

        let left_u32 = prev[cur];
        let after_u32 = next[right];

        // --------------------------------------------
        // Remove old edges:
        //
        // left -> A
        // A    -> B
        // B    -> after
        // --------------------------------------------

        if left_u32 != NONE {
            let left = left_u32 as usize;

            pair_index.remove_edge(
                tokens[left],
                tokens[cur],
                meta,
            );
        }

        pair_index.remove_edge(
            tokens[cur],
            tokens[right],
            meta,
        );

        if after_u32 != NONE {
            let after = after_u32 as usize;

            pair_index.remove_edge(
                tokens[right],
                tokens[after],
                meta,
            );
        }

        // --------------------------------------------
        // Merge A+B into current node.
        // --------------------------------------------

        tokens[cur] = new_id;
        next[cur] = after_u32;

        if after_u32 != NONE {
            prev[after_u32 as usize] = cur_u32;
        }

        // Mark the right node as dead.
        next[right] = REMOVED;
        prev[right] = REMOVED;

        // --------------------------------------------
        // Add new edges:
        //
        // left -> NEW
        // NEW  -> after
        // --------------------------------------------

        if left_u32 != NONE {
            let left = left_u32 as usize;

            pair_index.add_edge(
                left_u32,
                tokens[left],
                new_id,
                meta,
            );
        }

        if after_u32 != NONE {
            let after = after_u32 as usize;

            pair_index.add_edge(
                cur_u32,
                new_id,
                tokens[after],
                meta,
            );
        }

        merged += 1;
    }

    // Update the heap once after the entire merge.
    pair_index.flush_heap(meta);

    merged
}

// ------------------------------------------------------------
// Fully optimized vocab builder
// ------------------------------------------------------------

fn add_byte_fallback_tokens(vocab: &mut Vec<String>) {
    for byte in 0u16..=255 {
        vocab.push(format!("<0x{:02X}>", byte));
    }
}

pub fn make_vocab(
    text: &str,
    token_num: usize,
    reserved_token_num: usize,
    special_tokens: Option<&[&str]>,
) -> Vec<String> {
    let now = std::time::Instant::now();
    // --------------------------------------------------------
    // Initial vocabulary:
    //   0..=reserved_token_num-1 -> reserved empty IDs
    //   reserved_token_num      -> extra empty slot
    //   special tokens
    //   256 byte fallback tokens
    // --------------------------------------------------------

    let special_count =
        special_tokens.map_or(0, |s| s.len());

    let initial_vocab_size =
        reserved_token_num +
            1 +
            special_count +
            256;

    let mut vocab =
        Vec::with_capacity(
            initial_vocab_size + token_num,
        );

    // Important:
    // reserved_token_num = 1
    // => two empty slots: IDs 0 and 1.
    vocab.resize(
        reserved_token_num + 1,
        String::new(),
    );

    if let Some(special) = special_tokens {
        vocab.extend(
            special.iter().map(|token| (*token).to_string())
        );
    }

    let byte_fallback_base =
        vocab.len();

    add_byte_fallback_tokens(&mut vocab);

    debug_assert_eq!(
        byte_fallback_base + 256,
        vocab.len()
    );

    // `token_num` is the number of NEW learned tokens.
    let target_vocab_size =
        vocab.len() + token_num;

    if vocab.len() > u32::MAX as usize {
        panic!("Vocabulary exceeds u32 token-ID capacity");
    }

    // --------------------------------------------------------
    // Token metadata
    //
    // Reserved slots are not mergeable.
    // Special tokens are not mergeable.
    // Byte fallback tokens are mergeable.
    // --------------------------------------------------------

    let mut meta =
        Vec::with_capacity(target_vocab_size);

    for id in 0..vocab.len() {
        let mergeable =
            id >= byte_fallback_base;

        meta.push(
            TokenMeta::from_token(
                &vocab[id],
                mergeable,
            )
        );
    }

    // --------------------------------------------------------
    // Initial tokenization
    //
    // The base vocabulary already contains every byte,
    // so the trie can tokenize the whole corpus directly.
    //
    // Special tokens automatically win through longest-match.
    // --------------------------------------------------------

    let trie =
        Trie::from_vocab(&vocab);

    let mut tokens =
        trie.tokenize_u32(text);

    if tokens.is_empty() {
        return vocab;
    }

    // --------------------------------------------------------
    // Convert initial tokenization into a linked sequence.
    // --------------------------------------------------------

    let n =
        tokens.len();

    let mut prev =
        vec![NONE; n];

    let mut next =
        vec![NONE; n];

    for i in 0..n {
        if i > 0 {
            prev[i] =
                (i - 1) as u32;
        }

        if i + 1 < n {
            next[i] =
                (i + 1) as u32;
        }
    }

    // --------------------------------------------------------
    // Build initial pair counts + occurrence index.
    // --------------------------------------------------------

    let mut pair_index =
        PairIndex::build(
            &tokens,
            &meta,
        );

    // --------------------------------------------------------
    // Incremental vocabulary construction.
    // --------------------------------------------------------

    while vocab.len() < target_vocab_size {
        let Some((best_key, count)) =
            pair_index.best_pair(&meta)
        else {
            println!(
                "Not enough text to make {} tokens!",
                token_num
            );
            break;
        };

        if count < 2 {
            break;
        }

        let (a, b) =
            pair_ids(best_key);

        // ----------------------------------------------------
        // Construct the new token exactly once.
        // ----------------------------------------------------

        // --------------------------------------------------------
        // Construct the new token from DECODED bytes.
        //
        // This is crucial:
        //
        //   "<0xC3>" + "<0xA9>"
        //
        // must become:
        //
        //   "é"
        //
        // rather than:
        //
        //   "<0xC3><0xA9>"
        //
        // For incomplete/invalid UTF-8 byte sequences, keep the
        // canonical `<0xXX>` representation.
        // --------------------------------------------------------

        let a_token =
            &vocab[a as usize];

        let b_token =
            &vocab[b as usize];

        let decoded_len =
            meta[a as usize].byte_len as usize +
                meta[b as usize].byte_len as usize;

        let mut bytes =
            Vec::with_capacity(decoded_len);

        append_token_bytes(
            a_token,
            &mut bytes,
        );

        append_token_bytes(
            b_token,
            &mut bytes,
        );

        let new_token =
            match String::from_utf8(bytes.clone()) {
                Ok(s) => s,

                Err(_) => {
                    let mut encoded =
                        String::with_capacity(
                            bytes.len() * 6
                        );

                    for byte in bytes {
                        use std::fmt::Write;

                        write!(
                            encoded,
                            "<0x{:02X}>",
                            byte
                        )
                            .unwrap();
                    }

                    encoded
                }
            };

        // ----------------------------------------------------
        // Metadata.
        // ----------------------------------------------------

        let new_meta =
            TokenMeta::fuse(
                meta[a as usize],
                meta[b as usize],
            );

        let new_id =
            vocab.len() as u32;

        vocab.push(new_token);
        meta.push(new_meta);

        // ----------------------------------------------------
        // Merge only actual occurrences.
        // ----------------------------------------------------

        let merged =
            merge_pair(
                best_key,
                new_id,

                &mut tokens,
                &mut prev,
                &mut next,

                &mut pair_index,
                &meta,
            );

        debug_assert!(
            merged > 0,
            "Best pair existed in the count table but had no live occurrences"
        );

        if merged == 0 {
            break;
        }
    }
    println!("Vocabulary creation finished in {:?}.", now.elapsed());

    vocab
}
