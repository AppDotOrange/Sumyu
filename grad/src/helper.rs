use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
pub use crate::datasets::*;
pub use crate::model_configs::*;
pub use crate::vocabs::*;

pub fn dump_vocab_as_rust(
    vocab: &[String],
    path: &str,
) -> std::io::Result<()> {
    let file = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    let mut w = BufWriter::new(file);

    writeln!(w, "vec![")?;

    for token in vocab {
        writeln!(w, "    {:?},", token)?;
    }

    writeln!(w, "]")?;

    Ok(())
}

const NONE: u32 = u32::MAX;
const REMOVED: u32 = u32::MAX - 1;


// ------------------------------------------------------------
// Byte-aware trie tokenizer
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

#[inline]
fn parse_byte_token(token: &str) -> Option<u8> {
    let b = token.as_bytes();

    if b.len() != 6
        || b[0] != b'<'
        || b[1] != b'0'
        || b[2] != b'x'
        || b[5] != b'>'
    {
        return None;
    }

    let hi = hex_value(b[3])?;
    let lo = hex_value(b[4])?;

    Some((hi << 4) | lo)
}

#[inline]
fn is_byte_encoded_token(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }

    let bytes = token.as_bytes();
    let mut pos = 0usize;

    while pos < bytes.len() {
        if pos + 6 > bytes.len() {
            return false;
        }

        if parse_byte_token(
            unsafe {
                std::str::from_utf8_unchecked(
                    &bytes[pos..pos + 6],
                )
            },
        )
            .is_none()
        {
            return false;
        }

        pos += 6;
    }

    true
}

type FastHashMap<K, V> =
HashMap<K, V, BuildHasherDefault<FastHasher>>;

type FastHashSet<T> =
HashSet<T, BuildHasherDefault<FastHasher>>;

trait FastHashMapExt<K, V> {
    fn with_capacity(capacity: usize) -> Self;
}

impl<K, V> FastHashMapExt<K, V> for FastHashMap<K, V> {
    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        HashMap::with_capacity_and_hasher(
            capacity,
            BuildHasherDefault::<FastHasher>::default(),
        )
    }
}

#[inline]
fn append_token_bytes(
    token: &str,
    out: &mut Vec<u8>,
) {
    if is_byte_encoded_token(token) {
        let bytes = token.as_bytes();
        let mut pos = 0usize;

        while pos < bytes.len() {
            let byte =
                parse_byte_token(
                    unsafe {
                        std::str::from_utf8_unchecked(
                            &bytes[pos..pos + 6],
                        )
                    },
                )
                    .expect("validated byte token");

            out.push(byte);
            pos += 6;
        }
    } else {
        out.extend_from_slice(
            token.as_bytes(),
        );
    }
}

#[inline]
fn decoded_token_len(token: &str) -> usize {
    if is_byte_encoded_token(token) {
        token.len() / 6
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
    max_token_bytes: usize,
}

impl Trie {
    pub(crate) fn from_vocab(
        vocab: &[String],
        byte_fallback_base: u32,
    ) -> Self {
        let total_bytes: usize =
            vocab
                .iter()
                .map(|token| decoded_token_len(token))
                .sum();

        let mut max_token_bytes = 1usize;

        let mut nodes =
            Vec::with_capacity(
                total_bytes + 1,
            );

        nodes.push(TrieNode {
            children:
            FastHashMap::default(),
            id: NONE,
        });

        for (id, token) in vocab.iter().enumerate() {
            let id = id as u32;

            let decoded_len =
                decoded_token_len(token);

            max_token_bytes =
                max_token_bytes.max(decoded_len);

            let mut current = 0u32;

            if is_byte_encoded_token(token) {
                let bytes = token.as_bytes();
                let mut pos = 0usize;

                while pos < bytes.len() {
                    let byte =
                        parse_byte_token(
                            unsafe {
                                std::str::from_utf8_unchecked(
                                    &bytes[pos..pos + 6],
                                )
                            },
                        )
                            .expect(
                                "validated byte-encoded token",
                            );

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

                        nodes.push(
                            TrieNode {
                                children:
                                FastHashMap::default(),
                                id: NONE,
                            },
                        );

                        nodes[current_idx]
                            .children
                            .insert(
                                byte,
                                next,
                            );

                        current = next;
                    }

                    pos += 6;
                }
            } else if let Some(byte) =
                parse_byte_token(token)
            {
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

                    nodes.push(
                        TrieNode {
                            children:
                            FastHashMap::default(),
                            id: NONE,
                        },
                    );

                    nodes[current_idx]
                        .children
                        .insert(
                            byte,
                            next,
                        );

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

                        nodes.push(
                            TrieNode {
                                children:
                                FastHashMap::default(),
                                id: NONE,
                            },
                        );

                        nodes[current_idx]
                            .children
                            .insert(
                                byte,
                                next,
                            );

                        current = next;
                    }
                }
            }

            if !token.is_empty() {
                nodes[current as usize].id =
                    id;
            }
        }

        assert!(
            byte_fallback_base as usize + 256
                <= vocab.len(),
            "Byte fallback block is incomplete",
        );

        #[cfg(debug_assertions)]
        for byte in 0u16..=255 {
            let id =
                byte_fallback_base as usize
                    + byte as usize;

            assert_eq!(
                vocab[id],
                format!(
                    "<0x{:02X}>",
                    byte,
                ),
            );
        }

        Self {
            nodes,
            byte_fallback_base,
            max_token_bytes,
        }
    }

    #[inline]
    pub(crate) fn tokenize_bytes_u16(
        &self,
        bytes: &[u8],
    ) -> Vec<u16> {
        let mut result =
            Vec::with_capacity(
                bytes.len().saturating_div(2),
            );

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

            if best_id == NONE {
                result.push(
                    (
                        self.byte_fallback_base
                            + bytes[pos] as u32
                    ) as u16,
                );

                pos += 1;
            } else {
                result.push(
                    best_id as u16,
                );

                pos += best_len;
            }
        }

        result
    }

    #[inline]
    pub(crate) fn tokenize_u32(
        &self,
        text: &str,
    ) -> Vec<u32> {
        self.tokenize_bytes(
            text.as_bytes(),
        )
    }

    #[inline]
    pub(crate) fn tokenize_bytes(
        &self,
        bytes: &[u8],
    ) -> Vec<u32> {
        let mut result =
            Vec::with_capacity(
                bytes.len().saturating_div(2),
            );

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
                    best_len =
                        j - pos;
                }
            }

            if best_id == NONE {
                result.push(
                    self.byte_fallback_base
                        + bytes[pos] as u32,
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

#[derive(Default)]
struct OccurrenceStore {
    // key -> occurrence positions
    //
    // Each u32 is the LEFT node of the pair:
    //
    //     tokens[pos] == A
    //     next[pos]  == right
    //
    // The vector is append-ordered, just like the old
    // occurrence linked list was.
    lists: FastHashMap<u32, Vec<u32>>,

    total_len: usize,
}

impl OccurrenceStore {
    fn from_pairs(
        pairs: &PairTable,
    ) -> Self {
        let mut lists =
            FastHashMap::with_capacity(
                pairs.len().saturating_mul(2),
            );

        let mut total_len =
            0usize;

        /*
         * IMPORTANT:
         *
         * We store EVERY pair occurrence here,
         * including singleton pairs.
         *
         * A pair with initial count=1 can later become
         * count=2 after a merge. We therefore must retain
         * its original occurrence.
         */
        for pair in pairs.iter() {
            let count =
                pair.count() as usize;

            if count == 0 {
                continue;
            }

            total_len =
                total_len
                    .checked_add(count)
                    .expect(
                        "occurrence count overflow",
                    );

            lists.insert(
                pair.key,
                Vec::with_capacity(
                    count,
                ),
            );
        }

        Self {
            lists,
            total_len,
        }
    }

    #[inline]
    fn len(
        &self,
    ) -> usize {
        self.total_len
    }

    #[inline]
    fn append(
        &mut self,
        key: u32,
        node: u32,
    ) -> std::io::Result<()> {
        self.lists
            .entry(key)
            .or_default()
            .push(node);

        self.total_len =
            self.total_len
                .checked_add(1)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "occurrence count overflow",
                    )
                })?;

        Ok(())
    }

    #[inline]
    fn remove(
        &mut self,
        key: u32,
    ) {
        if let Some(list) =
            self.lists.remove(&key)
        {
            self.total_len =
                self.total_len
                    .saturating_sub(
                        list.len(),
                    );
        }
    }

    #[inline]
    fn take(
        &mut self,
        key: u32,
    ) -> Vec<u32> {
        let list =
            self.lists
                .remove(&key)
                .unwrap_or_default();

        self.total_len =
            self.total_len
                .saturating_sub(
                    list.len(),
                );

        list
    }

    /*
     * Remove stale occurrence positions from every pair.
     *
     * This is dramatically cheaper than rebuilding the old
     * disk linked-list because all lists are contiguous RAM.
     *
     * `tokens` and `next` describe the CURRENT linked-list
     * topology.
     */
    fn compact(
        &mut self,
        pairs: &PairTable,
        tokens: &[u16],
        next: &[u32],
    ) {
        let mut total =
            0usize;

        for (
            key,
            list,
        ) in self.lists.iter_mut()
        {
            list.retain(
                |&pos_u32| {
                    let pos =
                        pos_u32 as usize;

                    if pos >= tokens.len() {
                        return false;
                    }

                    let right_u32 =
                        next[pos];

                    if right_u32
                        == NONE
                        || right_u32
                        == REMOVED
                    {
                        return false;
                    }

                    let right =
                        right_u32
                            as usize;

                    if right
                        >= tokens.len()
                    {
                        return false;
                    }

                    pair_key(
                        tokens[pos],
                        tokens[right],
                    ) == *key
                },
            );

            /*
             * Actually give excessive unused capacity back.
             * This only runs during explicit compaction, so the
             * cost is acceptable.
             */
            if list.capacity()
                > list.len()
                .saturating_mul(2)
                .saturating_add(1024)
            {
                list.shrink_to_fit();
            }

            total =
                total
                    .checked_add(
                        list.len(),
                    )
                    .expect(
                        "occurrence total overflow",
                    );
        }

        /*
         * Pair count zero means the pair itself was removed,
         * so its occurrence list should disappear too.
         */
        self.lists.retain(
            |key, list| {
                !list.is_empty()
                    && pairs
                    .get(*key)
                    .is_some()
            },
        );

        /*
         * Recompute because retain() above may have removed
         * empty lists.
         */
        self.total_len =
            self.lists
                .values()
                .map(Vec::len)
                .sum();
    }
}


// ------------------------------------------------------------
// Fast hasher
// ------------------------------------------------------------

#[derive(Default)]
struct FastHasher(u64);

impl FastHasher {
    #[inline]
    fn mix(
        mut x: u64,
    ) -> u64 {
        x ^= x >> 30;

        x = x.wrapping_mul(
            0xbf58476d1ce4e5b9,
        );

        x ^= x >> 27;

        x = x.wrapping_mul(
            0x94d049bb133111eb,
        );

        x ^ (x >> 31)
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn finish(
        &self,
    ) -> u64 {
        self.0
    }

    #[inline]
    fn write(
        &mut self,
        bytes: &[u8],
    ) {
        let mut h =
            0x9e3779b97f4a7c15u64;

        for &b in bytes {
            h ^= b as u64;

            h = h.wrapping_mul(
                0x100000001b3,
            );
        }

        self.0 =
            Self::mix(h);
    }

    #[inline]
    fn write_u32(
        &mut self,
        value: u32,
    ) {
        self.0 =
            Self::mix(
                value as u64
            );
    }

    #[inline]
    fn write_u64(
        &mut self,
        value: u64,
    ) {
        self.0 =
            Self::mix(value);
    }

    #[inline]
    fn write_usize(
        &mut self,
        value: usize,
    ) {
        self.0 =
            Self::mix(
                value as u64
            );
    }
}


// ------------------------------------------------------------
// Incremental tokenizer
// ------------------------------------------------------------

pub(crate) struct IncrementalTokenizer<'a> {
    trie: &'a Trie,

    pub(crate) stable_tokens:
        Vec<u32>,

    pub(crate) pending:
        Vec<u8>,

    max_tokens: usize,
}

impl<'a> IncrementalTokenizer<'a> {
    pub(crate) fn new(
        trie: &'a Trie,
        max_tokens: usize,
    ) -> Self {
        Self {
            trie,
            stable_tokens:
            Vec::with_capacity(
                max_tokens,
            ),
            pending:
            Vec::new(),
            max_tokens,
        }
    }

    pub(crate) fn push_token(
        &mut self,
        token: &str,
    ) {
        append_token_bytes(
            token,
            &mut self.pending,
        );

        self.resolve_stable();
    }

    fn resolve_stable(
        &mut self,
    ) {
        let bytes =
            &self.pending;

        let mut pos = 0usize;

        while pos < bytes.len() {
            let mut current =
                0u32;

            let mut best_id =
                NONE;

            let mut best_len =
                0usize;

            let mut j =
                pos;

            while j < bytes.len() {
                let byte =
                    bytes[j];

                let Some(&next) =
                    self.trie
                        .nodes[
                        current as usize
                        ]
                        .children
                        .get(&byte)
                else {
                    break;
                };

                current = next;
                j += 1;

                let id =
                    self.trie
                        .nodes[
                        current as usize
                        ]
                        .id;

                if id != NONE {
                    best_id =
                        id;

                    best_len =
                        j - pos;
                }
            }

            if j == bytes.len()
                && !self.trie
                .nodes[
                current as usize
                ]
                .children
                .is_empty()
            {
                break;
            }

            debug_assert!(
                best_id != NONE,
                "Byte fallback vocabulary should match every input byte",
            );

            if best_id == NONE {
                self.stable_tokens.push(
                    self.trie
                        .byte_fallback_base
                        + bytes[pos] as u32,
                );

                pos += 1;
            } else {
                self.stable_tokens
                    .push(best_id);

                pos += best_len;
            }
        }

        if pos > 0 {
            let remaining =
                self.pending.len()
                    - pos;

            self.pending.copy_within(
                pos..,
                0,
            );

            self.pending
                .truncate(remaining);
        }

        if self.stable_tokens.len()
            > self.max_tokens
        {
            let remove =
                self.stable_tokens.len()
                    - self.max_tokens;

            self.stable_tokens
                .drain(..remove);
        }
    }

    pub(crate) fn current_ids(
        &self,
        context_len: usize,
        pad_id: usize,
    ) -> Vec<usize> {
        let pending_ids =
            self.trie
                .tokenize_bytes(
                    &self.pending,
                );

        let mut ids =
            Vec::with_capacity(
                self.stable_tokens.len()
                    + pending_ids.len(),
            );

        ids.extend(
            self.stable_tokens
                .iter()
                .map(|&x| x as usize),
        );

        ids.extend(
            pending_ids
                .into_iter()
                .map(|x| x as usize),
        );

        if ids.len()
            > context_len
        {
            ids =
                ids[
                    ids.len()
                        - context_len..
                    ]
                    .to_vec();
        } else if ids.len()
            < context_len
        {
            let mut padded =
                vec![
                    pad_id;
                    context_len
                        - ids.len()
                ];

            padded.extend(ids);

            ids = padded;
        }

        ids
    }

    pub(crate) fn push_raw_bytes(
        &mut self,
        bytes: &[u8],
    ) {
        self.pending
            .extend_from_slice(bytes);

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
    mergeable: bool,
}

impl TokenMeta {
    fn from_token(
        token: &str,
        mergeable: bool,
    ) -> Self {
        if !is_byte_encoded_token(token) {
            let mut word_count =
                0u32;

            let mut in_word =
                false;

            let mut starts_non_ws =
                false;

            let mut ends_non_ws =
                false;

            let mut first =
                true;

            for ch in token.chars() {
                if ch.is_whitespace() {
                    in_word = false;
                    ends_non_ws = false;
                } else {
                    if first {
                        starts_non_ws =
                            true;
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
                byte_len:
                token.len() as u32,
                word_count,
                starts_non_ws,
                ends_non_ws,
                mergeable,
            };
        }

        let bytes =
            token.as_bytes();

        let mut word_count =
            0u32;

        let mut in_word =
            false;

        let mut starts_non_ws =
            false;

        let mut ends_non_ws =
            false;

        let mut first =
            true;

        let mut pos =
            0usize;

        while pos < bytes.len() {
            let byte =
                parse_byte_token(
                    unsafe {
                        std::str::from_utf8_unchecked(
                            &bytes[pos..pos + 6],
                        )
                    },
                )
                    .expect(
                        "validated byte token",
                    );

            if byte.is_ascii_whitespace() {
                in_word = false;
                ends_non_ws = false;
            } else {
                if first {
                    starts_non_ws =
                        true;
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
            byte_len:
            (bytes.len() / 6) as u32,
            word_count,
            starts_non_ws,
            ends_non_ws,
            mergeable,
        }
    }

    #[inline]
    fn fuse(
        a: Self,
        b: Self,
    ) -> Self {
        let merges_words =
            a.ends_non_ws
                && b.starts_non_ws
                && a.word_count > 0
                && b.word_count > 0;

        Self {
            byte_len:
            a.byte_len
                + b.byte_len,

            word_count:
            a.word_count
                + b.word_count
                - merges_words as u32,

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
            a.mergeable
                && b.mergeable,
        }
    }

    #[inline]
    fn valid_fusion(
        a: Self,
        b: Self,
    ) -> bool {
        a.mergeable
            && b.mergeable
            && Self::fuse(a, b)
            .word_count
            <= 1
    }
}


// ------------------------------------------------------------
// Pair helpers
// ------------------------------------------------------------

#[inline]
fn pair_key(
    a: u16,
    b: u16,
) -> u32 {
    ((a as u32) << 16)
        | b as u32
}

#[inline]
fn pair_ids(
    key: u32,
) -> (u16, u16) {
    (
        (key >> 16) as u16,
        key as u16,
    )
}

#[inline]
fn pair_score(
    key: u32,
    count: u32,
    meta: &[TokenMeta],
) -> u64 {
    let (a, b) =
        pair_ids(key);

    let len =
        meta[a as usize]
            .byte_len
            + meta[b as usize]
            .byte_len;

    (
        (count as f64)
            * (len as f64).sqrt()
    ) as u64
}


// ------------------------------------------------------------
// Compact pair table
// ------------------------------------------------------------

const EMPTY_PAIR: u32 =
    u32::MAX;

const COUNT_QUEUED: u32 =
    1 << 31;

const COUNT_MASK: u32 =
    COUNT_QUEUED - 1;

#[derive(Clone, Copy)]
struct PairSlot {
    key: u32,
    count: u32,
}

impl PairSlot {
    #[inline]
    fn count(
        &self,
    ) -> u32 {
        self.count & COUNT_MASK
    }

    #[inline]
    fn set_count(
        &mut self,
        count: u32,
    ) {
        debug_assert!(
            count <= COUNT_MASK
        );

        self.count =
            (self.count
                & COUNT_QUEUED)
                | count;
    }

    #[inline]
    fn queued(
        &self,
    ) -> bool {
        (self.count
            & COUNT_QUEUED)
            != 0
    }

    #[inline]
    fn set_queued(
        &mut self,
        value: bool,
    ) {
        if value {
            self.count |=
                COUNT_QUEUED;
        } else {
            self.count &=
                COUNT_MASK;
        }
    }
}

struct PairTable {
    slots: Vec<PairSlot>,
    len: usize,
}

impl PairTable {

    fn with_capacity(
        capacity: usize,
    ) -> Self {
        let capacity =
            capacity
                .max(16)
                .next_power_of_two();

        Self {
            slots:
            vec![
                PairSlot {
                    key: EMPTY_PAIR,
                    count: 0,
                };
                capacity
            ],
            len: 0,
        }
    }

    #[inline]
    fn hash(
        key: u32,
    ) -> usize {
        FastHasher::mix(
            key as u64,
        ) as usize
    }

    #[inline]
    fn find_slot(
        &self,
        key: u32,
    ) -> usize {
        let mask =
            self.slots.len()
                - 1;

        let mut index =
            Self::hash(key)
                & mask;

        loop {
            let existing =
                self.slots[index]
                    .key;

            if existing
                == EMPTY_PAIR
                || existing == key
            {
                return index;
            }

            index =
                (index + 1)
                    & mask;
        }
    }

    #[inline]
    fn get(
        &self,
        key: u32,
    ) -> Option<&PairSlot> {
        let index =
            self.find_slot(key);

        if self.slots[index]
            .key
            == key
        {
            Some(
                &self.slots[index],
            )
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(
        &mut self,
        key: u32,
    ) -> Option<&mut PairSlot> {
        let index =
            self.find_slot(key);

        if self.slots[index]
            .key
            == key
        {
            Some(
                &mut self.slots[index],
            )
        } else {
            None
        }
    }

    #[inline]
    fn get_or_insert(
        &mut self,
        key: u32,
    ) -> &mut PairSlot {
        let mut index =
            self.find_slot(key);

        // Already present.
        if self.slots[index].key == key {
            return &mut self.slots[index];
        }

        // Need to grow before inserting.
        if (self.len + 1) * 10
            >= self.slots.len() * 7
        {
            self.rehash();

            index =
                self.find_slot(key);
        }

        self.slots[index] =
            PairSlot {
                key,
                count: 0,
            };

        self.len += 1;

        &mut self.slots[index]
    }

    fn remove(
        &mut self,
        key: u32,
    ) {
        let index =
            self.find_slot(key);

        if self.slots[index]
            .key
            != key
        {
            return;
        }

        self.slots[index] =
            PairSlot {
                key: EMPTY_PAIR,
                count: 0,
            };

        self.len -= 1;

        let mask =
            self.slots.len()
                - 1;

        let mut scan =
            (index + 1)
                & mask;

        while self.slots[scan]
            .key
            != EMPTY_PAIR
        {
            let slot =
                self.slots[scan];

            self.slots[scan] =
                PairSlot {
                    key: EMPTY_PAIR,
                    count: 0,
                };

            self.len -= 1;

            let dst =
                self.find_slot(
                    slot.key,
                );

            self.slots[dst] =
                slot;

            self.len += 1;

            scan =
                (scan + 1)
                    & mask;
        }
    }

    fn rehash(
        &mut self,
    ) {
        let old_capacity =
            self.slots.len();

        let old_slots =
            std::mem::replace(
                &mut self.slots,
                vec![
                    PairSlot {
                        key: EMPTY_PAIR,
                        count: 0,
                    };
                    old_capacity * 2
                ],
            );

        self.len = 0;

        for slot in old_slots {
            if slot.key
                == EMPTY_PAIR
            {
                continue;
            }

            let index =
                self.find_slot(
                    slot.key,
                );

            self.slots[index] =
                slot;

            self.len += 1;
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn iter_mut(
        &mut self,
    ) -> impl Iterator<
        Item = &mut PairSlot
    > {
        self.slots
            .iter_mut()
            .filter(
                |slot| {
                    slot.key
                        != EMPTY_PAIR
                },
            )
    }

    #[inline]
    fn iter(
        &self,
    ) -> impl Iterator<Item = &PairSlot> {
        self.slots
            .iter()
            .filter(
                |slot| {
                    slot.key != EMPTY_PAIR
                },
            )
    }
}
// ------------------------------------------------------------
// Pair index
//
// IMPORTANT:
//
// This intentionally models the OLD implementation:
//
//     counts[pair] = live count
//     occurrences[pair] = stale-capable occurrence list
//
// The occurrence lists are simply stored on disk.
// ------------------------------------------------------------

struct PairIndex {
    pairs: PairTable,

    prev: Vec<u32>,
    next: Vec<u32>,

    occ: OccurrenceStore,

    heap:
        BinaryHeap<
            (u64, u32, u32)
        >,

    dirty:
        FastHashSet<u32>,
}

impl PairIndex {
    fn build(
        tokens: &[u16],
        meta: &[TokenMeta],
    ) -> std::io::Result<Self> {
        let n =
            tokens.len();

        assert!(
            n <= u32::MAX as usize,
            "Corpus has too many token positions",
        );

        println!(
            "Building pair index for {} initial tokens...",
            n,
        );

        println!(
            "RAM linked-list memory: {:.2} GiB",
            (n as f64 * 8.0)
                / (
                1024.0
                    * 1024.0
                    * 1024.0
            ),
        );

        // --------------------------------------------------------
        // Linked-list topology
        // --------------------------------------------------------

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
        // Pair counts
        // --------------------------------------------------------

        let mut pairs =
            PairTable::with_capacity(
                1 << 20,
            );

        println!(
            "Pass 1/2: counting pair frequencies..."
        );

        let progress_step =
            64 * 1024 * 1024;

        for i in 1..n {
            let a =
                tokens[i - 1];

            let b =
                tokens[i];

            if !TokenMeta::valid_fusion(
                meta[a as usize],
                meta[b as usize],
            ) {
                continue;
            }

            let key =
                pair_key(a, b);

            let pair =
                pairs
                    .get_or_insert(key);

            let count =
                pair.count()
                    + 1;

            assert!(
                count <= COUNT_MASK,
                "Pair occurrence count overflow",
            );

            pair.set_count(
                count,
            );

            if i % progress_step == 0
                || i + 1 == n
            {
                println!(
                    "Pair counting: {}/{} ({:.1}%)",
                    i,
                    n,
                    i as f64
                        * 100.0
                        / n as f64,
                );
            }
        }

        println!(
            "Distinct pairs: {}",
            pairs.len(),
        );

        // --------------------------------------------------------
        // RAM occurrence lists
        // --------------------------------------------------------

        println!(
            "Building in-RAM occurrence lists..."
        );

        let mut occ =
            OccurrenceStore::from_pairs(
                &pairs,
            );

        /*
         * Because from_pairs() reserved the exact initial
         * frequencies, pass 2 doesn't cause repeated Vec growth.
         */

        for i in 1..n {
            let a =
                tokens[i - 1];

            let b =
                tokens[i];

            if !TokenMeta::valid_fusion(
                meta[a as usize],
                meta[b as usize],
            ) {
                continue;
            }

            let key =
                pair_key(a, b);

            occ.append(
                key,
                (i - 1) as u32,
            )?;

            if i % progress_step == 0
                || i + 1 == n
            {
                println!(
                    "Occurrence building: {}/{} ({:.1}%)",
                    i,
                    n,
                    i as f64
                        * 100.0
                        / n as f64,
                );
            }
        }

        println!(
            "Occurrence positions: {} ({:.2} GiB)",
            occ.len(),
            occ.len() as f64
                * std::mem::size_of::<u32>() as f64
                / (
                1024.0
                    * 1024.0
                    * 1024.0
            ),
        );

        let mut index =
            Self {
                pairs,
                prev,
                next,
                occ,
                heap:
                BinaryHeap::new(),
                dirty:
                FastHashSet::default(),
            };

        // --------------------------------------------------------
        // Initial heap
        // --------------------------------------------------------

        for pair
        in index.pairs.iter_mut()
        {
            let count =
                pair.count();

            if count >= 2 {
                pair.set_queued(
                    true,
                );

                index.heap.push(
                    (
                        pair_score(
                            pair.key,
                            count,
                            meta,
                        ),
                        pair.key,
                        count,
                    ),
                );
            }
        }

        println!(
            "Initial heap entries: {}",
            index.heap.len(),
        );

        Ok(index)
    }

    #[inline]
    fn maybe_queue_pair(
        &mut self,
        key: u32,
        meta: &[TokenMeta],
    ) {
        let count;

        {
            let Some(pair) =
                self.pairs.get_mut(
                    key,
                )
            else {
                return;
            };

            count =
                pair.count();

            if count < 2
                || pair.queued()
            {
                return;
            }

            pair.set_queued(
                true,
            );
        }

        self.heap.push(
            (
                pair_score(
                    key,
                    count,
                    meta,
                ),
                key,
                count,
            ),
        );
    }

    #[inline]
    fn add_edge(
        &mut self,
        node: u32,
        a: u16,
        b: u16,
        meta: &[TokenMeta],
    ) -> std::io::Result<()> {
        if !TokenMeta::valid_fusion(
            meta[a as usize],
            meta[b as usize],
        ) {
            return Ok(());
        }

        let key =
            pair_key(a, b);

        let count;

        {
            let pair =
                self.pairs
                    .get_or_insert(key);

            count =
                pair.count()
                    + 1;

            assert!(
                count <= COUNT_MASK,
                "Pair occurrence count overflow",
            );

            pair.set_count(
                count,
            );
        }

        /*
         * IMPORTANT:
         *
         * Store even count=1 occurrences.
         * A singleton can become a merge candidate later.
        */
        self.occ.append(
            key,
            node,
        )?;

        self.dirty.insert(
            key,
        );

        self.maybe_queue_pair(
            key,
            meta,
        );

        Ok(())
    }

    #[inline]
    fn remove_edge(
        &mut self,
        a: u16,
        b: u16,
        meta: &[TokenMeta],
    ) {
        if !TokenMeta::valid_fusion(
            meta[a as usize],
            meta[b as usize],
        ) {
            return;
        }

        let key =
            pair_key(a, b);

        let new_count;

        {
            let Some(pair) =
                self.pairs.get_mut(key)
            else {
                return;
            };

            let old_count =
                pair.count();

            assert!(
                old_count > 0,
                "Attempted to remove pair with zero count",
            );

            new_count =
                old_count - 1;

            pair.set_count(
                new_count,
            );

            if new_count > 0 {
                pair.set_queued(
                    false,
                );
            }
        }

        /*
         * Once a pair has no live occurrences at all,
         * discard its stale occurrence vector immediately.
         */
        if new_count == 0 {
            self.occ.remove(
                key,
            );

            self.pairs.remove(
                key,
            );
        } else {
            self.dirty.insert(
                key,
            );
        }
    }

    fn take_occurrences(
        &mut self,
        key: u32,
    ) -> Vec<u32> {
        let pair =
            self.pairs
                .get_mut(key)
                .expect(
                    "selected pair disappeared",
                );

        assert!(
            pair.count() >= 2,
            "attempted to take a non-candidate pair",
        );

        pair.set_queued(
            false,
        );

        let occurrences =
            self.occ.take(key);

        debug_assert!(
            !occurrences.is_empty(),
            "selected pair has no occurrence list",
        );

        occurrences
    }

    fn flush_heap(
        &mut self,
        meta: &[TokenMeta],
    ) {
        for key in
            self.dirty.drain()
        {
            if let Some(count) =
                self.pairs
                    .get(key)
                    .map(
                        |pair| pair.count()
                    )
            {
                if count >= 2 {
                    self.heap.push(
                        (
                            pair_score(
                                key,
                                count,
                                meta,
                            ),
                            key,
                            count,
                        ),
                    );
                }
            }
        }
    }

    fn rebuild_heap_if_needed(
        &mut self,
        meta: &[TokenMeta],
    ) {
        let pair_count =
            self.pairs.len();

        let heap_limit =
            pair_count
                .saturating_add(
                    4096,
                );

        if self.heap.len()
            <= heap_limit
        {
            return;
        }

        self.heap.clear();

        for pair
        in self.pairs.iter_mut()
        {
            pair.set_queued(
                false,
            );
        }

        for pair
        in self.pairs.iter_mut()
        {
            let count =
                pair.count();

            if count >= 2 {
                pair.set_queued(
                    true,
                );

                self.heap.push(
                    (
                        pair_score(
                            pair.key,
                            count,
                            meta,
                        ),
                        pair.key,
                        count,
                    ),
                );
            }
        }
    }

    fn best_pair(
        &mut self,
        meta: &[TokenMeta],
    ) -> Option<(u32, u32)> {
        loop {
            let Some((
                         _score,
                         key,
                         snapshot_count,
                     )) = self.heap.pop()
            else {
                return None;
            };

            let current_count =
                match self.pairs
                    .get_mut(key)
                {
                    Some(pair) => {
                        pair.set_queued(
                            false,
                        );

                        pair.count()
                    }

                    None => {
                        continue;
                    }
                };

            if current_count
                == snapshot_count
                && current_count >= 2
            {
                return Some(
                    (
                        key,
                        current_count,
                    ),
                );
            }

            if current_count >= 2 {
                self.maybe_queue_pair(
                    key,
                    meta,
                );
            }
        }
    }
}

const MAX_CORPUS_BYTES: u64 = 2 * 1024 * 1024 * 1024;

fn tokenize_reader_u16<R: Read>(
    trie: &Trie,
    mut reader: R,
    initial_capacity: usize,
) -> std::io::Result<Vec<u16>> {
    const CHUNK_SIZE:
    usize = 4 * 1024 * 1024;

    let keep =
        trie.max_token_bytes;

    let mut total_input_bytes =
        0u64;

    let mut tokens =
        Vec::<u16>::with_capacity(
            initial_capacity,
        );

    let mut chunk =
        vec![0u8; CHUNK_SIZE];

    let mut pending =
        Vec::with_capacity(
            keep + CHUNK_SIZE,
        );

    loop {
        let read =
            reader.read(
                &mut chunk,
            )?;

        if read == 0 {
            break;
        }

        total_input_bytes =
            total_input_bytes
                .checked_add(
                    read as u64,
                )
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "input size overflow",
                    )
                })?;

        if total_input_bytes
            > MAX_CORPUS_BYTES
        {
            return Err(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "corpus exceeds 2 GiB limit ({} bytes)",
                        MAX_CORPUS_BYTES,
                    ),
                ),
            );
        }

        pending.extend_from_slice(
            &chunk[..read],
        );

        if pending.len()
            <= keep
        {
            continue;
        }

        let tokenize_len =
            pending.len()
                - keep;

        let ids =
            trie.tokenize_bytes_u16(
                &pending[..tokenize_len],
            );

        tokens.extend_from_slice(
            &ids,
        );

        if tokens.len()
            > u32::MAX as usize
        {
            return Err(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "token count exceeds u32 index capacity",
                ),
            );
        }

        pending.copy_within(
            tokenize_len..,
            0,
        );

        pending.truncate(
            keep,
        );
    }

    if !pending.is_empty() {
        let ids =
            trie.tokenize_bytes_u16(
                &pending,
            );

        tokens.extend_from_slice(
            &ids,
        );
    }

    if tokens.len()
        > u32::MAX as usize
    {
        return Err(
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "token count exceeds u32 index capacity",
            ),
        );
    }

    println!(
        "Input bytes: {:.2} GiB",
        total_input_bytes as f64
            / (
            1024.0
                * 1024.0
                * 1024.0
        ),
    );

    println!(
        "Initial tokens: {}",
        tokens.len(),
    );

    Ok(tokens)
}

// ------------------------------------------------------------
// Merge
// ------------------------------------------------------------

fn merge_pair(
    key: u32,
    new_id: u16,
    tokens: &mut [u16],
    pair_index: &mut PairIndex,
    meta: &[TokenMeta],
    occ_limit: usize,
) -> std::io::Result<usize> {
    /*
     * Take the entire occurrence vector out of the store.
     *
     * This is important:
     *
     * 1. We no longer need occurrence-record IDs.
     * 2. The selected vector can now be compacted/freed
     *    independently of the global occurrence store.
     * 3. `pop()` preserves the old newest -> oldest traversal
     *    order because occurrences were appended chronologically.
     */
    let mut occurrences =
        pair_index.take_occurrences(
            key,
        );

    let (wanted_a, wanted_b) =
        pair_ids(key);

    let mut merged =
        0usize;

    /*
     * If stale records have accumulated elsewhere, clean them
     * before starting a potentially huge merge.
     */
    if pair_index.occ.len()
        > occ_limit
    {
        pair_index
            .occ
            .compact(
                &pair_index.pairs,
                tokens,
                &pair_index.next,
            );
    }

    while let Some(
        cur_u32,
    ) = occurrences.pop()
    {
        let cur =
            cur_u32 as usize;

        /*
         * Node was consumed by an earlier merge.
         */
        if pair_index.next[cur]
            == REMOVED
        {
            continue;
        }

        let right_u32 =
            pair_index.next[cur];

        if right_u32 == NONE
            || right_u32 == REMOVED
        {
            continue;
        }

        let right =
            right_u32 as usize;

        /*
         * Verify that this historical occurrence is
         * still the pair we want.
         */
        if tokens[cur]
            != wanted_a
            || tokens[right]
            != wanted_b
        {
            continue;
        }

        let left_u32 =
            pair_index.prev[cur];

        let after_u32 =
            pair_index.next[right];

        // ----------------------------------------------------
        // Remove old edges
        // ----------------------------------------------------

        if left_u32 != NONE {
            let left =
                left_u32 as usize;

            let left_token =
                tokens[left];

            pair_index.remove_edge(
                left_token,
                wanted_a,
                meta,
            );
        }

        pair_index.remove_edge(
            wanted_a,
            wanted_b,
            meta,
        );

        if after_u32 != NONE {
            let after =
                after_u32 as usize;

            let after_token =
                tokens[after];

            pair_index.remove_edge(
                wanted_b,
                after_token,
                meta,
            );
        }

        // ----------------------------------------------------
        // Merge A+B -> NEW
        // ----------------------------------------------------

        tokens[cur] =
            new_id;

        pair_index.next[cur] =
            after_u32;

        if after_u32 != NONE {
            pair_index.prev[
                after_u32 as usize
                ] = cur_u32;
        }

        // Kill B.
        pair_index.next[right] =
            REMOVED;

        pair_index.prev[right] =
            REMOVED;

        // ----------------------------------------------------
        // Add NEW edges
        // ----------------------------------------------------

        if left_u32 != NONE {
            let left =
                left_u32 as usize;

            let left_token =
                tokens[left];

            pair_index.add_edge(
                left_u32,
                left_token,
                new_id,
                meta,
            )?;
        }

        if after_u32 != NONE {
            let after =
                after_u32 as usize;

            let after_token =
                tokens[after];

            pair_index.add_edge(
                cur_u32,
                new_id,
                after_token,
                meta,
            )?;
        }

        merged += 1;

        /*
         * Keep the occurrence store bounded.
         *
         * We can safely compact DURING the merge now because
         * the selected occurrence list is a separate Vec<u32>.
         */
        const COMPACT_CHECK:
        usize = 1 << 20;

        if merged % COMPACT_CHECK
            == 0
            && pair_index.occ.len()
            > occ_limit
        {
            println!(
                "    occurrence compaction at {} merged...",
                merged,
            );

            pair_index
                .occ
                .compact(
                    &pair_index.pairs,
                    tokens,
                    &pair_index.next,
                );
        }
    }

    /*
     * Flush heap exactly once per vocabulary merge.
     */
    pair_index.flush_heap(
        meta,
    );

    pair_index
        .rebuild_heap_if_needed(
            meta,
        );

    Ok(merged)
}

// ------------------------------------------------------------
// Vocabulary helpers
// ------------------------------------------------------------

fn add_byte_fallback_tokens(
    vocab: &mut Vec<String>,
) {
    for byte in 0u16..=255 {
        vocab.push(
            format!(
                "<0x{:02X}>",
                byte,
            ),
        );
    }
}

// ------------------------------------------------------------
// Main vocabulary builder
// ------------------------------------------------------------

fn build_vocab_from_reader_with_capacity<R: Read>(
    reader: R,
    token_num: usize,
    reserved_token_num: usize,
    special_tokens: Option<&[&str]>,
    initial_token_capacity: usize,
) -> std::io::Result<Vec<String>> {
    let special_count =
        special_tokens.map_or(
            0,
            |s| s.len(),
        );

    let initial_vocab_size =
        reserved_token_num
            + 1
            + special_count
            + 256;

    let target_vocab_size =
        initial_vocab_size
            + token_num;

    assert!(
        target_vocab_size
            <= u16::MAX as usize,
        "Vocabulary is too large for u16 token IDs",
    );

    let mut vocab =
        Vec::with_capacity(
            target_vocab_size,
        );

    vocab.resize(
        reserved_token_num + 1,
        String::new(),
    );

    if let Some(special) =
        special_tokens
    {
        vocab.extend(
            special
                .iter()
                .map(
                    |token| {
                        (*token)
                            .to_string()
                    },
                ),
        );
    }

    let byte_fallback_base =
        vocab.len();

    add_byte_fallback_tokens(
        &mut vocab,
    );

    let mut meta =
        Vec::with_capacity(
            target_vocab_size,
        );

    for id in 0..vocab.len() {
        meta.push(
            TokenMeta::from_token(
                &vocab[id],
                id >= byte_fallback_base,
            ),
        );
    }

    let trie =
        Trie::from_vocab(
            &vocab,
            byte_fallback_base as u32,
        );

    let mut tokens =
        tokenize_reader_u16(
            &trie,
            reader,
            initial_token_capacity,
        )?;

    drop(trie);

    if tokens.len() == 0 {
        return Ok(vocab);
    }

    let mut pair_index =
        PairIndex::build(
            &mut tokens,
            &meta,
        )?;
    let mut merged_count = 0;
    while vocab.len() < target_vocab_size {
        let Some((
                     best_key,
                     count,
                 )) =
            pair_index.best_pair(
                &meta,
            )
        else {
            println!("Not enough text to make {} tokens!", token_num, );
            break;
        };

        if count < 2 {
            break;
        }

        let (a, b) =
            pair_ids(
                best_key,
            );

        // ----------------------------------------------------
        // Construct new token from DECODED bytes.
        //
        // This is still token-ID-based BPE:
        //
        //     token_id A + token_id B
        //
        // The decoded bytes are only used to create the
        // vocabulary representation of the NEW token.
        // ----------------------------------------------------

        let decoded_len =
            meta[a as usize]
                .byte_len
                as usize
                + meta[b as usize]
                .byte_len
                as usize;

        let mut bytes =
            Vec::with_capacity(
                decoded_len,
            );

        append_token_bytes(
            &vocab[a as usize],
            &mut bytes,
        );

        append_token_bytes(
            &vocab[b as usize],
            &mut bytes,
        );

        let new_token =
            String::from_utf8(
                bytes,
            ).unwrap_or_else(|e| {
                let bytes =
                    e.into_bytes();

                let mut encoded =
                    String::with_capacity(
                        bytes.len()
                            * 6,
                    );

                for byte in bytes {
                    use std::fmt::Write;

                    write!(
                        encoded,
                        "<0x{:02X}>",
                        byte,
                    )
                        .unwrap();
                }

                encoded
            });

        let new_meta =
            TokenMeta::fuse(
                meta[a as usize],
                meta[b as usize],
            );

        let new_id =
            vocab.len()
                as u16;

        // Exactly the old semantics:
        //
        // every successful merge creates
        // a fresh vocabulary ID.
        vocab.push(
            new_token,
        );

        meta.push(
            new_meta,
        );

        let occ_limit =
            tokens.len()
                .saturating_add(
                    tokens.len() / 2,
                );

        let merged =
            merge_pair(
                best_key,
                new_id,
                &mut tokens,
                &mut pair_index,
                &meta,
                occ_limit,
            )?;

        merged_count += 1;
        println!("Number of merged tokens: {}", merged_count);

        assert!(
            merged > 0,
            "Best pair existed in the count table but had no live occurrences: key={:08X}, count={}",
            best_key,
            count,
        );
    }

    drop(pair_index);
    drop(tokens);

    Ok(vocab)
}


// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------

pub fn make_vocab_from_reader<R: Read>(
    reader: R,
    token_num: usize,
    reserved_token_num: usize,
    special_tokens: Option<&[&str]>,
) -> std::io::Result<Vec<String>> {
    let now =
        std::time::Instant::now();

    let vocab =
        build_vocab_from_reader_with_capacity(
            reader,
            token_num,
            reserved_token_num,
            special_tokens,
            1 << 20,
        )?;

    println!(
        "Vocabulary creation finished in {:?}.",
        now.elapsed(),
    );

    Ok(vocab)
}

pub fn make_vocab_from_file(
    path: &str,
    token_num: usize,
    reserved_token_num: usize,
    special_tokens: Option<&[&str]>,
) -> std::io::Result<Vec<String>> {
    let now =
        std::time::Instant::now();

    let file =
        File::open(path)?;

    let reader =
        BufReader::with_capacity(
            1024 * 1024,
            file,
        );

    let vocab =
        build_vocab_from_reader_with_capacity(
            reader,
            token_num,
            reserved_token_num,
            special_tokens,
            1 << 20,
        )?;

    println!(
        "Total vocabulary pipeline time: {:?}.",
        now.elapsed(),
    );

    Ok(vocab)
}
