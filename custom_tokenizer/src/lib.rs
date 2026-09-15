/*!
whitelist_tokenizer_rs

# Token ID layout (deterministic, built from word_whitelist at startup)
--------------------------------------------------------------------------
  0 .. N_FIXED-1   : fixed tokens (punct, space, newline, "'s" suffix)
                     — in the order defined in code
  N_FIXED ..       : whitelist words + whole contractions — in file/definition order
  (last 10)        : special tokens — in the order defined in SPECIAL_TOKENS

# Spacing model
---------------
  The pretokenizer emits whitespace as explicit `Sp` and `Nl` stream items.
  The encoder maps them to token ids directly:

  • Nl  → newline_id (always)
  • Sp  → space_id   (always)
  • Word→Word single space: the *first* `Sp` between two word-kind tokens is
    implicit (dropped); every additional `Sp` emits one space_id.
    e.g. "dog  cat" → [Sp, Sp] between words → drop first, emit one space_id.
  • Fixed tokens never consume implicit spaces; every Sp before/after a Fixed
    token is emitted as-is.

  This keeps the encoder to a linear, stateless pass — no lookahead, no
  spaces_before/after arithmetic.

# On-disk format
----------------
  word_whitelist — one lowercase word per line, file order preserved.
                   Lines starting with # and blank lines are ignored.
*/

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use once_cell::sync::Lazy;
use regex::Regex;

// ---------------------------------------------------------------------------
// Fixed vocabulary — order here determines token ids
// ---------------------------------------------------------------------------

pub const PUNCT_TOKENS: &[&str] = &[".", ",", "'", "\"", "?", "!", "(", ")"];
pub const SPACE_TOKEN:   &str   = " ";
pub const NEWLINE_TOKEN: &str   = "\n";
pub const CONTRACTION_SUFFIXES: &[&str] = &["'s"];

/// Whole contractions — fixed order so ids are always deterministic.
pub const WHOLE_CONTRACTIONS: &[&str] = &[
    // negatives
    "aren't", "can't", "couldn't",
    "didn't", "doesn't", "don't",
    "haven't", "isn't",
    "shouldn't", "wasn't", "weren't",
    "won't", "wouldn't",
    // be
    "i'm", "you're", "he's", "she's", "it's", "we're", "they're",
    // have
    "i've", "you've", "we've", "they've",
    // will
    "i'll", "you'll", "he'll", "she'll", "it'll", "we'll", "they'll",
    // would / had
    "i'd", "you'd",
    // miscellaneous
    "let's",
];

pub const SPECIAL_TOKENS: &[&str] = &[
    "<|bos|>",
    "<|unk|>",
    "<|mask|>",
    "<|unused|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|output_start|>",
    "<|output_end|>",
];

// ---------------------------------------------------------------------------
// Stream items produced by the pretokenizer
// ---------------------------------------------------------------------------

/// A single item in the pretokenized stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamItem {
    /// An ASCII-alphabetic word or whole contraction (word-kind: implicit spacing).
    Word(String),
    /// A fixed-vocab token: punctuation or contraction suffix "'s".
    Fixed(String),
    /// Anything that cannot be represented (digit, non-ASCII, etc.).
    Unk,
    /// One space character (may appear multiple times in a run).
    Sp,
    /// One newline character.
    Nl,
}

// ---------------------------------------------------------------------------
// Static tables
// ---------------------------------------------------------------------------

static WHOLE_CONTRACTION_SET: Lazy<std::collections::HashSet<&'static str>> =
    Lazy::new(|| WHOLE_CONTRACTIONS.iter().copied().collect());

static ALLOWED_PUNCT: Lazy<std::collections::HashSet<char>> =
    Lazy::new(|| ['.', ',', '\'', '"', '?', '!', '(', ')'].iter().copied().collect());

static SUFFIX_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)('s)$").unwrap());
static DASH_RE:   Lazy<Regex> = Lazy::new(|| Regex::new(r"[-\u{2013}\u{2014}]").unwrap());

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

fn normalize(text: &str) -> String {
    text.chars().map(|ch| match ch {
        '\u{2018}' | '\u{2019}' | '\u{02BC}' | '\u{0060}' => '\'',
        '\u{201C}' | '\u{201D}' => '"',
        other => other,
    }).collect()
}

// ---------------------------------------------------------------------------
// Surface splitting — returns Word/Fixed/Unk items only (no Sp/Nl)
// ---------------------------------------------------------------------------

fn split_surface(surface: &str) -> Vec<StreamItem> {
    let mut result = Vec::new();
    let chars: Vec<char> = surface.chars().collect();

    // Peel leading allowed punctuation.
    let mut lo = 0;
    while lo < chars.len() && ALLOWED_PUNCT.contains(&chars[lo]) {
        result.push(StreamItem::Fixed(chars[lo].to_string()));
        lo += 1;
    }

    // Peel trailing allowed punctuation.
    let mut hi = chars.len();
    let mut trailing = Vec::new();
    while hi > lo && ALLOWED_PUNCT.contains(&chars[hi - 1]) {
        trailing.push(StreamItem::Fixed(chars[hi - 1].to_string()));
        hi -= 1;
    }
    trailing.reverse();

    let core: String = chars[lo..hi].iter().collect();
    if !core.is_empty() {
        result.extend(classify_core(&core));
    } else if trailing.is_empty() {
        // entire surface was punct — spaces_after is handled by caller, nothing extra here
    }

    result.extend(trailing);
    result
}

/// Classify a dash-free, punctuation-stripped core.
fn classify_core(core: &str) -> Vec<StreamItem> {
    let lower = core.to_lowercase();

    // Whole contraction → Word kind.  Must be checked before any splitting.
    if WHOLE_CONTRACTION_SET.contains(lower.as_str()) {
        return vec![StreamItem::Word(lower)];
    }

    // Contraction suffix  "dog's" → [Word("dog"), Fixed("'s")]
    // Also checked before embedded-punct split so "don't" isn't torn apart at "'".
    if let Some(m) = SUFFIX_RE.find(core) {
        let bare = &core[..m.start()];
        let mut parts = Vec::new();
        if !bare.is_empty() {
            parts.extend(classify_core(bare));
        }
        parts.push(StreamItem::Fixed(m.as_str().to_lowercase()));
        return parts;
    }

    // Embedded punctuation  "dog,cat" → [Word("dog"), Fixed(","), Word("cat")]
    // We skip apostrophes here — they're handled by the contraction/suffix
    // checks above. Any apostrophe that survived to this point is part of
    // a token that couldn't be classified (e.g. "don't,you" splits only on
    // the comma, leaving "don't" to be re-classified as a whole contraction).
    let chars: Vec<char> = core.chars().collect();
    if chars.iter().any(|c| *c != '\'' && ALLOWED_PUNCT.contains(c)) {
        let mut parts = Vec::new();
        let mut seg_start = 0;
        for i in 0..chars.len() {
            if chars[i] != '\'' && ALLOWED_PUNCT.contains(&chars[i]) {
                if i > seg_start {
                    let seg: String = chars[seg_start..i].iter().collect();
                    parts.extend(classify_core(&seg));
                }
                parts.push(StreamItem::Fixed(chars[i].to_string()));
                seg_start = i + 1;
            }
        }
        if seg_start < chars.len() {
            let seg: String = chars[seg_start..].iter().collect();
            parts.extend(classify_core(&seg));
        }
        return parts;
    }

    // Dash split — each dash becomes Unk.
    let dash_parts: Vec<&str> = DASH_RE.split(core).collect();
    if dash_parts.len() > 1 {
        let mut parts = Vec::new();
        for (idx, part) in dash_parts.iter().enumerate() {
            if idx > 0 { parts.push(StreamItem::Unk); }
            if !part.is_empty() { parts.extend(classify_core(part)); }
        }
        return parts;
    }

    // Bare word or unk.
    if core.chars().all(|c| c.is_ascii_alphabetic()) {
        vec![StreamItem::Word(lower)]
    } else {
        vec![StreamItem::Unk]
    }
}

// ---------------------------------------------------------------------------
// Public pretokenizer
// ---------------------------------------------------------------------------

/// Tokenize `text` into a flat stream of [`StreamItem`]s.
///
/// Whitespace is represented as explicit `Sp` and `Nl` items so the encoder
/// needs no lookahead or spacing arithmetic.
pub fn pretokenize(text: &str) -> Vec<StreamItem> {
    let normalized = normalize(text);
    let chars: Vec<char> = normalized.chars().collect();
    let n = chars.len();
    let mut items = Vec::new();
    let mut i = 0;

    while i < n {
        match chars[i] {
            '\n' => {
                items.push(StreamItem::Nl);
                i += 1;
            }
            c if c.is_whitespace() => {
                items.push(StreamItem::Sp);
                i += 1;
            }
            _ => {
                // Collect non-whitespace run.
                let start = i;
                while i < n && !chars[i].is_whitespace() {
                    i += 1;
                }
                let surface: String = chars[start..i].iter().collect();
                items.extend(split_surface(&surface));
            }
        }
    }

    items
}

// ---------------------------------------------------------------------------
// WhitelistTokenizer
// ---------------------------------------------------------------------------

pub struct WhitelistTokenizer {
    pub word_ids:   HashMap<String, u32>,
    pub id_to_word: Vec<String>,
    pub space_id:   u32,
    pub newline_id: u32,
    pub unk_id:     u32,
    pub bos_id:     u32,
    pub mask_id:     u32,
    /// One past the last fixed-token id.
    pub n_fixed:    u32,
    pub special_ids: HashMap<String, u32>,
}

impl WhitelistTokenizer {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    pub fn from_directory(dir: &Path) -> anyhow::Result<Self> {
        let words_path = dir.join("word_whitelist");
        let words_text = fs::read_to_string(&words_path)
            .map_err(|e| anyhow::anyhow!("Cannot read {}: {e}", words_path.display()))?;
        Self::from_allowed_words(words_text.lines())
    }

    pub fn from_allowed_words<'a>(
        lines: impl Iterator<Item = &'a str>,
    ) -> anyhow::Result<Self> {
        let mut word_ids:  HashMap<String, u32> = HashMap::new();
        let mut id_to_word: Vec<String>         = Vec::new();
        let mut next_id: u32 = 0;

        let assign = |surface: &str, wids: &mut HashMap<String,u32>, itw: &mut Vec<String>, nid: &mut u32| {
            if !wids.contains_key(surface) {
                wids.insert(surface.to_string(), *nid);
                itw.push(surface.to_string());
                *nid += 1;
            }
        };

        // 1. Fixed tokens
        for &s in PUNCT_TOKENS          { assign(s, &mut word_ids, &mut id_to_word, &mut next_id); }
        assign(SPACE_TOKEN,  &mut word_ids, &mut id_to_word, &mut next_id);
        assign(NEWLINE_TOKEN,&mut word_ids, &mut id_to_word, &mut next_id);
        for &s in CONTRACTION_SUFFIXES  { assign(s, &mut word_ids, &mut id_to_word, &mut next_id); }

        let n_fixed = next_id;

        // 2. Whitelist words + whole contractions — file order, no sorting
        for &s in WHOLE_CONTRACTIONS    { assign(s, &mut word_ids, &mut id_to_word, &mut next_id); }
        for line in lines {
            let w = line.trim();
            if w.is_empty() || w.starts_with('#') { continue; }
            let w = w.to_lowercase();
            if w.chars().all(|c| c.is_ascii_alphabetic()) {
                assign(&w, &mut word_ids, &mut id_to_word, &mut next_id);
            }
        }

        // 3. Special tokens
        let mut special_ids: HashMap<String, u32> = HashMap::new();
        for &name in SPECIAL_TOKENS {
            special_ids.insert(name.to_string(), next_id);
            word_ids.insert(name.to_string(), next_id);
            id_to_word.push(name.to_string());
            next_id += 1;
        }

        let space_id   = word_ids[SPACE_TOKEN];
        let newline_id = word_ids[NEWLINE_TOKEN];
        let unk_id     = special_ids["<|unk|>"];
        let bos_id     = special_ids["<|bos|>"];
        let mask_id     = special_ids["<|mask|>"];

        Ok(WhitelistTokenizer { word_ids, id_to_word, space_id, newline_id, unk_id, bos_id, mask_id, n_fixed, special_ids })
    }

    // ------------------------------------------------------------------
    // Encoding
    // ------------------------------------------------------------------

    pub fn encode(&self, text: &str, prepend_bos: bool) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 4 + 4);
        if prepend_bos { ids.push(self.bos_id); }
        self.encode_stream(&pretokenize(text), &mut ids);
        ids
    }

    pub fn encode_batch(&self, texts: &[&str], prepend_bos: bool) -> Vec<Vec<u32>> {
        use rayon::prelude::*;
        texts.par_iter().map(|t| self.encode(t, prepend_bos)).collect()
    }

    /// Walk the stream linearly.
    ///
    /// The only non-trivial rule: a single `Sp` between two word-kind items is
    /// implicit and produces no token. Every other `Sp` emits `space_id`.
    fn encode_stream(&self, stream: &[StreamItem], ids: &mut Vec<u32>) {
        // We need to know whether the *previous content item* (non-Sp, non-Nl)
        // was word-kind, so that we can decide whether to drop one leading Sp.
        //
        // Strategy: iterate over runs of (Sp*)(content_item).
        // For each content item, first resolve pending spaces then emit the item.

        let mut prev_word = false; // was the last non-space item word-kind?
        let mut sp_run = 0u32;     // spaces accumulated since last content item

        for item in stream {
            match item {
                StreamItem::Sp => {
                    sp_run += 1;
                }
                StreamItem::Nl => {
                    // Flush any pending spaces as-is (they precede a newline,
                    // which is always fixed — no implicit space consumed).
                    for _ in 0..sp_run { ids.push(self.space_id); }
                    sp_run = 0;
                    ids.push(self.newline_id);
                    prev_word = false;
                }
                StreamItem::Word(text) => {
                    // One space is implicit between word→word; drop it.
                    let emit = if prev_word && sp_run > 0 { sp_run - 1 } else { sp_run };
                    for _ in 0..emit { ids.push(self.space_id); }
                    sp_run = 0;
                    ids.push(self.word_ids.get(text.as_str()).copied().unwrap_or(self.unk_id));
                    prev_word = true;
                }
                StreamItem::Unk => {
                    let emit = if prev_word && sp_run > 0 { sp_run - 1 } else { sp_run };
                    for _ in 0..emit { ids.push(self.space_id); }
                    sp_run = 0;
                    ids.push(self.unk_id);
                    prev_word = true; // unk is word-kind for spacing
                }
                StreamItem::Fixed(text) => {
                    // Fixed tokens: every pending space emits explicitly (no implicit).
                    for _ in 0..sp_run { ids.push(self.space_id); }
                    sp_run = 0;
                    ids.push(self.word_ids.get(text.as_str()).copied().unwrap_or(self.unk_id));
                    // "'s" and ")" act like word-closers: a following word's single
                    // space is still implicit, matching the decoder's behaviour.
                    prev_word = matches!(text.as_str(), "'s" | ")");
                }
            }
        }
        // Trailing spaces (e.g. "hello  ") — emit as-is, no implicit consumed
        // because there's no following content item.
        for _ in 0..sp_run { ids.push(self.space_id); }
    }

    // ------------------------------------------------------------------
    // Decoding
    // ------------------------------------------------------------------

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::with_capacity(ids.len() * 4);
        let mut prev_was_word = false;

        for &id in ids {
            if let Some(s) = self.id_to_word.get(id as usize) {
                if id == self.space_id || id == self.newline_id {
                    out.push_str(s);
                    if id == self.newline_id { prev_was_word = false; }
                    // space_id does NOT clear prev_was_word: the implicit space
                    // will still be inserted before the next word, which is correct
                    // because space_id tokens represent *extra* spaces beyond the
                    // first implicit one.
                } else {
                    let is_word = id == self.unk_id || id >= self.n_fixed;
                    if prev_was_word && is_word {
                        out.push(' ');
                    }
                    out.push_str(s);
                    prev_was_word = is_word || matches!(s.as_str(), "'s" | ")");
                }
            }
        }
        out
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_word.get(id as usize).map(|s| s.as_str())
    }

    pub fn vocab_size(&self) -> usize { self.id_to_word.len() }
    pub fn bos_id(&self)     -> u32   { self.bos_id }
    pub fn mask_id(&self)    -> u32   { self.mask_id }
    pub fn unk_id(&self)     -> u32   { self.unk_id }
    pub fn space_id(&self)   -> u32   { self.space_id }
    pub fn newline_id(&self) -> u32   { self.newline_id }
    pub fn special_id(&self, name: &str) -> Option<u32> { self.special_ids.get(name).copied() }
}

// ---------------------------------------------------------------------------
// Python extension
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
pub mod python;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> WhitelistTokenizer {
        WhitelistTokenizer::from_allowed_words(
            "left right well known really he said dog cat fox hello world toy know let i you do".split_whitespace()
        ).unwrap()
    }

    fn roundtrip(tok: &WhitelistTokenizer, input: &str) -> String {
        tok.decode(&tok.encode(input, false))
    }

    // --- vocab layout ------------------------------------------------------

    #[test]
    fn test_fixed_tokens_have_ids_below_n_fixed() {
        let tok = tok();
        for &s in PUNCT_TOKENS.iter().chain(&[SPACE_TOKEN, NEWLINE_TOKEN]).chain(CONTRACTION_SUFFIXES) {
            let id = tok.word_ids[s];
            assert!(id < tok.n_fixed, "fixed token {s:?} should have id < n_fixed");
        }
    }

    #[test]
    fn test_whole_contractions_have_ids_above_n_fixed() {
        let tok = tok();
        for &s in WHOLE_CONTRACTIONS {
            let id = tok.word_ids[s];
            assert!(id >= tok.n_fixed, "contraction {s:?} should have id >= n_fixed");
        }
    }

    #[test]
    fn test_special_tokens_have_ids_above_n_fixed() {
        let tok = tok();
        for &s in SPECIAL_TOKENS {
            let id = tok.word_ids[s];
            assert!(id >= tok.n_fixed, "special token {s:?} should have id >= n_fixed");
        }
    }

    #[test]
    fn test_vocab_is_deterministic() {
        let tok1 = tok();
        let tok2 = tok();
        assert_eq!(tok1.word_ids, tok2.word_ids);
    }

    // --- encode/decode round-trips -----------------------------------------

    #[test]
    fn test_plain_words_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog cat fox"), "dog cat fox");
    }

    #[test]
    fn test_double_space_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog  cat"), "dog  cat");
    }

    #[test]
    fn test_triple_space_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog   cat"), "dog   cat");
    }

    #[test]
    fn test_many_spaces_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog     cat"), "dog     cat");
    }

    #[test]
    fn test_newline_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog\ncat"), "dog\ncat");
    }

    #[test]
    fn test_multiple_newlines_not_collapsed() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog\n\n\ncat"), "dog\n\n\ncat");
    }

    #[test]
    fn test_inter_newline_spaces_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog\n  \ncat"), "dog\n  \ncat");
    }

    #[test]
    fn test_trailing_punct_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog, cat."), "dog, cat.");
    }

    #[test]
    fn test_space_padded_comma_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog , cat"), "dog , cat");
    }

    #[test]
    fn test_multi_space_padded_comma_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog  ,  cat"), "dog  ,  cat");
    }

    #[test]
    fn test_no_space_comma_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog,cat"), "dog,cat");
    }

    #[test]
    fn test_possessive_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog's toy"), "dog's toy");
    }

    #[test]
    fn test_whole_contraction_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i don't know"), "i don't know");
    }

    #[test]
    fn test_all_whole_contractions_roundtrip() {
        let tok = tok();
        for &c in WHOLE_CONTRACTIONS {
            assert_eq!(roundtrip(&tok, c), c, "contraction {c:?} failed to round-trip");
        }
    }

    #[test]
    fn test_contraction_spacing_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i don't know"), "i don't know");
        assert_eq!(roundtrip(&tok, "i  don't  know"), "i  don't  know");
    }

    #[test]
    fn test_unk_roundtrip_produces_unk_token() {
        let tok = tok();
        let ids = tok.encode("hello 42 world", false);
        assert!(ids.contains(&tok.unk_id));
    }

    #[test]
    fn test_unknown_word_becomes_unk() {
        let tok = tok();
        let ids = tok.encode("xylophone", false);
        assert!(ids.contains(&tok.unk_id));
    }

    #[test]
    fn test_bos_prepended_when_requested() {
        let tok = tok();
        assert_eq!(tok.encode("dog", true)[0], tok.bos_id);
    }

    #[test]
    fn test_bos_not_prepended_by_default() {
        let tok = tok();
        assert_ne!(tok.encode("dog", false)[0], tok.bos_id);
    }

    #[test]
    fn test_curly_apostrophe_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog\u{2019}s toy"), "dog's toy");
    }

    #[test]
    fn test_special_token_ids_accessible() {
        let tok = tok();
        for &name in SPECIAL_TOKENS {
            assert!(tok.special_id(name).is_some());
        }
    }

    #[test]
    fn test_id_to_token_roundtrip() {
        let tok = tok();
        for (id, surface) in tok.id_to_word.iter().enumerate() {
            assert_eq!(tok.id_to_token(id as u32), Some(surface.as_str()));
        }
    }

    #[test]
    fn test_encode_batch_matches_sequential() {
        let tok = tok();
        let texts = vec!["dog cat", "hello world", "dog  cat", "i don't know"];
        let batch = tok.encode_batch(&texts, false);
        for (text, ids) in texts.iter().zip(batch.iter()) {
            assert_eq!(ids, &tok.encode(text, false));
        }
    }

    // --- explicit token-sequence checks ------------------------------------

    #[test]
    fn test_single_space_produces_no_explicit_space_token() {
        let tok = tok();
        let ids = tok.encode("dog cat", false);
        assert!(!ids.contains(&tok.space_id));
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_double_space_produces_one_explicit_space_token() {
        let tok = tok();
        let ids = tok.encode("dog  cat", false);
        assert_eq!(ids.iter().filter(|&&id| id == tok.space_id).count(), 1,
            "two spaces should produce exactly one explicit SPACE token");
    }

    #[test]
    fn test_triple_space_produces_two_explicit_space_tokens() {
        let tok = tok();
        let ids = tok.encode("dog   cat", false);
        assert_eq!(ids.iter().filter(|&&id| id == tok.space_id).count(), 2);
    }

    #[test]
    fn test_n_spaces_produces_n_minus_one_explicit_space_tokens() {
        let tok = tok();
        for n in 1u32..=6 {
            let input = format!("dog{}cat", " ".repeat(n as usize));
            let ids = tok.encode(&input, false);
            let explicit = ids.iter().filter(|&&id| id == tok.space_id).count() as u32;
            assert_eq!(explicit, n.saturating_sub(1),
                "{n} spaces should produce {} explicit SPACE tokens", n.saturating_sub(1));
        }
    }

    #[test]
    fn test_contraction_single_space_produces_no_explicit_space_token() {
        let tok = tok();
        let ids = tok.encode("i don't know", false);
        assert!(!ids.contains(&tok.space_id));
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_contraction_double_space_produces_explicit_space_tokens() {
        let tok = tok();
        let ids = tok.encode("i  don't  know", false);
        assert_eq!(ids.iter().filter(|&&id| id == tok.space_id).count(), 2,
            "two double-spaces around contraction should produce 2 explicit SPACE tokens");
    }

    #[test]
    fn test_newline_produces_newline_token() {
        let tok = tok();
        let ids = tok.encode("dog\ncat", false);
        assert!(ids.contains(&tok.newline_id));
        assert!(!ids.contains(&tok.space_id));
    }

    #[test]
    fn test_three_newlines_produce_three_newline_tokens() {
        let tok = tok();
        let ids = tok.encode("dog\n\n\ncat", false);
        assert_eq!(ids.iter().filter(|&&id| id == tok.newline_id).count(), 3);
    }

    #[test]
    fn test_inter_newline_spaces_produce_space_tokens() {
        let tok = tok();
        let ids = tok.encode("dog\n  \ncat", false);
        assert_eq!(ids.iter().filter(|&&id| id == tok.newline_id).count(), 2);
        assert_eq!(ids.iter().filter(|&&id| id == tok.space_id).count(), 2);
    }

    #[test]
    fn test_space_padded_comma_token_sequence() {
        let tok = tok();
        let dog_id   = tok.word_ids["dog"];
        let comma_id = tok.word_ids[","];
        let cat_id   = tok.word_ids["cat"];
        let sp       = tok.space_id;
        let ids = tok.encode("dog , cat", false);
        assert_eq!(ids, vec![dog_id, sp, comma_id, sp, cat_id]);
    }

    #[test]
    fn test_no_space_comma_token_sequence() {
        let tok = tok();
        let ids = tok.encode("dog,cat", false);
        assert!(!ids.contains(&tok.space_id));
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_possessive_token_sequence() {
        let tok = tok();
        let ids = tok.encode("dog's toy", false);
        assert_eq!(ids, vec![tok.word_ids["dog"], tok.word_ids["'s"], tok.word_ids["toy"]]);
    }

    #[test]
    fn test_contraction_token_sequence() {
        let tok = tok();
        let ids = tok.encode("i don't know", false);
        assert_eq!(ids, vec![tok.word_ids["i"], tok.word_ids["don't"], tok.word_ids["know"]]);
    }

    // --- contraction spacing -----------------------------------------------

    #[test]
    fn test_contraction_single_space_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i don't know"), "i don't know");
    }

    #[test]
    fn test_contraction_double_space_not_collapsed() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i  don't  know"), "i  don't  know");
    }

    #[test]
    fn test_contraction_triple_space_not_collapsed() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i   don't   know"), "i   don't   know");
    }

    #[test]
    fn test_contraction_many_spaces_not_collapsed() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i     don't     know"), "i     don't     know");
    }

    #[test]
    fn test_contraction_newline_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i\ndon't\nknow"), "i\ndon't\nknow");
    }

    #[test]
    fn test_contraction_multiple_newlines_not_collapsed() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i\n\n\ndon't"), "i\n\n\ndon't");
    }

    #[test]
    fn test_contraction_inter_newline_spaces_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i\n  \ndon't"), "i\n  \ndon't");
    }

    #[test]
    fn test_contraction_trailing_punct_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i don't, you know."), "i don't, you know.");
    }

    #[test]
    fn test_contraction_space_padded_comma_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i don't , you know"), "i don't , you know");
    }

    #[test]
    fn test_contraction_multi_space_padded_comma_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i don't  ,  you know"), "i don't  ,  you know");
    }

    #[test]
    fn test_contraction_no_space_comma_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i don't,you know"), "i don't,you know");
    }

    #[test]
    fn test_all_contractions_single_space_roundtrip() {
        let tok = tok();
        for &c in WHOLE_CONTRACTIONS {
            let input = format!("i {c} know");
            assert_eq!(roundtrip(&tok, &input), input, "contraction {c:?} failed single-space round-trip");
        }
    }

    #[test]
    fn test_all_contractions_double_space_roundtrip() {
        let tok = tok();
        for &c in WHOLE_CONTRACTIONS {
            let input = format!("i  {c}  know");
            assert_eq!(roundtrip(&tok, &input), input, "contraction {c:?} failed double-space round-trip");
        }
    }

    // --- mixed spacing and newlines ----------------------------------------

    #[test]
    fn test_spaces_then_newline_then_spaces_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog  \n  cat"), "dog  \n  cat");
    }

    #[test]
    fn test_word_newline_word_newline_word_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog\ncat\nfox"), "dog\ncat\nfox");
    }

    #[test]
    fn test_blank_line_between_words_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog\n\ncat"), "dog\n\ncat");
    }

    #[test]
    fn test_many_blank_lines_not_collapsed() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog\n\n\n\n\ncat"), "dog\n\n\n\n\ncat");
    }

    #[test]
    fn test_spaces_and_newlines_mixed_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog  cat\n\nfox   hello\nworld"), "dog  cat\n\nfox   hello\nworld");
    }

    #[test]
    fn test_contraction_in_multiline_text_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i don't know\n\ndo you?"), "i don't know\n\ndo you?");
    }

    #[test]
    fn test_contraction_with_spaces_before_newline_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "i  don't\n\n  know"), "i  don't\n\n  know");
    }

    // --- empty and single-token inputs ------------------------------------

    #[test]
    fn test_empty_string_encodes_to_empty() {
        let tok = tok();
        assert_eq!(tok.encode("", false), vec![]);
    }

    #[test]
    fn test_empty_string_decodes_to_empty() {
        let tok = tok();
        assert_eq!(tok.decode(&[]), "");
    }

    #[test]
    fn test_single_word_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog"), "dog");
    }

    #[test]
    fn test_single_word_no_space_tokens() {
        let tok = tok();
        let ids = tok.encode("dog", false);
        assert_eq!(ids.len(), 1);
        assert!(!ids.contains(&tok.space_id));
        assert!(!ids.contains(&tok.newline_id));
    }

    #[test]
    fn test_single_unknown_word_is_one_unk() {
        let tok = tok();
        let ids = tok.encode("xylophone", false);
        assert_eq!(ids, vec![tok.unk_id]);
    }

    #[test]
    fn test_single_punct_encodes_to_one_token() {
        let tok = tok();
        let ids = tok.encode(".", false);
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], tok.word_ids["."]);
    }

    #[test]
    fn test_single_newline_encodes_to_one_newline_token() {
        let tok = tok();
        let ids = tok.encode("\n", false);
        assert_eq!(ids, vec![tok.newline_id]);
    }

    // --- BOS token placement ----------------------------------------------

    #[test]
    fn test_bos_on_empty_string() {
        let tok = tok();
        let ids = tok.encode("", true);
        assert_eq!(ids, vec![tok.bos_id]);
    }

    #[test]
    fn test_bos_followed_by_word() {
        let tok = tok();
        let ids = tok.encode("dog", true);
        assert_eq!(ids[0], tok.bos_id);
        assert_eq!(ids[1], tok.word_ids["dog"]);
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_bos_id_not_in_encode_without_flag() {
        let tok = tok();
        let ids = tok.encode("dog cat fox", false);
        assert!(!ids.contains(&tok.bos_id));
    }

    // --- case normalisation -----------------------------------------------

    #[test]
    fn test_uppercase_word_normalised_to_lowercase() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "DOG"), "dog");
    }

    #[test]
    fn test_mixed_case_word_normalised() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "Dog"), "dog");
    }

    #[test]
    fn test_uppercase_encodes_same_as_lowercase() {
        let tok = tok();
        assert_eq!(tok.encode("DOG", false), tok.encode("dog", false));
    }

    #[test]
    fn test_uppercase_contraction_normalised() {
        let tok = tok();
        assert_eq!(tok.encode("DON'T", false), tok.encode("don't", false));
    }

    // --- normalisation: smart quotes --------------------------------------

    #[test]
    fn test_left_single_quote_normalised() {
        // U+2018 LEFT SINGLE QUOTATION MARK
        let tok = tok();
        assert_eq!(tok.encode("\u{2018}hello\u{2019}", false),
                   tok.encode("'hello'", false));
    }

    #[test]
    fn test_grave_accent_normalised_to_apostrophe() {
        // U+0060 GRAVE ACCENT used as apostrophe
        let tok = tok();
        assert_eq!(tok.encode("dog\u{0060}s", false),
                   tok.encode("dog's", false));
    }

    #[test]
    fn test_modifier_apostrophe_normalised() {
        // U+02BC MODIFIER LETTER APOSTROPHE
        let tok = tok();
        assert_eq!(tok.encode("dog\u{02BC}s", false),
                   tok.encode("dog's", false));
    }

    #[test]
    fn test_left_double_quote_normalised() {
        let tok = tok();
        assert_eq!(tok.encode("\u{201C}hello\u{201D}", false),
                   tok.encode("\"hello\"", false));
    }

    // --- UNK behaviour ----------------------------------------------------

    #[test]
    fn test_digit_only_token_is_unk() {
        let tok = tok();
        let ids = tok.encode("42", false);
        assert_eq!(ids, vec![tok.unk_id]);
    }

    #[test]
    fn test_mixed_alpha_digit_is_unk() {
        let tok = tok();
        let ids = tok.encode("abc123", false);
        assert_eq!(ids, vec![tok.unk_id]);
    }

    #[test]
    fn test_non_ascii_word_is_unk() {
        let tok = tok();
        let ids = tok.encode("café", false);
        assert_eq!(ids, vec![tok.unk_id]);
    }

    #[test]
    fn test_emoji_is_unk() {
        let tok = tok();
        let ids = tok.encode("🎉", false);
        assert_eq!(ids, vec![tok.unk_id]);
    }

    #[test]
    fn test_unk_between_words_has_implicit_spaces() {
        // UNK is word-kind so single spaces around it are implicit.
        let tok = tok();
        let ids = tok.encode("dog 42 cat", false);
        assert!(!ids.contains(&tok.space_id));
        assert_eq!(ids, vec![tok.word_ids["dog"], tok.unk_id, tok.word_ids["cat"]]);
    }

    #[test]
    fn test_unk_double_space_produces_explicit_space() {
        let tok = tok();
        let ids = tok.encode("dog  42  cat", false);
        assert_eq!(ids.iter().filter(|&&id| id == tok.space_id).count(), 2);
    }

    #[test]
    fn test_word_not_in_whitelist_is_unk() {
        let tok = tok();
        // "zebra" is not in our test whitelist
        let ids = tok.encode("zebra", false);
        assert_eq!(ids, vec![tok.unk_id]);
    }

    #[test]
    fn test_unk_decode_roundtrip_gives_unk_surface() {
        let tok = tok();
        let decoded = tok.decode(&[tok.unk_id]);
        assert_eq!(decoded, "<|unk|>");
    }

    // --- punctuation token sequences --------------------------------------

    #[test]
    fn test_leading_punct_token_sequence() {
        let tok = tok();
        let ids = tok.encode("(hello)", false);
        assert_eq!(ids, vec![
            tok.word_ids["("],
            tok.word_ids["hello"],
            tok.word_ids[")"],
        ]);
    }

    #[test]
    fn test_question_mark_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "do you know?"), "do you know?");
    }

    #[test]
    fn test_exclamation_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "hello world!"), "hello world!");
    }

    #[test]
    fn test_double_quote_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "he said \"hello\""), "he said \"hello\"");
    }

    #[test]
    fn test_all_punct_tokens_round_trip() {
        let tok = tok();
        for &p in PUNCT_TOKENS {
            let ids = tok.encode(p, false);
            assert_eq!(ids.len(), 1, "punct {p:?} should encode to exactly one token");
            assert_eq!(tok.decode(&ids), p, "punct {p:?} should round-trip");
        }
    }

    #[test]
    fn test_no_space_period_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog.cat"), "dog.cat");
    }

    #[test]
    fn test_space_padded_period_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog . cat"), "dog . cat");
    }

    #[test]
    fn test_multiple_trailing_punct_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "really?!"), "really?!");
    }

    #[test]
    fn test_punct_only_string_roundtrip() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "..."), "...");
    }

    // --- possessive edge cases --------------------------------------------

    #[test]
    fn test_possessive_no_space_before_next_word() {
        // dog's → [dog, 's, toy] — no explicit space between 's and toy
        let tok = tok();
        let ids = tok.encode("dog's toy", false);
        assert!(!ids.contains(&tok.space_id));
    }

    #[test]
    fn test_possessive_double_space_after() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog's  toy"), "dog's  toy");
    }

    #[test]
    fn test_possessive_followed_by_punct() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "dog's toy."), "dog's toy.");
    }

    #[test]
    fn test_possessive_uppercase_normalised() {
        let tok = tok();
        assert_eq!(roundtrip(&tok, "DOG'S toy"), "dog's toy");
    }

    // --- hyphen / dash ----------------------------------------------------

    #[test]
    fn test_hyphenated_both_parts_present() {
        let tok = tok();
        let ids = tok.encode("well-known", false);
        assert!(ids.contains(&tok.word_ids["well"]));
        assert!(ids.contains(&tok.word_ids["known"] ));
        assert!(ids.contains(&tok.unk_id)); // the hyphen
    }

    #[test]
    fn test_en_dash_both_parts_present() {
        let tok = tok();
        let ids = tok.encode("cat\u{2013}dog", false);
        assert!(ids.contains(&tok.word_ids["cat"]));
        assert!(ids.contains(&tok.word_ids["dog"]));
        assert!(ids.contains(&tok.unk_id));
    }

    #[test]
    fn test_em_dash_produces_unk() {
        let tok = tok();
        let ids = tok.encode("cat\u{2014}dog", false);
        assert!(ids.contains(&tok.unk_id));
    }

    #[test]
    fn test_hyphen_only_is_unk() {
        let tok = tok();
        let ids = tok.encode("-", false);
        assert_eq!(ids, vec![tok.unk_id]);
    }

    // --- vocab size and layout --------------------------------------------

    #[test]
    fn test_vocab_size_at_least_n_fixed_plus_contractions_plus_specials() {
        let tok = tok();
        let min = tok.n_fixed as usize + WHOLE_CONTRACTIONS.len() + SPECIAL_TOKENS.len();
        assert!(tok.vocab_size() >= min);
    }

    #[test]
    fn test_n_fixed_equals_punct_plus_space_newline_plus_suffixes() {
        let tok = tok();
        let expected = PUNCT_TOKENS.len() + 1 + 1 + CONTRACTION_SUFFIXES.len();
        assert_eq!(tok.n_fixed as usize, expected);
    }

    #[test]
    fn test_space_token_id_is_valid() {
        let tok = tok();
        assert!(tok.space_id < tok.vocab_size() as u32);
        assert_eq!(tok.id_to_token(tok.space_id), Some(" "));
    }

    #[test]
    fn test_newline_token_id_is_valid() {
        let tok = tok();
        assert!(tok.newline_id < tok.vocab_size() as u32);
        assert_eq!(tok.id_to_token(tok.newline_id), Some("\n"));
    }

    #[test]
    fn test_id_to_token_out_of_range_returns_none() {
        let tok = tok();
        assert_eq!(tok.id_to_token(tok.vocab_size() as u32), None);
        assert_eq!(tok.id_to_token(u32::MAX), None);
    }

    #[test]
    fn test_special_id_unknown_name_returns_none() {
        let tok = tok();
        assert_eq!(tok.special_id("<|nonexistent|>"), None);
    }

    #[test]
    fn test_whitelist_words_have_ids_in_range() {
        let tok = tok();
        for word in ["dog", "cat", "fox", "hello", "world"] {
            let id = tok.word_ids[word];
            assert!(id >= tok.n_fixed, "{word:?} should be above n_fixed");
            assert!(id < tok.vocab_size() as u32);
        }
    }

    #[test]
    fn test_duplicate_whitelist_words_deduplicated() {
        // Building with repeated words should not create duplicate ids.
        let tok = WhitelistTokenizer::from_allowed_words(
            "dog dog cat cat dog".split_whitespace()
        ).unwrap();
        let dog_id = tok.word_ids["dog"];
        let cat_id = tok.word_ids["cat"];
        assert_ne!(dog_id, cat_id);
        // Only one entry for each
        assert_eq!(tok.id_to_word.iter().filter(|s| s.as_str() == "dog").count(), 1);
        assert_eq!(tok.id_to_word.iter().filter(|s| s.as_str() == "cat").count(), 1);
    }

    // --- whitelist filtering at load time ---------------------------------

    #[test]
    fn test_whitelist_ignores_comment_lines() {
        let tok = WhitelistTokenizer::from_allowed_words(
            "# this is a comment\ndog\ncat".lines()
        ).unwrap();
        assert!(tok.word_ids.contains_key("dog"));
        assert!(!tok.word_ids.contains_key("# this is a comment"));
    }

    #[test]
    fn test_whitelist_ignores_blank_lines() {
        let tok = WhitelistTokenizer::from_allowed_words(
            "\n\ndog\n\ncat\n\n".lines()
        ).unwrap();
        assert!(tok.word_ids.contains_key("dog"));
        assert!(tok.word_ids.contains_key("cat"));
    }

    #[test]
    fn test_whitelist_rejects_words_with_digits() {
        let tok = WhitelistTokenizer::from_allowed_words(
            "dog abc123 cat".split_whitespace()
        ).unwrap();
        assert!(!tok.word_ids.contains_key("abc123"));
        assert!(tok.word_ids.contains_key("dog"));
        assert!(tok.word_ids.contains_key("cat"));
    }

    #[test]
    fn test_whitelist_rejects_non_ascii_words() {
        let tok = WhitelistTokenizer::from_allowed_words(
            "dog café cat".split_whitespace()
        ).unwrap();
        assert!(!tok.word_ids.contains_key("café"));
    }

    #[test]
    fn test_whitelist_lowercases_input_words() {
        let tok = WhitelistTokenizer::from_allowed_words(
            "Dog CAT Fox".split_whitespace()
        ).unwrap();
        assert!(tok.word_ids.contains_key("dog"));
        assert!(tok.word_ids.contains_key("cat"));
        assert!(tok.word_ids.contains_key("fox"));
        assert!(!tok.word_ids.contains_key("Dog"));
    }

    // --- encode_batch ---------------------------------------------------------

    #[test]
    fn test_encode_batch_empty_input() {
        let tok = tok();
        assert_eq!(tok.encode_batch(&[], false), Vec::<Vec<u32>>::new());
    }

    #[test]
    fn test_encode_batch_single_item() {
        let tok = tok();
        let batch = tok.encode_batch(&["dog cat"], false);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0], tok.encode("dog cat", false));
    }

    #[test]
    fn test_encode_batch_preserves_order() {
        let tok = tok();
        let texts = vec!["dog", "cat", "fox", "hello", "world"];
        let batch = tok.encode_batch(&texts, false);
        for (i, text) in texts.iter().enumerate() {
            assert_eq!(batch[i], tok.encode(text, false),
                "batch[{i}] should match sequential encode of {text:?}");
        }
    }

    #[test]
    fn test_encode_batch_with_bos() {
        let tok = tok();
        let texts = vec!["dog cat", "hello world"];
        let batch = tok.encode_batch(&texts, true);
        for ids in &batch {
            assert_eq!(ids[0], tok.bos_id);
        }
    }

    // --- decode edge cases ------------------------------------------------

    #[test]
    fn test_decode_space_only() {
        let tok = tok();
        assert_eq!(tok.decode(&[tok.space_id]), " ");
    }

    #[test]
    fn test_decode_newline_only() {
        let tok = tok();
        assert_eq!(tok.decode(&[tok.newline_id]), "\n");
    }

    #[test]
    fn test_decode_multiple_spaces() {
        let tok = tok();
        let ids = vec![tok.space_id, tok.space_id, tok.space_id];
        assert_eq!(tok.decode(&ids), "   ");
    }

    #[test]
    fn test_decode_bos_token() {
        let tok = tok();
        assert_eq!(tok.decode(&[tok.bos_id]), "<|bos|>");
    }

    #[test]
    fn test_decode_unknown_id_silently_skipped() {
        // An id beyond vocab_size has no entry; it should be skipped, not panic.
        let tok = tok();
        let ids = vec![tok.word_ids["dog"], u32::MAX, tok.word_ids["cat"]];
        // Should not panic; "dog" and "cat" with an implicit space between them.
        let result = tok.decode(&ids);
        assert!(result.contains("dog"));
        assert!(result.contains("cat"));
    }

    // --- encode id ordering / no-overlap ----------------------------------

    #[test]
    fn test_all_special_token_ids_are_unique() {
        let tok = tok();
        let mut ids: Vec<u32> = SPECIAL_TOKENS.iter()
            .map(|&s| tok.word_ids[s])
            .collect();
        let before = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), before, "special token ids should all be unique");
    }

    #[test]
    fn test_fixed_and_word_ids_do_not_overlap() {
        let tok = tok();
        for &s in PUNCT_TOKENS.iter().chain(&[SPACE_TOKEN, NEWLINE_TOKEN]).chain(CONTRACTION_SUFFIXES) {
            let id = tok.word_ids[s];
            // Must be below n_fixed — i.e. not sharing an id with any word token.
            assert!(id < tok.n_fixed);
        }
        for &s in WHOLE_CONTRACTIONS {
            assert!(tok.word_ids[s] >= tok.n_fixed);
        }
    }

    // --- pretokenizer stream ----------------------------------------------

    #[test]
    fn test_pretokenize_empty() {
        assert_eq!(pretokenize(""), vec![]);
    }

    #[test]
    fn test_pretokenize_single_word() {
        let items = pretokenize("dog");
        assert_eq!(items, vec![StreamItem::Word("dog".into())]);
    }

    #[test]
    fn test_pretokenize_space_is_sp() {
        let items = pretokenize("dog cat");
        assert!(items.contains(&StreamItem::Sp));
    }

    #[test]
    fn test_pretokenize_newline_is_nl() {
        let items = pretokenize("dog\ncat");
        assert!(items.contains(&StreamItem::Nl));
    }

    #[test]
    fn test_pretokenize_two_spaces_two_sp_items() {
        let items = pretokenize("dog  cat");
        let sp_count = items.iter().filter(|i| **i == StreamItem::Sp).count();
        assert_eq!(sp_count, 2);
    }

    #[test]
    fn test_pretokenize_whole_contraction_is_word_item() {
        let items = pretokenize("don't");
        assert_eq!(items, vec![StreamItem::Word("don't".into())]);
    }

    #[test]
    fn test_pretokenize_unk_is_unk_item() {
        let items = pretokenize("42");
        assert_eq!(items, vec![StreamItem::Unk]);
    }

    #[test]
    fn test_pretokenize_uppercase_word_lowercased() {
        let items = pretokenize("DOG");
        assert_eq!(items, vec![StreamItem::Word("dog".into())]);
    }

    #[test]
    fn test_pretokenize_possessive_splits_to_word_and_fixed() {
        let items = pretokenize("dog's");
        assert_eq!(items, vec![
            StreamItem::Word("dog".into()),
            StreamItem::Fixed("'s".into()),
        ]);
    }
}
