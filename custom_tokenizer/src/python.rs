/*!
python.rs — PyO3 bindings for WhitelistTokenizer.

Usage from Python:
    import whitelist_tokenizer_rs as wt
    tok = wt.WhitelistTokenizer("path/to/dir/")   # dir must contain word_whitelist
    ids   = tok.encode("The quick brown fox", prepend_bos=True)
    text  = tok.decode(ids)
    batch = tok.encode_batch(["hello world", "foo bar"], prepend_bos=True)
    tok.id_to_token(42)   # -> "some"  or  None
*/

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::path::Path;
use std::sync::Arc;

use crate::WhitelistTokenizer;

#[pyclass(name = "WhitelistTokenizer")]
pub struct WhitelistTokenizerPy {
    inner: Arc<WhitelistTokenizer>,
}

#[pymethods]
impl WhitelistTokenizerPy {
    /// Load from a directory containing word_whitelist.
    #[new]
    pub fn new(directory: &str) -> PyResult<Self> {
        let tok = WhitelistTokenizer::from_directory(Path::new(directory))
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(WhitelistTokenizerPy { inner: Arc::new(tok) })
    }

    /// encode(text, prepend_bos=False) -> list[int]
    #[pyo3(signature = (text, prepend_bos=false))]
    pub fn encode(&self, text: &str, prepend_bos: bool) -> Vec<u32> {
        self.inner.encode(text, prepend_bos)
    }

    /// encode_batch(texts, prepend_bos=False) -> list[list[int]]
    #[pyo3(signature = (texts, prepend_bos=false))]
    pub fn encode_batch(&self, texts: Vec<String>, prepend_bos: bool) -> Vec<Vec<u32>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner.encode_batch(&refs, prepend_bos)
    }

    /// decode(ids) -> str
    pub fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    /// id_to_token(id) -> str | None
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(|s| s.to_string())
    }

    pub fn vocab_size(&self) -> usize { self.inner.vocab_size() }
    pub fn bos_id(&self)     -> u32   { self.inner.bos_id() }
    pub fn mask_id(&self)    -> u32   { self.inner.mask_id() }
    pub fn unk_id(&self)     -> u32   { self.inner.unk_id() }
    pub fn space_id(&self)   -> u32   { self.inner.space_id() }
    pub fn newline_id(&self) -> u32   { self.inner.newline_id() }

    pub fn special_id(&self, name: &str) -> Option<u32> {
        self.inner.special_id(name)
    }
}

#[pymodule]
pub fn whitelist_tokenizer_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WhitelistTokenizerPy>()?;
    Ok(())
}
