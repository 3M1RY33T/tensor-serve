//! Tokenisation and postings construction for the BM25 index.
//!
//! Index building is the one part of the pipeline that is genuinely Python-bound:
//! retrieval spends 81% of its time inside PyTorch and 4% inside FAISS, but
//! building the keyword index is loops over Python strings and dicts, measured
//! at 5.33s per 20,000 chunks.
//!
//! This module mirrors `api/lexical.py` and `api/bm25.py` exactly. It is an
//! optional accelerator: `api/bm25.py` falls back to the pure-Python path when
//! the extension is not built, and the two are checked against each other in
//! `test/test_rust_postings.py`.

use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashMap;

/// Split text into lowercase search terms.
///
/// A term is a run of alphanumerics, optionally joined by `. _ - /` into a
/// compound identifier. Compounds are emitted whole and then split into their
/// parts, so `asyncio.gather` is findable by the full name and by `gather`.
fn tokenize_into(text: &str, out: &mut Vec<String>) {
    let bytes: Vec<char> = text.chars().collect();
    let n = bytes.len();
    let mut i = 0usize;

    let is_word = |c: char| c.is_ascii_alphanumeric();
    let is_join = |c: char| c == '.' || c == '_' || c == '-' || c == '/';

    while i < n {
        let c = bytes[i].to_ascii_lowercase();
        if !is_word(c) {
            i += 1;
            continue;
        }

        // Scan the whole compound: word ( join word )*
        let start = i;
        let mut parts: Vec<(usize, usize)> = Vec::new();
        loop {
            let pstart = i;
            while i < n && is_word(bytes[i].to_ascii_lowercase()) {
                i += 1;
            }
            parts.push((pstart, i));
            if i + 1 < n && is_join(bytes[i]) && is_word(bytes[i + 1].to_ascii_lowercase()) {
                i += 1; // consume the joiner and continue the compound
            } else {
                break;
            }
        }

        let compound: String = bytes[start..i].iter().flat_map(|c| c.to_lowercase()).collect();
        out.push(compound);

        if parts.len() > 1 {
            for (a, b) in parts {
                out.push(bytes[a..b].iter().flat_map(|c| c.to_lowercase()).collect());
            }
        }
    }
}

/// Tokenise one string. Mirrors `api.lexical.tokenize`.
#[pyfunction]
fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    tokenize_into(text, &mut out);
    out
}

/// Build the inverted index for a corpus.
///
/// Returns `(terms, doc_ids, term_freqs, term_offsets, doc_lengths)` as flat
/// lists, in the layout `PostingsBM25` already stores on disk: postings for
/// `terms[i]` occupy `term_offsets[i] .. term_offsets[i + 1]`.
#[pyfunction]
fn build_postings(
    texts: &Bound<'_, PyList>,
) -> PyResult<(Vec<String>, Vec<i64>, Vec<f32>, Vec<i64>, Vec<f32>)> {
    let n_docs = texts.len();
    let mut doc_lengths: Vec<f32> = Vec::with_capacity(n_docs);
    let mut postings: HashMap<String, Vec<(i64, f32)>> = HashMap::new();
    let mut tokens: Vec<String> = Vec::new();

    for (doc_id, item) in texts.iter().enumerate() {
        let text: String = item.extract()?;
        tokens.clear();
        tokenize_into(&text, &mut tokens);
        doc_lengths.push(tokens.len() as f32);

        // Scoped per document: the counts borrow `tokens`, which is reused.
        let mut counts: HashMap<&str, u32> = HashMap::new();
        for t in &tokens {
            *counts.entry(t.as_str()).or_insert(0) += 1;
        }
        for (term, freq) in counts.iter() {
            postings
                .entry((*term).to_string())
                .or_default()
                .push((doc_id as i64, *freq as f32));
        }
    }

    let mut terms: Vec<String> = postings.keys().cloned().collect();
    terms.sort_unstable();

    let mut doc_ids: Vec<i64> = Vec::new();
    let mut term_freqs: Vec<f32> = Vec::new();
    let mut term_offsets: Vec<i64> = Vec::with_capacity(terms.len() + 1);
    term_offsets.push(0);

    for term in &terms {
        let entries = &postings[term];
        for (doc, freq) in entries {
            doc_ids.push(*doc);
            term_freqs.push(*freq);
        }
        term_offsets.push(doc_ids.len() as i64);
    }

    Ok((terms, doc_ids, term_freqs, term_offsets, doc_lengths))
}

#[pymodule]
fn tensor_postings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(build_postings, m)?)?;
    Ok(())
}
