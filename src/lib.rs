//! Hotword-boosted CTC prefix-beam decoder — a faithful, dependency-free Rust port of the
//! `pyctcdecode` beam search (no-LM path) with per-utterance hotword boosting.
//!
//! Intended second pass for the Shenava on-device ASR stack: feed the FastConformer CTC
//! log-probs for a finished utterance plus a hotword list (e.g. from a Vosk pass), and get a
//! keyword-boosted transcript. Matches pyctcdecode's decode to (near) parity.
//!
//! Defaults mirror pyctcdecode: beam_width 100, token_min_logp -5.0, beam_prune_logp -10.0,
//! hotword_weight 10.0. BPE word-start marker is `▁` (U+2581); blank = the empty-string label.

use std::collections::HashMap;

pub mod rescore;

const BPE: char = '\u{2581}'; // ▁

/// Decoder holds the vocabulary (label per class index) and derived blank id.
pub struct CtcBeamDecoder {
    labels: Vec<String>,
    blank_id: usize,
}

/// Per-utterance hotwords (unigram words to boost).
pub struct Hotwords {
    unigrams: Vec<String>,
    set: std::collections::HashSet<String>,
    weight: f32,
}

impl Hotwords {
    /// Build from an iterator of words/phrases (phrases are split into unigrams).
    pub fn new<I: IntoIterator<Item = String>>(words: I, weight: f32) -> Self {
        let mut unigrams = Vec::new();
        let mut set = std::collections::HashSet::new();
        for w in words {
            for u in w.split_whitespace() {
                if !u.is_empty() {
                    unigrams.push(u.to_string());
                    set.insert(u.to_string());
                }
            }
        }
        Hotwords { unigrams, set, weight }
    }
    #[inline]
    fn is_word(&self, w: &str) -> bool {
        self.set.contains(w)
    }
    /// Partial-token credit: weight * len(word_part) / len(shortest unigram with that prefix).
    fn partial(&self, word_part: &str) -> f32 {
        if word_part.is_empty() || self.unigrams.is_empty() {
            return 0.0;
        }
        let wp = word_part.chars().count();
        let mut min_len = usize::MAX;
        for u in &self.unigrams {
            if u.starts_with(word_part) {
                let l = u.chars().count();
                if l < min_len {
                    min_len = l;
                }
            }
        }
        if min_len == usize::MAX {
            0.0
        } else {
            self.weight * (wp as f32) / (min_len as f32)
        }
    }
}

#[derive(Clone)]
struct Beam {
    text: String,      // completed words, space-joined
    word_part: String, // current in-progress word
    last_idx: i32,     // last emitted class idx (-1 = start/None); blank_id after a blank
    hw_count: u32,     // # hotword words already folded into `text`
    logit: f32,        // accumulated CTC log-prob (no hotword term)
}

#[inline]
fn log_sum_exp(a: f32, b: f32) -> f32 {
    if a >= b {
        a + (1.0 + (b - a).exp()).ln()
    } else {
        b + (1.0 + (a - b).exp()).ln()
    }
}

impl CtcBeamDecoder {
    /// `labels[i]` = the token for class i; the empty-string label marks CTC blank.
    pub fn new(labels: Vec<String>) -> Self {
        let blank_id = labels
            .iter()
            .position(|s| s.is_empty())
            .unwrap_or(labels.len() - 1);
        CtcBeamDecoder { labels, blank_id }
    }

    /// Decode `log_probs` (T rows × V log-probabilities) with per-utterance `hotwords`.
    pub fn decode(
        &self,
        log_probs: &[Vec<f32>],
        hotwords: &Hotwords,
        beam_width: usize,
        token_min_logp: f32,
        beam_prune_logp: f32,
    ) -> String {
        let mut beams: Vec<Beam> = vec![Beam {
            text: String::new(),
            word_part: String::new(),
            last_idx: -1,
            hw_count: 0,
            logit: 0.0,
        }];

        for col in log_probs {
            // candidate classes: those >= token_min_logp, plus the argmax
            let mut argmax = 0usize;
            let mut argmax_v = f32::NEG_INFINITY;
            let mut cands: Vec<usize> = Vec::new();
            for (i, &v) in col.iter().enumerate() {
                if v > argmax_v {
                    argmax_v = v;
                    argmax = i;
                }
                if v >= token_min_logp {
                    cands.push(i);
                }
            }
            if !cands.contains(&argmax) {
                cands.push(argmax);
            }

            // expand
            let mut merged: HashMap<(String, String, i32), Beam> = HashMap::new();
            let mut push = |b: Beam| {
                let key = (b.text.clone(), b.word_part.clone(), b.last_idx);
                merged
                    .entry(key)
                    .and_modify(|e| e.logit = log_sum_exp(e.logit, b.logit))
                    .or_insert(b);
            };
            for &idx in &cands {
                let p = col[idx];
                let is_blank = idx == self.blank_id;
                let tok = &self.labels[idx];
                for beam in &beams {
                    if is_blank || idx as i32 == beam.last_idx {
                        // blank or repeat -> stay (CTC collapse)
                        push(Beam {
                            text: beam.text.clone(),
                            word_part: beam.word_part.clone(),
                            last_idx: idx as i32,
                            hw_count: beam.hw_count,
                            logit: beam.logit + p,
                        });
                    } else if tok.starts_with(BPE) {
                        // word boundary: fold current word_part into text, start new word
                        let (text, hw) = self.fold(&beam.text, &beam.word_part, beam.hw_count, hotwords);
                        let clean: String = tok.trim_start_matches(BPE).trim_end_matches(BPE).to_string();
                        push(Beam {
                            text,
                            word_part: clean,
                            last_idx: idx as i32,
                            hw_count: hw,
                            logit: beam.logit + p,
                        });
                    } else {
                        // continue current word
                        let mut wp = beam.word_part.clone();
                        wp.push_str(tok);
                        push(Beam {
                            text: beam.text.clone(),
                            word_part: wp,
                            last_idx: idx as i32,
                            hw_count: beam.hw_count,
                            logit: beam.logit + p,
                        });
                    }
                }
            }

            // score (logit + hotword full-word + hotword partial), prune, trim
            let mut scored: Vec<(f32, Beam)> = merged
                .into_values()
                .map(|b| {
                    let s = b.logit
                        + hotwords.weight * b.hw_count as f32
                        + hotwords.partial(&b.word_part);
                    (s, b)
                })
                .collect();
            let max_s = scored.iter().map(|x| x.0).fold(f32::NEG_INFINITY, f32::max);
            scored.retain(|x| x.0 >= max_s + beam_prune_logp);
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(beam_width);
            beams = scored.into_iter().map(|x| x.1).collect();
        }

        // finalize: fold trailing word_part, pick best by hotword-augmented score
        let mut best_text = String::new();
        let mut best_s = f32::NEG_INFINITY;
        for beam in &beams {
            let (text, hw) = self.fold(&beam.text, &beam.word_part, beam.hw_count, hotwords);
            let s = beam.logit + hotwords.weight * hw as f32;
            if s > best_s {
                best_s = s;
                best_text = text;
            }
        }
        best_text
    }

    #[inline]
    fn fold(&self, text: &str, word_part: &str, hw_count: u32, hw: &Hotwords) -> (String, u32) {
        if word_part.is_empty() {
            return (text.to_string(), hw_count);
        }
        let new_text = if text.is_empty() {
            word_part.to_string()
        } else {
            format!("{} {}", text, word_part)
        };
        let new_hw = hw_count + if hw.is_word(word_part) { 1 } else { 0 };
        (new_text, new_hw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hotword_flips_close_call() {
        // "▁cat" (−0.7) vs "▁cap" (−0.8): acoustic prefers "cat"; hotword "cap" should flip it.
        let labels = vec!["▁cat".into(), "▁cap".into(), "".into()];
        let dec = CtcBeamDecoder::new(labels);
        let lp = vec![vec![-0.7f32, -0.8, -5.0]];
        let none = Hotwords::new(Vec::<String>::new(), 10.0);
        assert_eq!(dec.decode(&lp, &none, 10, -5.0, -10.0), "cat");
        let hw = Hotwords::new(vec!["cap".into()], 10.0);
        assert_eq!(dec.decode(&lp, &hw, 10, -5.0, -10.0), "cap");
    }

    #[test]
    fn greedy_two_words_with_blank_reset() {
        // ▁one · blank · ▁two · blank  ->  "one two"
        let labels = vec!["▁one".into(), "▁two".into(), "".into()];
        let dec = CtcBeamDecoder::new(labels);
        let none = Hotwords::new(Vec::<String>::new(), 10.0);
        let lp = vec![
            vec![-0.1f32, -5.0, -5.0],
            vec![-5.0, -5.0, -0.1],
            vec![-5.0, -0.1, -5.0],
            vec![-5.0, -5.0, -0.1],
        ];
        assert_eq!(dec.decode(&lp, &none, 10, -5.0, -10.0), "one two");
    }

    #[test]
    fn repeat_collapses_to_single_word() {
        // ▁hi · ▁hi(repeat, no blank between) -> "hi" (CTC collapse, not "hi hi")
        let labels = vec!["▁hi".into(), "".into()];
        let dec = CtcBeamDecoder::new(labels);
        let none = Hotwords::new(Vec::<String>::new(), 10.0);
        let lp = vec![vec![-0.1f32, -5.0], vec![-0.1, -5.0]];
        assert_eq!(dec.decode(&lp, &none, 10, -5.0, -10.0), "hi");
    }
}
