//! Optional Vosk second-pass glue (enable with `--features vosk`).
//!
//! Turns a finished utterance's PCM into a hotword list (via a Vosk model), then boosts the
//! FastConformer CTC log-probs with the [`crate::CtcBeamDecoder`]. This is the full Shenava
//! ensemble rescore in one call:
//!
//! ```ignore
//! let model = vosk::Model::new("vosk-model-small-fa-0.5").unwrap();
//! let dec   = shenava_ctc_beam::CtcBeamDecoder::new(labels);
//! // `logprobs` = FastConformer CTC log-probs for the utterance (from tract int4).
//! let text = shenava_ctc_beam::rescore::ensemble(&dec, &logprobs, &model, &pcm_i16, 16000.0, 10.0, 80);
//! ```
//!
//! Requires `libvosk` on the link path (see VOSK_INTEGRATION.md). The pure decoder in
//! [`crate`] has no native deps; only this module does.

#[cfg(feature = "vosk")]
use crate::{CtcBeamDecoder, Hotwords};

/// Run Vosk over 16-bit mono PCM and return its recognized words (deduped, ≥3 chars) as hotwords.
#[cfg(feature = "vosk")]
pub fn vosk_hotwords(model: &vosk::Model, pcm_i16: &[i16], sample_rate: f32) -> Vec<String> {
    let mut rec = vosk::Recognizer::new(model, sample_rate).expect("create Vosk recognizer");
    rec.set_words(true);
    rec.accept_waveform(pcm_i16).ok();
    let text = rec
        .final_result()
        .single()
        .map(|r| r.text.to_string())
        .unwrap_or_default();
    let mut seen = std::collections::HashSet::new();
    text.split_whitespace()
        .filter(|w| w.chars().count() >= 3 && seen.insert(w.to_string()))
        .map(|w| w.to_string())
        .collect()
}

/// Full ensemble second pass: Vosk words → hotwords → hotword-boosted CTC beam over the log-probs.
#[cfg(feature = "vosk")]
#[allow(clippy::too_many_arguments)]
pub fn ensemble(
    dec: &CtcBeamDecoder,
    logprobs: &[Vec<f32>],
    model: &vosk::Model,
    pcm_i16: &[i16],
    sample_rate: f32,
    hotword_weight: f32,
    beam_width: usize,
) -> String {
    let words = vosk_hotwords(model, pcm_i16, sample_rate);
    let hw = Hotwords::new(words, hotword_weight);
    dec.decode(logprobs, &hw, beam_width, -5.0, -10.0)
}
