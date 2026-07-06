# shenava-ctc-beam

A **hotword-boosted CTC prefix-beam decoder** in pure Rust — a faithful, dependency-free port of
[`pyctcdecode`](https://github.com/kensho-technologies/pyctcdecode)'s beam search (no-LM path)
with per-utterance hotword boosting. **476 KB binary, sub-ms/clip, no ML runtime.**

Built for the on-device [Shenava / VisualEars](https://shenava.app) Persian ASR stack: it's the
"second pass" that turns a streaming FastConformer + a Vosk keyword pass into the **ensemble** —
boosting the meaning-critical words (names/places/domain terms) that greedy CTC drops.

## Why

The Rust ASR ecosystem has streaming acoustic models (e.g. NeMo cache-aware FastConformer in
[tract](https://github.com/sonos/tract)) but no good **hotword-capable CTC beam decoder**. This
fills that gap. On the Shenava double-benchmark it takes Koochik's keyword-band error from
**10.5 (greedy) → 7.55** when fed a Vosk hotword list — matching the Python pipeline exactly.

## Verified parity

On real Koochik CTC log-probs + Vosk-small hotwords (20 golden-6669 clips), the decode is
**20/20 exact-match / 0.00 % word disagreement** vs. `pyctcdecode` (beam 80, hotword_weight 10).

## Use

```rust
use shenava_ctc_beam::{CtcBeamDecoder, Hotwords};

let dec = CtcBeamDecoder::new(labels);          // labels: Vec<String>, "" marks CTC blank
let hw  = Hotwords::new(vosk_words, 10.0);       // per-utterance boost words + weight
let text = dec.decode(&log_probs, &hw, 80, -5.0, -10.0);  // T×V log-probs -> transcript
```

`labels` = one token per class index (ve_tok_v4 BPE-1025 for Shenava; `▁`/U+2581 = word start,
`""` = blank). Feed the FastConformer CTC log-probs for a finished utterance; get the boosted text.

## Algorithm

Faithful pyctcdecode no-LM beam: per-frame candidate pruning (`token_min_logp`), CTC collapse
(blank/repeat), BPE word-boundary handling, `_merge_beams` via numerically-stable log-sum-exp,
hotword scoring = full-word count × weight + partial-prefix credit (`weight·len(part)/len(min
unigram with that prefix)`), `beam_prune_logp` outlier drop, top-N trim. Defaults match pyctcdecode
(beam 100, token_min_logp −5, beam_prune_logp −10, hotword_weight 10).

## Status

Standalone crate — publishable to crates.io. Part of the Shenava on-device ensemble:
`FastConformer (tract int4) → log-probs → [this decoder + Vosk hotwords] → keyword-accurate text`.
