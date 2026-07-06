# Vosk second-pass integration (the ensemble)

The pure decoder (`CtcBeamDecoder`) has **no native deps**. The optional `vosk` feature adds
`src/rescore.rs` — the glue that turns a Vosk pass into hotwords and runs the boosted beam.

## Pipeline

```
mic/audio ─┬─► FastConformer (tract int4, streaming)  ─► CTC log-probs  ─┐
           │        (live greedy caption, low latency)                    │
           └─► [utterance end] ─► Vosk (libvosk)  ─► words ─► hotwords ────┴─► shenava-ctc-beam ─► final caption
```

The ensemble is an **utterance-level rescore**: stream greedy CTC for the live low-latency
caption, then on utterance end feed the buffered log-probs + Vosk's words to the beam for the
keyword-accurate final. (Koochik keyword band: 10.5 greedy → 7.55 with Vosk-big.)

## Build with Vosk

1. Get `libvosk` for your target from the [vosk-api releases](https://github.com/alphacep/vosk-api/releases)
   (e.g. `vosk-osx-0.3.45.zip` → `libvosk.dylib` + `vosk_api.h`; `vosk-linux-*`, `vosk-android-*`).
   Put `libvosk.{dylib,so}` on the link path.
2. Build:
   ```sh
   export LIBRARY_PATH=/path/to/vosk-lib          # build-time link
   export DYLD_LIBRARY_PATH=/path/to/vosk-lib     # run-time (macOS); LD_LIBRARY_PATH on Linux
   cargo build --release --features vosk
   ```
3. Ship a Vosk model dir (small-fa ≈ 132 MB, big-fa ≈ 2.6 GB) alongside.

## Use

```rust
use shenava_ctc_beam::{CtcBeamDecoder, rescore};
let model = vosk::Model::new("vosk-model-small-fa-0.5").unwrap();
let dec   = CtcBeamDecoder::new(labels);           // ve_tok_v4 labels, "" = blank
// logprobs: FastConformer CTC log-probs for the finished utterance (from tract)
let text  = rescore::ensemble(&dec, &logprobs, &model, &pcm_i16, 16000.0, 10.0, 80);
```

`pcm_i16` = 16-bit mono PCM at the model's sample rate (16 kHz). Vosk's per-utterance words become
the hotwords; the beam boosts them into the FastConformer's decode.

## Tiers (which Vosk to bundle)

| tier | FastConformer int4 | Vosk | ~download | keyword band |
|---|---|---|---|---|
| stream-only | Pizeh 14 / Rizeh 40 / Koochik 138 MB | — | 14–138 MB | 38 / 18 / 10.5 |
| + small rescore | any above | small-fa 132 MB | +132 MB | 20 / 12 / 8.8 |
| + big rescore | Koochik 138 MB | big-fa 2.6 GB | +2.6 GB | 7.55 |
