//! CLI for parity testing: reads a JSON `{labels, hotword_weight?, beam_width?, clips:[{id,hotwords,logprobs}]}`
//! and prints `id\thyp` per clip. Used to validate against the Python pyctcdecode reference.
use serde::Deserialize;
use shenava_ctc_beam::{CtcBeamDecoder, Hotwords};

#[derive(Deserialize)]
struct Clip {
    id: String,
    hotwords: Vec<String>,
    logprobs: Vec<Vec<f32>>,
}
#[derive(Deserialize)]
struct Input {
    labels: Vec<String>,
    clips: Vec<Clip>,
    #[serde(default = "default_weight")]
    hotword_weight: f32,
    #[serde(default = "default_beam")]
    beam_width: usize,
}
fn default_weight() -> f32 {
    10.0
}
fn default_beam() -> usize {
    100
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: shenava-decode input.json");
    let data = std::fs::read_to_string(&path).expect("read input");
    let inp: Input = serde_json::from_str(&data).expect("parse input");
    let dec = CtcBeamDecoder::new(inp.labels);
    for c in &inp.clips {
        let hw = Hotwords::new(c.hotwords.iter().cloned(), inp.hotword_weight);
        let text = dec.decode(&c.logprobs, &hw, inp.beam_width, -5.0, -10.0);
        println!("{}\t{}", c.id, text);
    }
}
