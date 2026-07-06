//! Tier manager — reads `shenava_deploy_registry.json`, detects device capability, picks the
//! most keyword-accurate package that fits, and prints (or executes) the download plan.
//!
//! The "download-as-needed, as capable as possible" switcher for the Shenava on-device stack.
//!
//! ```sh
//! cargo run --example tier_manager                        # auto-detect this machine
//! cargo run --example tier_manager -- --ram-mb 96 --storage-mb 200   # simulate a 32-bit TV
//! cargo run --example tier_manager -- --download          # actually fetch the chosen package
//! ```
//! No new deps (serde_json only); RAM auto-detect via `sysctl`/`/proc/meminfo`; download via `curl`.

use serde::Deserialize;
use std::collections::HashMap;
use std::process::Command;

#[derive(Deserialize)]
struct Registry {
    components: HashMap<String, Component>,
    packages: Vec<Package>,
}
#[derive(Deserialize)]
struct Component {
    url: String,
    size_mb: f64,
}
#[derive(Deserialize)]
struct Package {
    id: String,
    components: Vec<String>,
    total_mb: u64,
    mode: String,
    ess_kw: f64,
    decode_ms: u64,
    requires: Requires,
    #[serde(default)]
    pareto_optimal: bool,
    #[serde(default)]
    dev_only: bool,
    #[serde(default)]
    targets: Vec<String>,
}
#[derive(Deserialize, Default)]
struct Requires {
    min_ram_mb: u64,
    #[serde(default)]
    min_storage_mb: u64,
}

/// Best-effort physical RAM in MB (std + platform tools; falls back to 2048).
fn detect_ram_mb() -> u64 {
    #[cfg(target_os = "macos")]
    if let Ok(o) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
        if let Ok(bytes) = String::from_utf8_lossy(&o.stdout).trim().parse::<u64>() {
            return bytes / 1024 / 1024;
        }
    }
    #[cfg(target_os = "linux")]
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for line in s.lines() {
            if let Some(kb) = line.strip_prefix("MemTotal:") {
                if let Ok(kb) = kb.trim().trim_end_matches(" kB").trim().parse::<u64>() {
                    return kb / 1024;
                }
            }
        }
    }
    2048
}

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1).cloned())
}
fn flag(name: &str) -> bool {
    std::env::args().any(|x| x == name)
}

/// hf://OWNER/NAME/FILE -> https HF resolve URL; pass others (e.g. vosk .zip) through.
fn resolve(url: &str) -> String {
    if let Some(rest) = url.strip_prefix("hf://") {
        let p: Vec<&str> = rest.splitn(3, '/').collect();
        if p.len() == 3 {
            return format!("https://huggingface.co/{}/{}/resolve/main/{}", p[0], p[1], p[2]);
        }
    }
    url.to_string()
}

fn main() {
    let path = arg("--registry").unwrap_or_else(|| "shenava_deploy_registry.json".into());
    let reg: Registry = serde_json::from_str(&std::fs::read_to_string(&path).expect("read registry"))
        .expect("parse registry");

    let ram = arg("--ram-mb").and_then(|s| s.parse().ok()).unwrap_or_else(detect_ram_mb);
    let storage = arg("--storage-mb").and_then(|s| s.parse().ok()).unwrap_or(u64::MAX / 2);
    let budget = arg("--budget-mb").and_then(|s| s.parse().ok()).unwrap_or(u64::MAX / 2);

    let dev_mode = flag("--dev");
    println!("device: {} MB RAM · {} storage · budget {}{}",
             ram,
             if storage == u64::MAX / 2 { "∞".into() } else { format!("{} MB", storage) },
             if budget == u64::MAX / 2 { "∞".into() } else { format!("{} MB", budget) },
             if dev_mode { "   [DEV MODE — all models unlocked]" } else { "" });

    // Production policy: Rizeh/Pizeh never ship naked; only Koochik solo. dev_only tiers hidden
    // unless dev mode is on.
    let fits = |p: &&Package| {
        (dev_mode || !p.dev_only)
            && p.requires.min_ram_mb <= ram
            && p.total_mb <= budget
            && p.requires.min_storage_mb <= storage
    };
    let candidates: Vec<&Package> = reg.packages.iter().filter(fits).collect();

    if candidates.is_empty() {
        println!("\n⚠ no {}package fits this device.", if dev_mode { "" } else { "production " });
        if !dev_mode {
            println!("  production floor: needs ~300 MB RAM to hold Vosk-small — Rizeh/Pizeh never ship naked,");
            println!("  and only Koochik may go solo. Run with --dev to unlock the naked small models.");
        }
        return;
    }
    // prefer Pareto-optimal, then lowest keyword-band error.
    let p = *candidates
        .iter()
        .filter(|p| p.pareto_optimal)
        .min_by(|a, b| a.ess_kw.partial_cmp(&b.ess_kw).unwrap())
        .or_else(|| candidates.iter().min_by(|a, b| a.ess_kw.partial_cmp(&b.ess_kw).unwrap()))
        .unwrap();

    println!("\n▶ selected: {}  ({} MB, {}, keyword-band {:.2}, {} ms/utt{})",
             p.id, p.total_mb, p.mode, p.ess_kw, p.decode_ms,
             if p.pareto_optimal { "" } else { "  ⚠ not Pareto-optimal" });
    if !p.targets.is_empty() {
        println!("  targets: {}", p.targets.join(", "));
    }
    println!("\n  download plan:");
    let mut cache = std::path::PathBuf::from(arg("--cache").unwrap_or_else(|| "shenava_models".into()));
    cache.push(&p.id);
    for cname in &p.components {
        let c = &reg.components[cname];
        let url = resolve(&c.url);
        println!("    - {:<16} {:>6.0} MB   {}", cname, c.size_mb, url);
    }
    println!("  -> {}", cache.display());

    if flag("--download") {
        std::fs::create_dir_all(&cache).ok();
        for cname in &p.components {
            let url = resolve(&reg.components[cname].url);
            let fname = url.rsplit('/').next().unwrap_or("file");
            let out = cache.join(fname);
            println!("\n  fetching {cname} …");
            let ok = Command::new("curl")
                .args(["-L", "--fail", "-o", out.to_str().unwrap(), &url])
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if ok && fname.ends_with(".zip") {
                Command::new("unzip").args(["-oq", out.to_str().unwrap(), "-d", cache.to_str().unwrap()]).status().ok();
            }
            println!("    {}", if ok { "done" } else { "FAILED" });
        }
        println!("\n✓ {} ready in {}", p.id, cache.display());
    } else {
        println!("\n  (dry-run — pass --download to fetch)");
    }
}
