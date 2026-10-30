//! Process memory during a validation run: the macOS physical footprint
//! (`footprint -p`), which on Apple silicon includes Metal/GPU allocations.
//! Sampled on its own thread so measuring never costs a frame; elsewhere the
//! samples stay empty. Whole-process memory, not GI's share of it.
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

const INTERVAL: Duration = Duration::from_secs(2);

pub(super) struct MemorySampler {
    samples: Arc<Mutex<Vec<(f32, u64)>>>,
    stop: Arc<AtomicBool>,
}

impl Default for MemorySampler {
    fn default() -> Self {
        Self {
            samples: Arc::default(),
            stop: Arc::new(AtomicBool::new(true)),
        }
    }
}

impl MemorySampler {
    /// Begin sampling this process (once; later calls do nothing).
    pub fn start(&mut self) {
        if !self.stop.swap(false, Ordering::Relaxed) {
            return;
        }
        let (samples, stop) = (Arc::clone(&self.samples), Arc::clone(&self.stop));
        let started = Instant::now();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Some(bytes) = footprint().as_deref().and_then(parse_footprint) {
                    samples
                        .lock()
                        .unwrap()
                        .push((started.elapsed().as_secs_f32(), bytes));
                }
                std::thread::sleep(INTERVAL);
            }
        });
    }

    /// Stop sampling; the report fields and the final category breakdown.
    pub fn finish(&mut self) -> (serde_json::Value, Option<String>) {
        self.stop.store(true, Ordering::Relaxed);
        let samples = self.samples.lock().unwrap().clone();
        let peak = samples.iter().map(|&(_, bytes)| bytes).max();
        let report = serde_json::json!({
            "source": "footprint -p (physical footprint; includes Metal/GPU allocations on Apple silicon)",
            "peak_mb": peak.map(|bytes| bytes as f64 / 1_048_576.0),
            "last_mb": samples.last().map(|&(_, bytes)| bytes as f64 / 1_048_576.0),
            "samples": samples.iter().map(|&(at, bytes)| [at as f64, bytes as f64 / 1_048_576.0]).collect::<Vec<_>>(),
        });
        (report, footprint())
    }
}

fn footprint() -> Option<String> {
    let output = std::process::Command::new("footprint")
        .arg("-p")
        .arg(std::process::id().to_string())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Bytes from the header line, e.g. `... Footprint: 1234 MB (16384 bytes per page)`.
fn parse_footprint(text: &str) -> Option<u64> {
    let rest = text.split("Footprint:").nth(1)?;
    let mut words = rest.split_whitespace();
    let value: f64 = words.next()?.parse().ok()?;
    let scale = match words.next()? {
        "B" => 1.0,
        "KB" => 1024.0,
        "MB" => 1_048_576.0,
        "GB" => 1_073_741_824.0,
        _ => return None,
    };
    Some((value * scale) as u64)
}

#[cfg(test)]
mod tests {
    use super::parse_footprint;

    #[test]
    fn footprint_header_parses_each_unit() {
        let header = "zsh [52522]: 64-bit    Footprint: 2096 KB (16384 bytes per page)";
        assert_eq!(parse_footprint(header), Some(2096 * 1024));
        assert_eq!(
            parse_footprint("x Footprint: 1.5 GB (y)"),
            Some(1_610_612_736)
        );
        assert_eq!(parse_footprint("no header"), None);
    }
}
