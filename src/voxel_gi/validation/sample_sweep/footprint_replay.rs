//! Replay saved native prefixes through the production filter oracle. This
//! creates renderer inputs, not new transport evidence or timing measurements.
use super::*;
use serde_json::{Value, json};
use std::{io::Write, path::Path};

fn write_frame(output: &Path, name: &str, values: &Means) {
    let mut file = std::io::BufWriter::new(
        std::fs::File::create_new(output.join(format!("{name}.bin"))).unwrap(),
    );
    for value in values {
        for channel in [value[0] as f32, value[1] as f32, value[2] as f32, 1.0] {
            assert!(channel.is_finite() && channel >= 0.0);
            file.write_all(&channel.to_le_bytes()).unwrap();
        }
    }
    file.flush().unwrap();
}

#[test]
#[ignore = "explicit saved native prefix source and new output directory required"]
fn export_footprint_replay() {
    let source = std::path::PathBuf::from(std::env::var("GI_FOOTPRINT_SOURCE").unwrap());
    let output = std::path::PathBuf::from(std::env::var("GI_FOOTPRINT_OUTPUT").unwrap());
    let read = |name| -> Value {
        serde_json::from_slice(&std::fs::read(source.join(name)).unwrap()).unwrap()
    };
    let reference = read("cave-reference.json");
    let sweep = read("cave-sample-sweep.json");
    let audit = read("cave-geometry-audit.json");
    assert_eq!(reference["qualification"]["qualified"], true);
    assert_eq!(read("cave-reference-status.json")["status"], "completed");
    assert_eq!(audit["unknown_voxels"], 0);
    assert_eq!(audit["mismatched_voxels"], 0);
    for key in [
        "schema_version",
        "face_count",
        "origin",
        "scene_generation",
        "transport",
    ] {
        assert_eq!(reference[key], sweep[key], "source mismatch: {key}");
    }
    assert_eq!(sweep["sampler"], "independent-paths");
    let faces = reference["faces"].as_array().unwrap();
    let sites: Vec<(IVec3, usize)> = faces
        .iter()
        .map(|row| {
            (
                IVec3::from_array(serde_json::from_value(row["voxel"].clone()).unwrap()),
                row["face"].as_u64().unwrap() as usize,
            )
        })
        .collect();
    assert_eq!(
        sites.len(),
        reference["face_count"].as_u64().unwrap() as usize
    );
    let truth: Means = faces
        .iter()
        .map(|row| serde_json::from_value(row["mean"].clone()).unwrap())
        .collect();
    let windows: Vec<Window> = serde_json::from_value(sweep["windows"].clone()).unwrap();
    assert_eq!(windows.len(), 4);
    // Refuse reuse/overwrite: an interrupted export must get a fresh directory.
    std::fs::create_dir(&output).unwrap();
    write_frame(&output, "cave-reference", &truth);
    let mut frames = vec![json!({"name":"cave-reference"})];
    let mut support = Vec::new();
    for (window, data) in windows.iter().enumerate() {
        for prefix in &data.prefixes {
            assert_eq!(prefix.means.len(), sites.len());
            assert_eq!(prefix.second_moments.as_ref().unwrap().len(), sites.len());
            if ![4, 16, 64, 128].contains(&prefix.paths) {
                continue;
            }
            for cap in [0, 32, 64, 128] {
                let label = if cap == 0 {
                    "evidence2048".into()
                } else {
                    format!("evidence2048cap{cap}")
                };
                let name = format!("{label}-{}-window-{window}", prefix.paths);
                let (values, mut report) = evidence_values(&sites, prefix, 2048, cap);
                report["window"] = json!(window);
                support.push(report);
                write_frame(&output, &name, &values);
                frames.push(json!({"name":name}));
            }
        }
    }
    std::fs::write(output.join("manifest.json"), serde_json::to_vec_pretty(&json!({
        "schema":1,"fixture":"cave-torch", "source":source, "source_transport":sweep["transport"],
        "source_generation":sweep["scene_generation"], "sites":sites.iter().map(|(v,f)|json!([v.to_array(),f])).collect::<Vec<_>>(),
        "frames":frames,"support":support,
        "note":"Saved qualified native prefixes; presentation replay only. Check reference/control render equivalence before using original image gates."
    })).unwrap()).unwrap();
    println!("Exported {} frames to {}", frames.len(), output.display());
}
