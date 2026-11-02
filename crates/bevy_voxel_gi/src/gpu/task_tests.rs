//! Task serialization oracle. The generic ShaderType encoder is retained only
//! here, not as a production mode.
use super::*;

pub(super) fn legacy_task_bytes(work: &TraceWork) -> Vec<u8> {
    let tasks: Vec<_> = work
        .faces
        .iter()
        .map(|face| GpuFaceTask {
            voxel: face.voxel,
            face: face.face as u32,
            sequence: UVec4::new(face.sequence, 0, 0, 0),
        })
        .collect();
    ShaderBuffer::from(tasks).data.unwrap()
}

#[test]
fn task_encoding_matches_shader_layout_and_clears_reused_tails() {
    let mut probes = vec![SurfaceProbe::new(IVec3::new(i32::MIN, -17, i32::MAX), 63)];
    probes[0].stats_mut().sequence = [0, 1, u32::MAX, 65536, 0x8000_0000, 12345];
    let work = TraceWork::from_probes(&probes);
    let mut bytes = Vec::new();
    for work in [&work, &TraceWork::default(), &work] {
        encode_tasks(&mut bytes, work);
        assert_eq!(bytes.len(), work.faces.len() * TASK_BYTES);
        if bytes.is_empty() {
            // Encase pads an empty runtime array to one element. Empty work
            // returns before task upload, so no dummy element is needed here.
            assert_eq!(legacy_task_bytes(work), vec![0; TASK_BYTES]);
        } else {
            assert_eq!(bytes, legacy_task_bytes(work));
        }
    }
}
