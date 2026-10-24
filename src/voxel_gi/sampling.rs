//! Stable per-face sample stream identities and the shared integer hash.
//! The quasi-random sequences themselves live in the GPU path shader.
use bevy::prelude::*;

pub(crate) fn pcg(value: u32) -> u32 {
    let state = value.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

pub(crate) fn stream_id(voxel: IVec3, face: u32) -> u32 {
    pcg(pcg(pcg(voxel.x as u32) ^ voxel.y as u32) ^ voxel.z as u32) ^ pcg(face.wrapping_add(1))
}
