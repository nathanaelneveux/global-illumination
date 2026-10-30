//! Shader reflection for compute kernels built with automatic layouts.
//! wgpu derives a kernel's bind group layouts from the resources its entry
//! point actually uses, so its bind groups must name exactly those
//! bindings. They are read from the shader, not written by hand.
use std::collections::BTreeSet;

pub(crate) struct Reflection {
    module: naga::Module,
    info: naga::valid::ModuleInfo,
}

impl Reflection {
    pub fn new(source: &str) -> Self {
        let module = naga::front::wgsl::parse_str(source).expect("kernel source parses");
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("kernel source validates");
        Self { module, info }
    }

    /// The bindings `entry` uses in groups 0-2, ascending.
    pub fn bindings(&self, entry: &str) -> [Vec<u32>; 3] {
        let index = self
            .module
            .entry_points
            .iter()
            .position(|point| point.name == entry)
            .unwrap_or_else(|| panic!("no entry point {entry}"));
        let usage = self.info.get_entry_point(index);
        let mut groups: [BTreeSet<u32>; 3] = Default::default();
        for (handle, variable) in self.module.global_variables.iter() {
            if let Some(binding) = &variable.binding
                && !usage[handle].is_empty()
            {
                assert!(binding.group < 3, "{entry} binds group {}", binding.group);
                groups[binding.group as usize].insert(binding.binding);
            }
        }
        groups.map(|group| group.into_iter().collect())
    }

    /// A `const` of type u32, by name.
    #[cfg(test)]
    pub fn constant(&self, name: &str) -> u32 {
        let (_, constant) = self
            .module
            .constants
            .iter()
            .find(|(_, constant)| constant.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no const {name}"));
        match self.module.global_expressions[constant.init] {
            naga::Expression::Literal(naga::Literal::U32(value)) => value,
            ref other => panic!("{name} is {other:?}, not a u32 literal"),
        }
    }

    /// A struct's members (name, byte offset) and size.
    #[cfg(test)]
    pub fn layout(&self, name: &str) -> (Vec<(String, u32)>, u32) {
        let ty = self
            .module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no struct {name}"))
            .1;
        match &ty.inner {
            naga::TypeInner::Struct { members, span } => (
                members
                    .iter()
                    .map(|m| (m.name.clone().unwrap_or_default(), m.offset))
                    .collect(),
                *span,
            ),
            other => panic!("{name} is {other:?}"),
        }
    }
}

/// Serialized like an upload, each field's first word must hold its
/// marker (index + 1, as u32 or f32) at the WGSL member's offset, in the
/// same order and under the same name.
#[cfg(test)]
pub(crate) fn assert_mirrors(bytes: &[u8], wgsl: (Vec<(String, u32)>, u32), fields: &[&str]) {
    let (members, span) = wgsl;
    assert_eq!(
        members
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        fields,
        "member names and order"
    );
    assert_eq!(bytes.len(), span as usize, "struct size");
    for (index, (name, offset)) in members.iter().enumerate() {
        let at = *offset as usize;
        let word = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let marker = index as u32 + 1;
        assert!(
            word == marker || f32::from_bits(word) == marker as f32,
            "{name} at {offset} reads {word:#x}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::Reflection;

    #[test]
    fn entry_points_bind_only_what_they_use() {
        let reflection = Reflection::new(
            "@group(0) @binding(0) var<storage, read_write> a: array<u32>;
             @group(0) @binding(3) var<storage, read_write> b: array<u32>;
             @group(2) @binding(7) var<uniform> c: vec4<u32>;
             fn helper() { b[0] = c.x; }
             @compute @workgroup_size(1) fn one() { a[0] = 1u; }
             @compute @workgroup_size(1) fn two() { helper(); }",
        );
        assert_eq!(reflection.bindings("one"), [vec![0], vec![], vec![]]);
        assert_eq!(
            reflection.bindings("two"),
            [vec![3], vec![], vec![7]],
            "through a helper"
        );
    }
}
