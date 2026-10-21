// Shared distance/flux retention policy for resident and view-owned receivers.
// The including consumer supplies the packed `events` buffer.
const NO_LIMIT: u32 = 0xffffffffu;
const PI: f32 = 3.14159265358979;
const REGION_EDGE: i32 = 16;

// H(d) = round((d / 12)^4) from squared distance, exactly in integers
// (retention.rs `limit`). At d^2 >= 192^2 it reaches the history ceiling.
fn curve_limit(d2: u32) -> u32 {
    if d2 >= 36864u { return NO_LIMIT; }
    return (d2 * d2 + 10368u) / 20736u;
}

// geometry.rs Bounds::distance_squared, saturating each axis at 4096.
fn box_distance2(lo: vec3<i32>, hi: vec3<i32>, p: vec3<i32>) -> u32 {
    let separation = vec3<u32>(min(max(max(lo - p, p - hi), vec3(0)), vec3(4096)));
    return dot(separation, separation);
}

fn load_height(top: i32, y: i32) -> f32 {
    return f32(clamp(top + 1 - y, 0, REGION_EDGE));
}


struct RetentionChange {
    limit: u32, change: f32, measured: bool, woken: bool,
    urgent: bool, urgent_if_binding: bool,
}
fn resolve_retention(p: vec3<i32>, event_count: u32, reach: u32, measured_mode: bool,
    urgent_measured: u32, urgent_curve: u32) -> RetentionChange {
    var cap = NO_LIMIT;
    var change = 0.0;
    var known = true;
    // Source changes cap every face (sources.rs history_limit): the base cap
    // outside the changed lights' spheres, 4 within. Any event in reach, or
    // any source change, revokes stability: the face is due for an audit.
    var source_cap = NO_LIMIT;
    var woken = false;
    // Player edits (bounds flagged bit 3): faces within the urgent radius
    // whose history the change binds (or that were dark relative to it) are
    // urgent; with unknown flux or the curve rule, every face in the smaller
    // radius is (publication.rs used to decide this from CPU statistics).
    var urgent_if_binding = false;
    var urgent = false;
    for (var e = 0u; e < event_count; e += 1u) {
        let h0 = events[e * 3u];
        let h1 = events[e * 3u + 1u];
        let h2 = events[e * 3u + 2u];
        if (h0.w & 2u) != 0u {
            var limit = h2.x;
            for (var b = 0u; b < h2.w; b += 1u) {
                let sphere = bitcast<vec4<f32>>(events[event_count * 3u + (h2.z + b) * 2u]);
                let d = vec3<f32>(p) - sphere.xyz;
                if dot(d, d) <= sphere.w { limit = min(limit, 4u); }
            }
            source_cap = min(source_cap, limit);
            woken = true;
            continue;
        }
        let outside = box_distance2(bitcast<vec3<i32>>(h0.xyz), bitcast<vec3<i32>>(h1.xyz), p);
        if outside > reach { continue; }
        woken = true;
        var exact = known;
        var nearest = NO_LIMIT;
        for (var b = 0u; b < h2.w; b += 1u) {
            let record = event_count * 3u + (h2.z + b) * 2u;
            let b0 = events[record];
            let b1 = events[record + 1u];
            let d = box_distance2(bitcast<vec3<i32>>(b0.xyz), bitcast<vec3<i32>>(b1.xyz), p);
            if d > reach { continue; }
            nearest = min(nearest, d);
            let edit_kind = b1.w & 7u;
            if (b1.w & 8u) != 0u && (edit_kind == 0u || edit_kind == 3u) {
                if edit_kind == 3u && measured_mode {
                    urgent_if_binding = urgent_if_binding || d <= urgent_measured;
                } else {
                    urgent = urgent || d <= urgent_curve;
                }
            }
            if exact {
                let kind = b1.w & 7u;
                if kind == 0u {
                    known = false;
                    exact = false;
                } else if kind != 1u {
                    let area = select(1.0, load_height(bitcast<i32>(b1.w) >> 3u, p.y), kind == 2u);
                    change += bitcast<f32>(b0.w) * area / (PI * f32(max(d, 1u)));
                }
            }
        }
        if nearest != NO_LIMIT { cap = min(cap, curve_limit(nearest)); }
    }
    let measured = measured_mode && known;
    let limit = min(source_cap, select(cap, NO_LIMIT, measured));
    return RetentionChange(limit, change, measured, woken, urgent, urgent_if_binding);
}
