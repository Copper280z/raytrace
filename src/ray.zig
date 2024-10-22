const std = @import("std");
const vector = @import("vector.zig");

pub fn MakeRay(vec_type: type) type {
    const vec = vector.Vector(vec_type);
    return struct {
        pub const Ray = struct {
            origin: vec_type,
            direction: vec_type,
            pub fn point_at_parameter(ray: @This(), t: f32) vec_type {
                return ray.origin + vec.Splat(t) * ray.direction;
            }
        };
    };
}
