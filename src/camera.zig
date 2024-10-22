const std = @import("std");
const vector = @import("vector.zig");
const ray = @import("ray.zig");

pub fn MakeCamTypes(vec_type: type) type {
    const vec = vector.Vector(vec_type);
    const Ray = ray.MakeRay(vec_type).Ray;
    return struct {
        pub const BasicCamera = struct {
            lower_left: vec_type = .{ -2.0, -1.0, -1.0 },
            horizontal: vec_type = .{ 4.0, 0.0, 0.0 },
            vertical: vec_type = .{ 0.0, 2.0, 0.0 },
            origin: vec_type = .{ 0.0, 0.0, 0.0 },

            pub fn get_ray(cam: @This(), u: f32, v: f32) Ray {
                const r = Ray{ .origin = cam.origin, .direction = cam.lower_left + vec.Splat(u) * cam.horizontal + vec.Splat(v) * cam.vertical };
                return r;
            }
        };
    };
}
