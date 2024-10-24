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
        pub const MovableCamera = struct {
            lower_left: vec_type = .{ -2.0, -1.0, -1.0 },
            horizontal: vec_type = .{ 4.0, 0.0, 0.0 },
            vertical: vec_type = .{ 0.0, 2.0, 0.0 },
            origin: vec_type = .{ 0.0, 0.0, 0.0 },
            pub fn SetPosition(self: *@This(), lookfrom: vec_type, lookat: vec_type, v_up: vec_type, vfov: f32, aspect: f32) void {
                self.origin = lookfrom;
                const theta = vfov * std.math.pi / 180.0;
                const half_height = std.math.tan(theta / 2);
                const half_width = aspect * half_height;
                const w = vec.normalize(lookfrom - lookat);
                const u = vec.normalize(vec.cross(v_up, w));
                const v = vec.cross(w, u);
                // self.lower_left = .{-half_width, -half_height, -1.0};
                self.lower_left = self.origin - vec.Splat(half_width) * u - vec.Splat(half_height) * v - w;
                self.horizontal = vec.Splat(2 * half_width) * u;
                self.vertical = vec.Splat(2 * half_height) * v;
            }
            pub fn get_ray(cam: @This(), s: f32, t: f32) Ray {
                const r = Ray{ .origin = cam.origin, .direction = cam.lower_left + vec.Splat(s) * cam.horizontal + vec.Splat(t) * cam.vertical - cam.origin };
                return r;
            }
        };
    };
}
