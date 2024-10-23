const std = @import("std");
const ray = @import("ray.zig");

const vec_type = @Vector(3, f32);
// const vec = vector.Vector(vec_type);
const Ray = ray.MakeRay(vec_type).Ray;

// const Material = union(enum) {
//     metal: Metal,
//     lambertian: Lambertian,

// };

// const Metal = struct {
//     albedo: vec_type,
//     pub fn scatter(r_in: Ray, )
// };

// const Lambertian = struct {

// };
