const std = @import("std");
const vector = @import("vector.zig");
const ray = @import("ray.zig");
const cameras = @import("camera.zig");
const img_io = @import("image.zig");

pub const std_options = .{
    .log_level = .err, // setting logging to info slows down rendering meaningfully, but std.debug.print does not
};
const log = std.log.scoped(.main);

const vec_type = @Vector(3, f32);
const vec3f = vector.Vector(vec_type);
const Ray = ray.MakeRay(vec_type).Ray;
const cam_types = cameras.MakeCamTypes(vec_type);

const RndGen = std.rand.Sfc64;

const Allocator = std.mem.Allocator;
const shape_s = struct {
    ny: usize = 0,
    nx: usize = 0,
};

const HitRecord = struct {
    t: f32,
    p: vec_type,
    normal: vec_type,
    material: Material,
};

const Sphere = struct {
    center: vec_type,
    radius: f32,
    material: Material,
    pub fn hit(
        sphere: Sphere,
        r: Ray,
        t_min: f32,
        t_max: f32,
    ) ?HitRecord {
        const oc = r.origin - sphere.center;
        const a = vec3f.dot(r.direction, r.direction);
        const b = vec3f.dot(oc, r.direction);
        const c = vec3f.dot(oc, oc) - sphere.radius * sphere.radius;
        const discriminant = b * b - a * c;
        if (discriminant > 0) {
            const temp = (-b - std.math.sqrt(b * b - a * c)) / a;
            if ((temp < t_max) and (temp > t_min)) {
                const point = r.point_at_parameter(temp);
                const rec = .{
                    .t = temp,
                    .p = point,
                    .normal = (point - sphere.center) / vec3f.Splat(sphere.radius),
                    .material = sphere.material,
                };
                return rec;
            }
            const temp_n = (-b + std.math.sqrt(b * b - a * c)) / a;
            if ((temp_n < t_max) and (temp_n > t_min)) {
                const point = r.point_at_parameter(temp_n);
                const rec = .{
                    .t = temp_n,
                    .p = point,
                    .normal = (point - sphere.center) / vec3f.Splat(sphere.radius),
                    .material = sphere.material,
                };
                return rec;
            }
        }
        return null;
    }
};

pub fn HittableList(T: anytype) type {
    return struct {
        objects: *const T,

        pub fn hit(
            self: @This(),
            r: Ray,
            t_min: f32,
            t_max: f32,
        ) ?HitRecord {
            var temp_rec: HitRecord = undefined;
            var hit_anything = false;
            var closest_so_far = t_max;
            for (self.objects) |obj| {
                if (obj.hit(r, t_min, closest_so_far)) |rec| {
                    hit_anything = true;
                    temp_rec = rec;
                    closest_so_far = rec.t;
                }
            }
            if (hit_anything) {
                return temp_rec;
            } else {
                return null;
            }
        }
    };
}

const ScatterResult = struct {
    attenuation: vec_type,
    ray: Ray,
};

const Material = union(enum) {
    metal: Metal,
    lambertian: Lambertian,
    dielectric: Dielectric,

    pub fn scatter(self: @This(), r_in: Ray, rec: HitRecord, rng: std.Random) ?ScatterResult {
        const result = switch (self) {
            .metal => self.metal.scatter(r_in, rec, rng),
            .lambertian => self.lambertian.scatter(r_in, rec, rng),
            .dielectric => self.dielectric.scatter(r_in, rec, rng),
        };
        return result;
    }
};

const Metal = struct {
    albedo: vec_type,
    roughness: f32,
    fn reflect(v: vec_type, n: vec_type) vec_type {
        return v - vec3f.Splat(2.0) * vec3f.Splat(vec3f.dot(v, n)) * n;
    }
    pub fn scatter(self: @This(), r_in: Ray, rec: HitRecord, rng: std.Random) ?ScatterResult {
        const reflected = reflect(vec3f.normalize(r_in.direction), rec.normal);
        const scattered: Ray = .{ .origin = rec.p, .direction = reflected + vec3f.Splat(self.roughness) * random_in_unit_sphere(rng) };

        if (vec3f.dot(scattered.direction, rec.normal) > 0) {
            return .{ .attenuation = self.albedo, .ray = scattered };
        } else {
            return null;
        }
    }
};

const Lambertian = struct {
    albedo: vec_type,
    pub fn scatter(self: @This(), r_in: Ray, rec: HitRecord, rng: anytype) ScatterResult {
        _ = r_in;
        const target = rec.p + rec.normal + random_in_unit_sphere(rng);
        const scattered: Ray = .{ .origin = rec.p, .direction = target - rec.p };

        return .{ .attenuation = self.albedo, .ray = scattered };
    }
};
const Dielectric = struct {
    color: vec_type,
    index: f32,
    fn reflect(v: vec_type, n: vec_type) vec_type {
        return v - vec3f.Splat(2.0) * vec3f.Splat(vec3f.dot(v, n)) * n;
    }
    fn refract(v: vec_type, n: vec_type, ni_nt: f32) ?vec_type {
        const uv = vec3f.normalize(v);
        const dt = vec3f.dot(uv, n);
        const discriminant = 1.0 - ni_nt * ni_nt * (1.0 - (dt * dt));
        if (discriminant > 0) {
            const refracted = vec3f.Splat(ni_nt) * (uv - n * vec3f.Splat(dt)) - n * vec3f.Splat(std.math.sqrt(discriminant));
            return refracted;
        } else {
            return null;
        }
    }
    fn schlick(cos: f32, idx: f32) f32 {
        var r0 = (1 - idx) / (1 + idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * std.math.pow(f32, 1 - cos, 5);
    }
    pub fn scatter(self: @This(), r_in: Ray, rec: HitRecord, rng: std.Random) ScatterResult {
        var outward_normal: vec_type = undefined;
        var ni_over_nt: f32 = undefined;
        var cosine: f32 = undefined;
        var reflect_prob: f32 = undefined;
        var scattered: Ray = undefined;
        const reflected = reflect(vec3f.normalize(r_in.direction), rec.normal);
        if (vec3f.dot(r_in.direction, rec.normal) > 0) {
            outward_normal = -rec.normal;
            ni_over_nt = self.index;
            cosine = self.index * vec3f.dot(r_in.direction, rec.normal) / vec3f.length(r_in.direction);
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0 / self.index;
            cosine = -vec3f.dot(r_in.direction, rec.normal) / vec3f.length(r_in.direction);
        }
        const refracted = refract(r_in.direction, outward_normal, ni_over_nt);
        if (refracted != null) {
            reflect_prob = schlick(cosine, self.index);
        } else {
            scattered = .{ .origin = rec.p, .direction = reflected };
            reflect_prob = 1.0;
        }
        if (rng.float(f32) < reflect_prob) {
            scattered = .{ .origin = rec.p, .direction = reflected };
        } else {
            scattered = .{ .origin = rec.p, .direction = refracted.? };
        }
        return .{ .attenuation = self.color, .ray = scattered };
    }
};

pub fn Render(loop_func: anytype, shape: shape_s, allocator: Allocator, rng: anytype) !std.ArrayList(@Vector(3, u8)) {
    @setFloatMode(.optimized);
    var img = try std.ArrayList(@Vector(3, u8)).initCapacity(allocator, shape.nx * shape.ny);
    log.info("Starting ray trace", .{});
    for (0..shape.ny) |jr| {
        for (0..shape.nx) |i| {
            const j = shape.ny - jr;
            const pix = loop_func(i, j, shape, rng);
            const ipix: @Vector(3, u8) = @intFromFloat(@as(vec_type, @splat(255.99)) * pix);
            img.appendAssumeCapacity(ipix);
        }
    }
    return img;
}

pub fn random_in_unit_sphere(rng: anytype) vec_type {
    const unit_cube = vec_type{ rng.float(f32), rng.float(f32), rng.float(f32) };
    const unit_sphere = vec3f.Splat(2.0) * vec3f.normalize(unit_cube) - vec3f.Splat(1.0);
    return unit_sphere * vec3f.Splat(std.math.cbrt(rng.float(f32)));
}

pub fn make_diffuse_surfs() type {
    const list: [5]Sphere = .{
        .{ .center = vec_type{ -1.0, 0.0, -1.0 }, .radius = 0.5, .material = .{ .lambertian = .{ .albedo = .{ 0.8, 0.3, 0.3 } } } },
        .{ .center = vec_type{ 0.0, -100.5, -1.0 }, .radius = 100, .material = .{ .lambertian = .{ .albedo = .{ 0.8, 0.8, 0.0 } } } },
        .{ .center = vec_type{ 1.0, 0.0, -1.0 }, .radius = 0.5, .material = .{ .metal = .{ .albedo = .{ 0.8, 0.6, 0.2 }, .roughness = 0.15 } } },
        // .{ .center = vec_type{ -1.0, 0.0, -1.0 }, .radius = 0.5, .material = .{ .metal = .{ .albedo = .{ 0.8, 0.8, 0.8 }, .roughness = 0.0 } } },
        .{ .center = vec_type{ 0.0, 0.0, -1.0 }, .radius = 0.5, .material = .{ .dielectric = .{ .color = .{ 0.95, 0.95, 0.95 }, .index = 1.5 } } },
        .{ .center = vec_type{ 0.0, 0.0, -1.0 }, .radius = -0.4, .material = .{ .dielectric = .{ .color = .{ 0.95, 0.95, 0.95 }, .index = 1.5 } } },
    };
    const Hittable = HittableList(@TypeOf(list));

    const hittable_objects = Hittable{ .objects = list[0..] };

    const num_samples = 500;
    // const depth_limit = 50;
    return struct {
        pub fn render(i: usize, j: usize, shape: shape_s, rng: anytype) vec_type {
            const fns = struct {
                fn color(r: Ray, world: Hittable, depth: u16, rnd: std.Random) vec_type {
                    if (world.hit(r, 0.001, std.math.floatMax(f32))) |rec| {
                        if (depth > 50) {
                            return vec3f.Splat(0.0);
                        }
                        if (Material.scatter(rec.material, r, rec, rnd)) |scatter_res| {
                            return scatter_res.attenuation * color(scatter_res.ray, world, depth + 1, rnd);
                        } else {
                            return vec3f.Splat(0.0);
                        }
                    }
                    const unit_dir = vec3f.normalize(r.direction);
                    const t = 0.5 * (unit_dir[1] + 1.0);
                    return vec3f.Splat(1.0 - t) * vec3f.Splat(1.0) + vec3f.Splat(t) * vec_type{ 0.5, 0.7, 1.0 };
                }
            };
            var col: vec_type = .{ 0.0, 0.0, 0.0 };
            for (0..num_samples) |_| {
                const u: f32 = (@as(f32, @floatFromInt(i)) + rng.float(f32)) / @as(f32, @floatFromInt(shape.nx));
                const v: f32 = (@as(f32, @floatFromInt(j)) + rng.float(f32)) / @as(f32, @floatFromInt(shape.ny));
                const r = cam.get_ray(u, v);
                col += fns.color(r, hittable_objects, 0, rng);
            }
            const pix = col / vec3f.Splat(num_samples);
            const rt_pix = vec_type{
                std.math.sqrt(pix[0]),
                std.math.sqrt(pix[1]),
                std.math.sqrt(pix[2]),
            };
            return rt_pix;
        }
    };
}

pub fn render_and_write(fname: []const u8, render_func: anytype, shape: shape_s, filetype: img_io.ImgType, allocator: Allocator) !void {
    var rnd = RndGen.init(0);
    const t0 = std.time.milliTimestamp();
    const img = try Render(render_func, shape, allocator, rnd.random());
    defer img.deinit();
    const t1 = std.time.milliTimestamp();

    std.debug.print("Took {d:.3} sec to render\n", .{@as(f32, @floatFromInt(t1 - t0)) / 1000});
    switch (filetype) {
        .ppm => {
            try img_io.write_ppm_image(fname, .{ .arr = img, .nx = shape.nx, .ny = shape.ny });
        },
        else => {
            // return error.Unsupported_image_type;
        },
    }
}
var cam = cam_types.MovableCamera{};
pub fn main() !void {
    const filetype = .ppm;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const nx = 1024;
    const ny = 512;
    var arena = std.heap.ArenaAllocator.init(allocator);
    cam.SetPosition(.{ -1.5, 1.0, 1.0 }, .{ 0.0, 0.0, -1.0 }, .{ 0.0, 1.0, 0.0 }, 75, 2.0);

    // var buf: [4096]u8 = undefined;
    // var fba = std.heap.FixedBufferAllocator.init(&buf);

    // const multisphere = make_multiple_spheres();
    // const antialiasing = make_antialiasing();
    const diffuse = make_diffuse_surfs();

    log.info("Creating {} x {} image", .{ nx, ny });
    // try render_and_write("test_image.ppm", .{ .nx = nx, .ny = ny }, hello_graphics, arena.allocator());
    // try render_and_write("my_first_raytrace.ppm", .{ .nx = nx, .ny = ny }, my_first_raytrace, arena.allocator());
    // try render_and_write("surface_normals.ppm", .{ .nx = nx, .ny = ny }, surface_normals, arena.allocator());
    // try render_and_write("multiple_spheres.ppm", .{ .nx = nx, .ny = ny }, multisphere.render, arena.allocator());
    // try render_and_write("antialiasing.ppm", .{ .nx = nx, .ny = ny }, antialiasing.render, arena.allocator());
    try render_and_write("reflective.ppm", diffuse.render, .{ .nx = nx, .ny = ny }, filetype, arena.allocator());
}
