const std = @import("std");
const vector = @import("vector.zig");
const ray = @import("ray.zig");
const cameras = @import("camera.zig");

pub const log_level: std.log.Level = .debug;
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
};

const Sphere = struct {
    center: vec_type,
    radius: f32,

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

pub fn random_in_unit_sphere(rng: anytype) vec_type {
    const r3_3 = std.math.sqrt(3.0) / 3.0 - 0.00001;
    const point = vec3f.Splat(2.0 * r3_3) * vec_type{ rng.float(f32), rng.float(f32), rng.float(f32) } - vec3f.Splat(1.0 * r3_3);
    return point;
}

pub fn write_ppm_image(fname: []const u8, shape: anytype, loop_func: anytype, allocator: Allocator) !void {
    var rnd = RndGen.init(0);
    const fs = std.fs;
    const file = try fs.cwd().createFile(
        fname,
        .{ .read = true },
    );
    defer file.close();
    var BW = std.io.bufferedWriter(file.writer());
    const writer = BW.writer(); // output was black image, don't know why
    // const writer = file.writer();
    try std.fmt.format(writer, "P3\n{} {}\n255\n", .{ shape.nx, shape.ny });
    // allocate image aray here
    var img = try std.ArrayList(@Vector(3, u8)).initCapacity(allocator, shape.nx * shape.ny);
    defer img.deinit();
    log.info("Starting ray trace", .{});
    for (0..shape.ny) |jr| {
        for (0..shape.nx) |i| {
            const j = shape.ny - jr;
            const pix = loop_func(i, j, shape, rnd.random());
            const ipix: @Vector(3, u8) = @intFromFloat(@as(vec_type, @splat(255.99)) * pix);
            // put in array here, move write to file to new loop
            try img.append(ipix);
        }
    }
    log.info("Starting file save", .{});
    for (try img.toOwnedSlice()) |rgb_pix| {
        try std.fmt.format(writer, "{} {} {}\n", .{ rgb_pix[0], rgb_pix[1], rgb_pix[2] });
    }
    try BW.flush();
    log.info("Done", .{});
}

fn hello_graphics(i: usize, j: usize, shape: shape_s, rnd: anytype) vec_type {
    _ = rnd;
    const r: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(shape.nx));
    const g: f32 = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(shape.ny));
    const b: f32 = 0.2;
    const color = vec_type{ r, g, b };
    return color;
}

pub fn my_first_raytrace(i: usize, j: usize, shape: shape_s, rnd: anytype) vec_type {
    _ = rnd;
    const fns = struct {
        fn hit_sphere(center: vec_type, radius: f32, r: Ray) bool {
            const oc = r.origin - center;
            const a = vec3f.dot(r.direction, r.direction);
            const b = 2.0 * vec3f.dot(oc, r.direction);
            const c = vec3f.dot(oc, oc) - radius * radius;
            const discriminant = b * b - 4 * a * c;
            return discriminant > 0;
        }
        fn color(r: Ray) vec_type {
            if (hit_sphere(.{ 0.0, 0.0, -1.0 }, 0.5, r)) {
                return .{ 1.0, 0.0, 0.0 };
            }
            const unit_dir = vec3f.normalize(r.direction);
            const t = 0.5 * (unit_dir[1] + 1);
            return vec3f.Splat(1.0 - t) * vec3f.Splat(1.0) + vec3f.Splat(t) * vec_type{ 0.5, 0.7, 1.0 };
        }
    };
    const lower_left: vec_type = .{ -2.0, -1.0, -1.0 };
    const horizontal: vec_type = .{ 4.0, 0.0, 0.0 };
    const vertical: vec_type = .{ 0.0, 2.0, 0.0 };
    const origin: vec_type = .{ 0.0, 0.0, 0.0 };
    const u: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(shape.nx));
    const v: f32 = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(shape.ny));
    const r = Ray{ .origin = origin, .direction = lower_left + vec3f.Splat(u) * horizontal + vec3f.Splat(v) * vertical };
    const col = fns.color(r);
    return col;
}

pub fn surface_normals(i: usize, j: usize, shape: shape_s, rnd: anytype) vec_type {
    _ = rnd;
    const fns = struct {
        fn hit_sphere(center: vec_type, radius: f32, r: Ray) f32 {
            const oc = r.origin - center;
            const a = vec3f.dot(r.direction, r.direction);
            const b = 2.0 * vec3f.dot(oc, r.direction);
            const c = vec3f.dot(oc, oc) - radius * radius;
            const discriminant = b * b - 4 * a * c;
            if (discriminant < 0) {
                return -1.0;
            } else {
                return (-b - std.math.sqrt(discriminant)) / (2.0 * a);
            }
        }
        fn color(r: Ray) vec_type {
            var t = hit_sphere(.{ 0.0, 0.0, -1.0 }, 0.5, r);
            if (t > 0) {
                const N = vec3f.normalize(r.point_at_parameter(t) - vec_type{ 0.0, 0.0, -1.0 });
                return vec3f.Splat(0.5) * (N + vec3f.Splat(1.0));
            }
            const unit_dir = vec3f.normalize(r.direction);
            t = 0.5 * (unit_dir[1] + 1);
            return vec3f.Splat(1.0 - t) * vec3f.Splat(1.0) + vec3f.Splat(t) * vec_type{ 0.5, 0.7, 1.0 };
        }
    };
    const lower_left: vec_type = .{ -2.0, -1.0, -1.0 };
    const horizontal: vec_type = .{ 4.0, 0.0, 0.0 };
    const vertical: vec_type = .{ 0.0, 2.0, 0.0 };
    const origin: vec_type = .{ 0.0, 0.0, 0.0 };
    const u: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(shape.nx));
    const v: f32 = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(shape.ny));
    const r = Ray{ .origin = origin, .direction = lower_left + vec3f.Splat(u) * horizontal + vec3f.Splat(v) * vertical };
    const col = fns.color(r);
    return col;
}

pub fn make_multiple_spheres() type {
    const list: [2]Sphere = .{
        .{ .center = vec_type{ 0.0, 0.0, -1.0 }, .radius = 0.5 },
        .{ .center = vec_type{ 0.0, -100.5, -1.0 }, .radius = 100 },
    };
    const Hittable = HittableList(@TypeOf(list));

    const hittable_objects = Hittable{ .objects = list[0..] };
    const cam = cam_types.BasicCamera{};
    return struct {
        pub fn render(i: usize, j: usize, shape: shape_s, rnd: anytype) vec_type {
            _ = rnd;
            const fns = struct {
                fn color(r: Ray, world: Hittable) vec_type {
                    if (world.hit(r, 0.0, std.math.floatMax(f32))) |rec| {
                        return vec3f.Splat(0.5) * (vec3f.normalize(rec.normal) + vec3f.Splat(1.0));
                    }
                    const unit_dir = vec3f.normalize(r.direction);
                    const t = 0.5 * (unit_dir[1] + 1.0);
                    return vec3f.Splat(1.0 - t) * vec3f.Splat(1.0) + vec3f.Splat(t) * vec_type{ 0.5, 0.7, 1.0 };
                }
            };

            const u: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(shape.nx));
            const v: f32 = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(shape.ny));
            const r = cam.get_ray(u, v);
            const col = fns.color(r, hittable_objects);
            return col;
        }
    };
}

pub fn make_antialiasing() type {
    const list: [2]Sphere = .{
        .{ .center = vec_type{ 0.0, 0.0, -1.0 }, .radius = 0.5 },
        .{ .center = vec_type{ 0.0, -100.5, -1.0 }, .radius = 100 },
    };
    const Hittable = HittableList(@TypeOf(list));

    const hittable_objects = Hittable{ .objects = list[0..] };
    const cam = cam_types.BasicCamera{};
    const num_samples = 100;
    return struct {
        pub fn render(i: usize, j: usize, shape: shape_s, rnd: anytype) vec_type {
            const fns = struct {
                fn color(r: Ray, world: Hittable) vec_type {
                    if (world.hit(r, 0.0, std.math.floatMax(f32))) |rec| {
                        return vec3f.Splat(0.5) * (vec3f.normalize(rec.normal) + vec3f.Splat(1.0));
                    }
                    const unit_dir = vec3f.normalize(r.direction);
                    const t = 0.5 * (unit_dir[1] + 1.0);
                    return vec3f.Splat(1.0 - t) * vec3f.Splat(1.0) + vec3f.Splat(t) * vec_type{ 0.5, 0.7, 1.0 };
                }
            };
            var col: vec_type = .{ 0.0, 0.0, 0.0 };
            for (0..num_samples) |_| {
                const u: f32 = (@as(f32, @floatFromInt(i)) + rnd.float(f32)) / @as(f32, @floatFromInt(shape.nx));
                const v: f32 = (@as(f32, @floatFromInt(j)) + rnd.float(f32)) / @as(f32, @floatFromInt(shape.ny));
                const r = cam.get_ray(u, v);
                col += fns.color(r, hittable_objects);
            }
            const pix = col / vec3f.Splat(num_samples);
            return pix;
        }
    };
}

pub fn make_diffuse_surfs() type {
    const list: [2]Sphere = .{
        .{ .center = vec_type{ 0.0, 0.0, -1.0 }, .radius = 0.5 },
        .{ .center = vec_type{ 0.0, -100.5, -1.0 }, .radius = 100 },
    };
    const Hittable = HittableList(@TypeOf(list));

    const hittable_objects = Hittable{ .objects = list[0..] };
    const cam = cam_types.BasicCamera{};
    const num_samples = 1000;
    return struct {
        pub fn render(i: usize, j: usize, shape: shape_s, rnd: anytype) vec_type {
            const fns = struct {
                fn color(r: Ray, world: Hittable, rng: anytype) vec_type {
                    if (world.hit(r, 0.001, std.math.floatMax(f32))) |rec| {
                        const target = rec.p + rec.normal + random_in_unit_sphere(rng);
                        const new_ray = Ray{ .origin = rec.p, .direction = target - rec.p };
                        return vec3f.Splat(0.5) * color(new_ray, world, rng);
                    }
                    const unit_dir = vec3f.normalize(r.direction);
                    const t = 0.5 * (unit_dir[1] + 1.0);
                    return vec3f.Splat(1.0 - t) * vec3f.Splat(1.0) + vec3f.Splat(t) * vec_type{ 0.5, 0.7, 1.0 };
                }
            };
            var col: vec_type = .{ 0.0, 0.0, 0.0 };
            for (0..num_samples) |_| {
                const u: f32 = (@as(f32, @floatFromInt(i)) + rnd.float(f32)) / @as(f32, @floatFromInt(shape.nx));
                const v: f32 = (@as(f32, @floatFromInt(j)) + rnd.float(f32)) / @as(f32, @floatFromInt(shape.ny));
                const r = cam.get_ray(u, v);
                col += fns.color(r, hittable_objects, rnd);
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

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const nx = 1024;
    const ny = 512;
    var arena = std.heap.ArenaAllocator.init(allocator);
    // var buf: [4096]u8 = undefined;
    // var fba = std.heap.FixedBufferAllocator.init(&buf);

    // const multisphere = make_multiple_spheres();
    // const antialiasing = make_antialiasing();
    const diffuse = make_diffuse_surfs();

    log.info("Creating {} x {} image", .{ nx, ny });
    // try write_ppm_image("test_image.ppm", .{ .nx = nx, .ny = ny }, hello_graphics, arena.allocator());
    // try write_ppm_image("my_first_raytrace.ppm", .{ .nx = nx, .ny = ny }, my_first_raytrace, arena.allocator());
    // try write_ppm_image("surface_normals.ppm", .{ .nx = nx, .ny = ny }, surface_normals, arena.allocator());
    // try write_ppm_image("multiple_spheres.ppm", .{ .nx = nx, .ny = ny }, multisphere.render, arena.allocator());
    // try write_ppm_image("antialiasing.ppm", .{ .nx = nx, .ny = ny }, antialiasing.render, arena.allocator());
    try write_ppm_image("diffuse.ppm", .{ .nx = nx, .ny = ny }, diffuse.render, arena.allocator());
}
