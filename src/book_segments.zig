// fn hello_graphics(i: usize, j: usize, shape: shape_s, rnd: anytype) vec_type {
//     _ = rnd;
//     const r: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(shape.nx));
//     const g: f32 = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(shape.ny));
//     const b: f32 = 0.2;
//     const color = vec_type{ r, g, b };
//     return color;
// }

// pub fn my_first_raytrace(i: usize, j: usize, shape: shape_s, rnd: anytype) vec_type {
//     _ = rnd;
//     const fns = struct {
//         fn hit_sphere(center: vec_type, radius: f32, r: Ray) bool {
//             const oc = r.origin - center;
//             const a = vec3f.dot(r.direction, r.direction);
//             const b = 2.0 * vec3f.dot(oc, r.direction);
//             const c = vec3f.dot(oc, oc) - radius * radius;
//             const discriminant = b * b - 4 * a * c;
//             return discriminant > 0;
//         }
//         fn color(r: Ray) vec_type {
//             if (hit_sphere(.{ 0.0, 0.0, -1.0 }, 0.5, r)) {
//                 return .{ 1.0, 0.0, 0.0 };
//             }
//             const unit_dir = vec3f.normalize(r.direction);
//             const t = 0.5 * (unit_dir[1] + 1);
//             return vec3f.Splat(1.0 - t) * vec3f.Splat(1.0) + vec3f.Splat(t) * vec_type{ 0.5, 0.7, 1.0 };
//         }
//     };
//     const lower_left: vec_type = .{ -2.0, -1.0, -1.0 };
//     const horizontal: vec_type = .{ 4.0, 0.0, 0.0 };
//     const vertical: vec_type = .{ 0.0, 2.0, 0.0 };
//     const origin: vec_type = .{ 0.0, 0.0, 0.0 };
//     const u: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(shape.nx));
//     const v: f32 = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(shape.ny));
//     const r = Ray{ .origin = origin, .direction = lower_left + vec3f.Splat(u) * horizontal + vec3f.Splat(v) * vertical };
//     const col = fns.color(r);
//     return col;
// }

// pub fn surface_normals(i: usize, j: usize, shape: shape_s, rnd: anytype) vec_type {
//     _ = rnd;
//     const fns = struct {
//         fn hit_sphere(center: vec_type, radius: f32, r: Ray) f32 {
//             const oc = r.origin - center;
//             const a = vec3f.dot(r.direction, r.direction);
//             const b = 2.0 * vec3f.dot(oc, r.direction);
//             const c = vec3f.dot(oc, oc) - radius * radius;
//             const discriminant = b * b - 4 * a * c;
//             if (discriminant < 0) {
//                 return -1.0;
//             } else {
//                 return (-b - std.math.sqrt(discriminant)) / (2.0 * a);
//             }
//         }
//         fn color(r: Ray) vec_type {
//             var t = hit_sphere(.{ 0.0, 0.0, -1.0 }, 0.5, r);
//             if (t > 0) {
//                 const N = vec3f.normalize(r.point_at_parameter(t) - vec_type{ 0.0, 0.0, -1.0 });
//                 return vec3f.Splat(0.5) * (N + vec3f.Splat(1.0));
//             }
//             const unit_dir = vec3f.normalize(r.direction);
//             t = 0.5 * (unit_dir[1] + 1);
//             return vec3f.Splat(1.0 - t) * vec3f.Splat(1.0) + vec3f.Splat(t) * vec_type{ 0.5, 0.7, 1.0 };
//         }
//     };
//     const lower_left: vec_type = .{ -2.0, -1.0, -1.0 };
//     const horizontal: vec_type = .{ 4.0, 0.0, 0.0 };
//     const vertical: vec_type = .{ 0.0, 2.0, 0.0 };
//     const origin: vec_type = .{ 0.0, 0.0, 0.0 };
//     const u: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(shape.nx));
//     const v: f32 = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(shape.ny));
//     const r = Ray{ .origin = origin, .direction = lower_left + vec3f.Splat(u) * horizontal + vec3f.Splat(v) * vertical };
//     const col = fns.color(r);
//     return col;
// }

// pub fn make_multiple_spheres() type {
//     const list: [2]Sphere = .{
//         .{ .center = vec_type{ 0.0, 0.0, -1.0 }, .radius = 0.5 },
//         .{ .center = vec_type{ 0.0, -100.5, -1.0 }, .radius = 100 },
//     };
//     const Hittable = HittableList(@TypeOf(list));

//     const hittable_objects = Hittable{ .objects = list[0..] };
//     const cam = cam_types.BasicCamera{};
//     return struct {
//         pub fn render(i: usize, j: usize, shape: shape_s, rnd: anytype) vec_type {
//             _ = rnd;
//             const fns = struct {
//                 fn color(r: Ray, world: Hittable) vec_type {
//                     if (world.hit(r, 0.0, std.math.floatMax(f32))) |rec| {
//                         return vec3f.Splat(0.5) * (vec3f.normalize(rec.normal) + vec3f.Splat(1.0));
//                     }
//                     const unit_dir = vec3f.normalize(r.direction);
//                     const t = 0.5 * (unit_dir[1] + 1.0);
//                     return vec3f.Splat(1.0 - t) * vec3f.Splat(1.0) + vec3f.Splat(t) * vec_type{ 0.5, 0.7, 1.0 };
//                 }
//             };

//             const u: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(shape.nx));
//             const v: f32 = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(shape.ny));
//             const r = cam.get_ray(u, v);
//             const col = fns.color(r, hittable_objects);
//             return col;
//         }
//     };
// }

// pub fn make_antialiasing() type {
//     const list: [2]Sphere = .{
//         .{ .center = vec_type{ 0.0, 0.0, -1.0 }, .radius = 0.5 },
//         .{ .center = vec_type{ 0.0, -100.5, -1.0 }, .radius = 100 },
//     };
//     const Hittable = HittableList(@TypeOf(list));

//     const hittable_objects = Hittable{ .objects = list[0..] };
//     const cam = cam_types.BasicCamera{};
//     const num_samples = 100;
//     return struct {
//         pub fn render(i: usize, j: usize, shape: shape_s, rnd: anytype) vec_type {
//             const fns = struct {
//                 fn color(r: Ray, world: Hittable) vec_type {
//                     if (world.hit(r, 0.0, std.math.floatMax(f32))) |rec| {
//                         return vec3f.Splat(0.5) * (vec3f.normalize(rec.normal) + vec3f.Splat(1.0));
//                     }
//                     const unit_dir = vec3f.normalize(r.direction);
//                     const t = 0.5 * (unit_dir[1] + 1.0);
//                     return vec3f.Splat(1.0 - t) * vec3f.Splat(1.0) + vec3f.Splat(t) * vec_type{ 0.5, 0.7, 1.0 };
//                 }
//             };
//             var col: vec_type = .{ 0.0, 0.0, 0.0 };
//             for (0..num_samples) |_| {
//                 const u: f32 = (@as(f32, @floatFromInt(i)) + rnd.float(f32)) / @as(f32, @floatFromInt(shape.nx));
//                 const v: f32 = (@as(f32, @floatFromInt(j)) + rnd.float(f32)) / @as(f32, @floatFromInt(shape.ny));
//                 const r = cam.get_ray(u, v);
//                 col += fns.color(r, hittable_objects);
//             }
//             const pix = col / vec3f.Splat(num_samples);
//             return pix;
//         }
//     };
// }

// pub fn make_diffuse_surfs() type {
//     const list: [2]Sphere = .{
//         .{ .center = vec_type{ 0.0, 0.0, -1.0 }, .radius = 0.5 },
//         .{ .center = vec_type{ 0.0, -100.5, -1.0 }, .radius = 100 },
//     };
//     const Hittable = HittableList(@TypeOf(list));

//     const hittable_objects = Hittable{ .objects = list[0..] };
//     const cam = cam_types.BasicCamera{};
//     const num_samples = 100;
//     return struct {
//         pub fn render(i: usize, j: usize, shape: shape_s, rnd: anytype) vec_type {
//             const fns = struct {
//                 fn color(r: Ray, world: Hittable, rng: anytype) vec_type {
//                     if (world.hit(r, 0.001, std.math.floatMax(f32))) |rec| {
//                         const target = rec.p + rec.normal + random_in_unit_sphere(rng);
//                         const new_ray = Ray{ .origin = rec.p, .direction = target - rec.p };
//                         return vec3f.Splat(0.5) * color(new_ray, world, rng);
//                     }
//                     const unit_dir = vec3f.normalize(r.direction);
//                     const t = 0.5 * (unit_dir[1] + 1.0);
//                     return vec3f.Splat(1.0 - t) * vec3f.Splat(1.0) + vec3f.Splat(t) * vec_type{ 0.5, 0.7, 1.0 };
//                 }
//             };
//             var col: vec_type = .{ 0.0, 0.0, 0.0 };
//             for (0..num_samples) |_| {
//                 const u: f32 = (@as(f32, @floatFromInt(i)) + rnd.float(f32)) / @as(f32, @floatFromInt(shape.nx));
//                 const v: f32 = (@as(f32, @floatFromInt(j)) + rnd.float(f32)) / @as(f32, @floatFromInt(shape.ny));
//                 const r = cam.get_ray(u, v);
//                 col += fns.color(r, hittable_objects, rnd);
//             }
//             const pix = col / vec3f.Splat(num_samples);
//             const rt_pix = vec_type{
//                 std.math.sqrt(pix[0]),
//                 std.math.sqrt(pix[1]),
//                 std.math.sqrt(pix[2]),
//             };
//             return rt_pix;
//         }
//     };
// }
