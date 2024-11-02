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
    center: vec_type, // 16 bytes
    material: Material,
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
    pub inline fn lessThan(ax: u2, a: Sphere, b: Sphere) bool {
        if ((a.center[ax] - a.radius) < (b.center[ax] - b.radius)) {
            return true;
        } else {
            return false;
        }
    }
};

pub fn HittableList(T: anytype) type {
    return struct {
        objects: T,

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

const interval = struct {
    min: f32 = undefined,
    max: f32 = undefined,
};

const AxisAlignedBBox = struct {
    x: interval,
    y: interval,
    z: interval,
    pub fn make(a: vec_type, b: vec_type) AxisAlignedBBox {
        var aabb: AxisAlignedBBox = undefined;
        if (a[0] <= b[0]) {
            aabb.x = .{ .min = a[0], .max = b[0] };
        } else {
            aabb.x = .{ .min = b[0], .max = a[0] };
        }
        if (a[1] <= b[1]) {
            aabb.y = .{ .min = a[1], .max = b[1] };
        } else {
            aabb.y = .{ .min = b[1], .max = a[1] };
        }
        if (a[2] <= b[2]) {
            aabb.z = .{ .min = a[2], .max = b[2] };
        } else {
            aabb.z = .{ .min = b[2], .max = a[2] };
        }
        return aabb;
    }
    pub fn axis_interval(self: @This(), num: u2) interval {
        const ax = switch (num) {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            else => unreachable,
        };
        return ax;
    }
    pub fn hit(
        self: @This(),
        r: Ray,
        t_min: f32,
        t_max: f32,
    ) bool {
        var ray_t: interval = .{ .min = t_min, .max = t_max };
        for (0..2) |axis| {
            const ax = self.axis_interval(@intCast(axis));
            const adinv = 1.0 / r.direction[axis];

            const t0 = (ax.min - r.origin[axis]) * adinv;
            const t1 = (ax.max - r.origin[axis]) * adinv;
            // if (t0 != t0 or t1 != t1) {
            //     return false;
            // }
            if (t0 < t1) {
                if (t0 > ray_t.min) ray_t.min = t0;
                if (t1 < ray_t.max) ray_t.max = t1;
            } else {
                if (t1 > ray_t.min) ray_t.min = t1;
                if (t0 < ray_t.max) ray_t.max = t0;
            }

            if (ray_t.max <= ray_t.min)
                return false;
        }
        return true;
    }
};

// const NodeType = enum { branch, leaf };

const TreeNode = union(enum) {
    branch: BranchNode,
    leaf: LeafNode,

    pub fn hit(
        self: @This(),
        r: Ray,
        t_min: f32,
        t_max: f32,
    ) ?HitRecord {
        const rec = switch (self) {
            .branch => |*f| f.*.hit(r, t_min, t_max),
            .leaf => |*f| f.*.hit(r, t_min, t_max),
        };
        return rec;
    }
};
const BranchNode = struct {
    bbox: AxisAlignedBBox,
    child1: *TreeNode,
    child2: *TreeNode,
    pub fn hit(
        self: @This(),
        r: Ray,
        t_min: f32,
        t_max: f32,
    ) ?HitRecord {
        // check if our bbox is hit
        if (self.bbox.hit(r, t_min, t_max)) {
            const rec1 = self.child1.hit(r, t_min, t_max);
            const rec2 = self.child2.hit(r, t_min, t_max);
            if ((rec1 != null) and (rec2 != null)) {
                if (rec1.?.t < rec2.?.t) {
                    return rec1;
                } else {
                    return rec2;
                }
            } else if (rec1 != null) {
                return rec1.?;
            } else if (rec2 != null) {
                return rec2.?;
            } else {
                return null;
            }
        } else {
            return null;
        }
    }
};
const LeafNode = struct {
    bbox: AxisAlignedBBox,
    obj_list: *std.ArrayList(Sphere),
    inds: []usize,
    pub fn hit(
        self: @This(),
        r: Ray,
        t_min: f32,
        t_max: f32,
    ) ?HitRecord {
        if (self.bbox.hit(r, t_min, t_max)) {
            var temp_rec: HitRecord = undefined;
            var hit_anything = false;
            var closest_so_far = t_max;
            for (self.inds) |idx| {
                if (self.obj_list.*.items[idx].hit(r, t_min, closest_so_far)) |rec| {
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
        } else {
            return null;
        }
    }
};

const obj_ind = struct {
    ind: usize,
    obj: Sphere,

    pub fn lessThan(ax: u2, lhs: obj_ind, rhs: obj_ind) bool {
        return Sphere.lessThan(ax, lhs.obj, rhs.obj);
    }
};

pub fn build_BVH(parent_arr: *std.ArrayList(Sphere), obj_inds: []usize, depth: u32, allocator: std.mem.Allocator) *TreeNode {
    var objects = std.ArrayList(obj_ind).init(allocator);
    defer objects.deinit();
    for (obj_inds) |ind| {
        objects.append(.{ .ind = ind, .obj = parent_arr.*.items[ind] }) catch unreachable;
    }

    var min_corner: vec_type = .{ 99999.0, 99999.0, 99999.0 };
    var max_corner: vec_type = .{ -99999.0, -99999.0, -99999.0 };
    for (objects.items) |obj| {
        for (0..3) |i| {
            if ((obj.obj.center[i] - obj.obj.radius) < min_corner[i]) {
                min_corner[i] = obj.obj.center[i] - obj.obj.radius;
            }
            if ((obj.obj.center[i] + obj.obj.radius) > max_corner[i]) {
                max_corner[i] = obj.obj.center[i] + obj.obj.radius;
            }
        }
    }
    // std.debug.print("BBOX for this group: min {d:.3},{d:.3},{d:.3} , max {d:.3},{d:.3},{d:.3}\n", .{ min_corner[0], min_corner[1], min_corner[2], max_corner[0], max_corner[1], max_corner[2] });
    const new_bbox = AxisAlignedBBox.make(min_corner, max_corner);

    if ((depth == 0) or (obj_inds.len <= 3)) {
        // var leaf = allocator.create(LeafNode) catch unreachable;

        const inds = allocator.alloc(usize, obj_inds.len) catch unreachable;
        errdefer allocator.free(inds);
        std.mem.copyForwards(usize, inds, obj_inds);
        // errdefer allocator.free(node);
        var leaf: LeafNode = undefined;
        leaf.inds = inds;
        leaf.obj_list = parent_arr;
        leaf.bbox = new_bbox;
        const node = allocator.create(TreeNode) catch unreachable;
        node.* = .{ .leaf = leaf };

        // var out = TreeNode{ .leaf = node[0] };
        // std.debug.print("Made a leaf with {} items\n", .{node.leaf.inds.len});
        return node;
    }
    std.debug.print("interval x: {d:.3},{d:.3}, y: {d:.3},{d:.3}, z: {d:.3},{d:.3}\n", .{
        new_bbox.x.min,
        new_bbox.x.max,
        new_bbox.y.min,
        new_bbox.y.max,
        new_bbox.z.min,
        new_bbox.z.max,
    });
    // split the list of objects in half along the longest axis
    const dimensions = max_corner - min_corner;
    const biggest_ax = vec3f.argmax(dimensions);
    std.sort.heap(obj_ind, objects.items, biggest_ax, obj_ind.lessThan);
    var sorted_inds = std.ArrayList(usize).init(allocator);
    defer sorted_inds.deinit();
    for (objects.items) |obj| {
        sorted_inds.append(obj.ind) catch unreachable;
    }
    const thresh = dimensions[biggest_ax] / 2 + min_corner[biggest_ax];
    std.debug.print("Threshold position: {d:.3} on ax: {}\n", .{ thresh, biggest_ax });
    var thresh_idx: u32 = 1;
    for (objects.items) |obj| {
        const point = obj.obj.center[biggest_ax] + obj.obj.radius;
        if (point > thresh) {
            break;
        }
        thresh_idx += 1;
    }
    var num_below: u32 = 0;
    var num_above: u32 = 0;
    for (objects.items) |obj| {
        const point = obj.obj.center[biggest_ax] + obj.obj.radius;
        if (point > thresh) {
            num_above += 1;
        } else {
            num_below += 1;
        }
    }
    std.debug.print("{} above thresh, {} below thresh\n", .{ num_above, num_below });
    thresh_idx = @min(thresh_idx, obj_inds.len - 1);
    // var node = allocator.alloc(BranchNode, 1) catch unreachable;
    // errdefer allocator.free(node);
    var node: BranchNode = undefined;
    var split: u32 = undefined;
    if (true) { // tree partitioning method
        split = thresh_idx;
    } else {
        split = @intCast(obj_inds.len / 2);
    }
    std.debug.print("Adding branch node at depth {}, with {} objects remaining, splitting at {}, along {}\n\n", .{ depth, obj_inds.len, thresh_idx, biggest_ax });
    node.bbox = new_bbox;
    node.child1 = build_BVH(parent_arr, sorted_inds.items[0..split], depth - 1, allocator);
    node.child2 = build_BVH(parent_arr, sorted_inds.items[split..], depth - 1, allocator);
    const out = allocator.create(TreeNode) catch unreachable;
    out.* = .{ .branch = node };
    return out;
}

pub fn HittableBVH(T: anytype) type {
    return struct {
        objects: T,
        bvh: *TreeNode,
        pub fn hit(
            self: @This(),
            r: Ray,
            t_min: f32,
            t_max: f32,
        ) ?HitRecord {
            if (self.bvh.hit(r, t_min, t_max)) |rec| {
                return rec;
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
    albedo: vec_type, // 16 bytes
    roughness: f32, // 16 bytes because of alignment
    fn reflect(v: vec_type, n: vec_type) vec_type {
        return v - vec3f.Splat(2.0) * vec3f.Splat(vec3f.dot(v, n)) * n;
    }
    pub fn scatter(self: @This(), r_in: Ray, rec: HitRecord, rng: std.Random) ?ScatterResult {
        const reflected = reflect(vec3f.normalize(r_in.direction), rec.normal);
        const scattered: Ray = .{ .origin = rec.p, .direction = reflected + vec3f.Splat(self.roughness) * random_in_unit_sphere(rng) };

        if (vec3f.dot(scattered.direction, rec.normal) > 0) {
            check_neg3(self.albedo);
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
        check_neg3(self.albedo);

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
        check_neg3(self.color);
        return .{ .attenuation = self.color, .ray = scattered };
    }
};

pub fn check_nan3(v: vec_type) void {
    _ = v;
    // if (v[0] != v[0] or v[1] != v[1] or v[2] != v[2]) {
    //     std.debug.print("v: {d:.3},{d:.3},{d:.3}\n", .{ v[0], v[1], v[2] });
    //     @panic("Pixel value out of range! Nan!");
    // }
}

pub fn check_neg3(v: vec_type) void {
    _ = v;
    // if (v[0] < 0.0 or v[1] < 0.0 or v[2] < 0.0) {
    //     std.debug.print("pix: {},{},{}\n", .{ v[0], v[1], v[2] });
    //     @panic("Pixel value out of range! Negative!");
    // }
}
pub fn Render(world: Hittable, shape: shape_s, seed: u64, img: *std.ArrayList(@Vector(3, u8))) !void {
    @setFloatMode(.optimized);
    var rnd = RndGen.init(seed);
    log.info("Starting ray trace", .{});
    for (0..shape.ny) |jr| {
        for (0..shape.nx) |i| {
            const j = shape.ny - jr;
            const pix = trace(i, j, shape, rnd.random(), world);
            check_nan3(pix);
            // std.debug.print("pix: {d:.3},{d:.3},{d:.3}\n", .{ pix[0], pix[1], pix[2] });
            const ipix: @Vector(3, u8) = @intFromFloat(@as(vec_type, @splat(255.99)) * pix);
            try img.append(ipix);
        }
    }
    // return img;
}

pub fn trace(i: usize, j: usize, shape: shape_s, rng: anytype, world: Hittable) vec_type {
    var col: vec_type = .{ 0.0, 0.0, 0.0 };
    for (0..num_samples) |_| {
        const u: f32 = (@as(f32, @floatFromInt(i)) + rng.float(f32)) / @as(f32, @floatFromInt(shape.nx));
        const v: f32 = (@as(f32, @floatFromInt(j)) + rng.float(f32)) / @as(f32, @floatFromInt(shape.ny));
        const r = cam.get_ray(u, v);
        col += color(r, world, 0, rng);
    }
    const pix = col / vec3f.Splat(num_samples);
    check_nan3(pix);
    check_neg3(pix);
    const rt_pix = vec_type{
        std.math.sqrt(pix[0]),
        std.math.sqrt(pix[1]),
        std.math.sqrt(pix[2]),
    };
    check_nan3(rt_pix);
    return rt_pix;
}

fn color(r: Ray, world: Hittable, depth: u16, rnd: std.Random) vec_type {
    if (world.hit(r, 0.001, std.math.floatMax(f32))) |rec| {
        if (depth > 16) {
            return vec3f.Splat(0.0);
        }
        if (Material.scatter(rec.material, r, rec, rnd)) |scatter_res| {
            const c = color(scatter_res.ray, world, depth + 1, rnd);
            check_nan3(c);
            check_neg3(c);
            check_neg3(scatter_res.attenuation);
            return scatter_res.attenuation * c;
        } else {
            return vec3f.Splat(0.0);
        }
    }
    const unit_dir = vec3f.normalize(r.direction);
    const t = 0.5 * (unit_dir[1] + 1.0);
    return vec3f.Splat(1.0 - t) * vec3f.Splat(1.0) + vec3f.Splat(t) * vec_type{ 0.5, 0.7, 1.0 };
}

pub fn random_in_unit_sphere(rng: anytype) vec_type {
    const unit_cube = vec_type{ rng.float(f32), rng.float(f32), rng.float(f32) };
    const unit_sphere = vec3f.Splat(2.0) * vec3f.normalize(unit_cube) - vec3f.Splat(1.0);
    return unit_sphere * vec3f.Splat(std.math.cbrt(rng.float(f32)));
}

pub fn render_and_write(fname: []const u8, world: Hittable, shape: shape_s, filetype: img_io.ImgType, allocator: Allocator) !void {
    const t0 = std.time.milliTimestamp();
    var final_img = try std.ArrayList(@Vector(3, u8)).initCapacity(allocator, shape.nx * shape.ny);
    defer final_img.deinit();

    if (std.Thread.getCpuCount()) |num_cpus| {
        const img_type = std.ArrayList(@Vector(3, u8));
        var thread_config = std.Thread.SpawnConfig{};
        thread_config.allocator = allocator;
        var threads = std.ArrayList(std.Thread).init(allocator);
        var images = std.ArrayList(img_type).init(allocator);
        for (0..num_cpus) |i| {
            const img = try img_type.initCapacity(allocator, shape.nx * shape.ny);

            try images.append(img);
            const handle =
                try std.Thread.spawn(thread_config, Render, .{ world, shape, i * 123, &images.items[i] });
            try threads.append(handle);
        }
        std.debug.print("Started {} threads\n", .{num_cpus});

        for (threads.items) |thread| {
            thread.join();
        }
        std.debug.print("all threads finished\n", .{});

        for (0..shape.nx * shape.ny) |idx| {
            var tmp_pix: @Vector(3, u32) = .{ 0, 0, 0 };
            for (images.items) |img| {
                tmp_pix += @intCast(img.items[idx]);
            }
            // std.debug.print("pix: {d:.3},{d:.3},{d:.3}\n", .{ tmp_pix[0] / num_cpus, tmp_pix[1] / num_cpus, tmp_pix[2] / num_cpus });
            try final_img.append(@intCast(tmp_pix / @as(@Vector(3, u32), @splat(@intCast(num_cpus)))));
            // try final_img.append(images.items[0].items[idx]);
        }
    } else |_| {
        try Render(world, shape, 123, &final_img);
    }
    const t1 = std.time.milliTimestamp();

    std.debug.print("Took {d:.3} sec to render\n", .{@as(f32, @floatFromInt(t1 - t0)) / 1000});
    switch (filetype) {
        .ppm => {
            try img_io.write_ppm_image(fname, .{ .arr = final_img, .nx = shape.nx, .ny = shape.ny });
        },
        else => {
            // return error.Unsupported_image_type;
        },
    }
}

// const list: [5]Sphere = .{
//     .{ .center = vec_type{ -1.0, 0.0, -1.0 }, .radius = 0.5, .material = .{ .lambertian = .{ .albedo = .{ 0.8, 0.3, 0.3 } } } },
//     .{ .center = vec_type{ 0.0, -1000.5, -1.0 }, .radius = 1000, .material = .{ .lambertian = .{ .albedo = .{ 0.5, 0.5, 0.5 } } } },
//     .{ .center = vec_type{ 1.0, 0.0, -1.0 }, .radius = 0.5, .material = .{ .metal = .{ .albedo = .{ 0.8, 0.6, 0.2 }, .roughness = 0.15 } } },
//     // .{ .center = vec_type{ -1.0, 0.0, -1.0 }, .radius = 0.5, .material = .{ .metal = .{ .albedo = .{ 0.8, 0.8, 0.8 }, .roughness = 0.0 } } },
//     .{ .center = vec_type{ 0.0, 0.0, -1.0 }, .radius = 0.5, .material = .{ .dielectric = .{ .color = .{ 0.95, 0.95, 0.95 }, .index = 5.5 } } },
//     .{ .center = vec_type{ 0.0, 0.0, -1.0 }, .radius = -0.4, .material = .{ .dielectric = .{ .color = .{ 0.95, 0.95, 0.95 }, .index = 5.5 } } },
// };

pub fn Random_Scene(half_size: usize, rng: std.Random, allocator: std.mem.Allocator) !Hittable {
    var list = allocator.create(std.ArrayList(Sphere)) catch unreachable;
    list.* = std.ArrayList(Sphere).init(allocator);
    // var list = std.MultiArrayList(Sphere);
    for (0..2 * half_size) |A| {
        for (0..2 * half_size) |B| {
            const a: f32 = @as(f32, @floatFromInt(A)) - @as(f32, @floatFromInt(half_size));
            const b: f32 = @as(f32, @floatFromInt(B)) - @as(f32, @floatFromInt(half_size));
            const choose_mat: u8 = @intFromFloat(rng.float(f32) * 100);
            const center: vec_type = .{ a + 0.9 * rng.float(f32), 0.2, b + 0.9 * rng.float(f32) };

            if (vec3f.length(center - vec_type{ 4.0, 0.2, 0.0 }) > 0.9 and (rng.float(f32) > 0.4)) {
                const sphere: Sphere = switch (choose_mat) {
                    0...50 => blk: { // Lambertian
                        const rnd_vec = random_in_unit_sphere(rng);
                        const normed_vec = vec3f.normalize(rnd_vec * rnd_vec);
                        check_neg3(normed_vec);
                        const albedo = normed_vec * vec3f.Splat(std.math.sqrt(std.math.pow(f32, rng.float(f32), 2)) * 0.5 + 0.5);
                        check_neg3(albedo);
                        break :blk .{ .center = center, .radius = 0.2, .material = .{ .lambertian = .{ .albedo = albedo } } };
                    },
                    51...80 => blk: { // Metal
                        const rnd_vec = random_in_unit_sphere(rng);
                        const normed_vec = vec3f.normalize(rnd_vec * rnd_vec);
                        check_neg3(normed_vec);
                        const albedo = normed_vec * vec3f.Splat(std.math.sqrt(std.math.pow(f32, rng.float(f32), 2)) * 0.5 + 0.5);
                        check_neg3(albedo);
                        break :blk .{ .center = center, .radius = 0.2, .material = .{ .metal = .{ .albedo = albedo, .roughness = 0.3 * rng.float(f32) } } };
                    },
                    81...90 => // Glass
                    .{ .center = center, .radius = 0.2, .material = .{ .dielectric = .{ .color = vec3f.Splat(0.95), .index = 1.5 + 8.5 * rng.float(f32) } } },
                    else => blk: { // Glass
                        const rnd_index = rng.float(f32);
                        const inside: Sphere = .{ .center = center, .radius = -0.15, .material = .{ .dielectric = .{ .color = vec3f.Splat(0.95), .index = 1.5 + 3.5 * rnd_index } } };
                        try list.append(inside);
                        break :blk .{ .center = center, .radius = 0.2, .material = .{ .dielectric = .{ .color = vec3f.Splat(0.95), .index = 1.5 + 3.5 * rnd_index } } };
                    },
                };
                try list.append(sphere);
            }
        }
    }
    try list.append(.{ .center = vec_type{ 0.0, -1000, 0.0 }, .radius = 1000, .material = .{ .lambertian = .{ .albedo = .{ 0.5, 0.5, 0.4 } } } });
    try list.append(.{ .center = vec_type{ 0.0, 1.0, 0.0 }, .radius = 1, .material = .{ .dielectric = .{ .color = .{ 1.0, 1.0, 1.0 }, .index = 1.5 } } });
    try list.append(.{ .center = vec_type{ 4.0, 1.0, 0.0 }, .radius = 1, .material = .{ .metal = .{ .albedo = .{ 0.7, 0.6, 0.5 }, .roughness = 0.0 } } });
    try list.append(.{ .center = vec_type{ -4.0, 1.0, 0.0 }, .radius = 1, .material = .{ .lambertian = .{ .albedo = .{ 0.4, 0.2, 0.1 } } } });

    var World: Hittable = undefined;

    if (Hittable == HittableBVH([]const Sphere)) {
        var inds = std.ArrayList(usize).init(allocator);
        defer inds.deinit();

        for (0..list.items.len) |i| {
            try inds.append(i);
        }

        const BVH = build_BVH(list, inds.items, 5, allocator);

        World = Hittable{ .objects = list.items, .bvh = BVH };
    } else {
        World = Hittable{ .objects = list.items };
    }
    return World;
}
var cam = cam_types.MovableCamera{};
const Hittable = HittableBVH([]const Sphere);
// const Hittable = HittableList([]const Sphere);
const num_samples = 256 / 8;

pub fn main() !void {
    // comptime {
    //     @compileLog("Size of Sphere: ", @sizeOf(Sphere));
    // }
    const filetype = .ppm;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const nx = 1024;
    const ny = 512;
    var arena = std.heap.ArenaAllocator.init(allocator);
    cam.SetPosition(.{ 13.0, 2.0, 3.5 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 1.0, 0.0 }, 20, @as(f32, @floatFromInt(nx)) / @as(f32, @floatFromInt(ny)));
    var rng = RndGen.init(0);

    // const World = Hittable{ .objects = list[0..] };
    const World = try Random_Scene(11, rng.random(), arena.allocator());
    log.info("Creating {} x {} image", .{ nx, ny });
    try render_and_write("Scene.ppm", World, .{ .nx = nx, .ny = ny }, filetype, arena.allocator());
}
