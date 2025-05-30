const std = @import("std");

// const vec_type = @Vector(3, f32);
pub fn Vector(vec_type: type) type {
    return struct {
        pub inline fn sum(v: vec_type) @TypeOf(v[0]) {
            return @reduce(.Add, v);
        }

        pub inline fn length(v: vec_type) @TypeOf(v[0]) {
            const v_out = std.math.sqrt(length_squared(v));
            return v_out;
        }
        pub inline fn length_squared(v: vec_type) @TypeOf(v[0]) {
            const v_out = sum(v * v);
            return v_out;
        }
        pub inline fn normalize(v: vec_type) @TypeOf(v) {
            return v / Splat(length(v));
        }

        pub fn dot(a: vec_type, b: vec_type) @TypeOf(a[0]) {
            return sum(a * b);
        }

        pub inline fn cross(a: vec_type, b: vec_type) @TypeOf(a) {
            const elem_type = @TypeOf(a[0]);
            return @Vector(3, elem_type){
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            };
        }
        pub inline fn Splat(val: anytype) vec_type {
            return @as(vec_type, @splat(val));
        }
        pub inline fn elem_sqrt(v: vec_type) vec_type {
            return vec_type{ std.math.sqrt(v[0]), std.math.sqrt(v[1]), std.math.sqrt(v[2]) };
        }
        pub fn argmax(v: vec_type) u2 {
            var arg_max: u2 = undefined;
            if (v[0] > v[1]) {
                if (v[0] > v[2]) {
                    arg_max = 0;
                } else {
                    arg_max = 2;
                }
            } else {
                if (v[1] > v[2]) {
                    arg_max = 1;
                } else {
                    arg_max = 2;
                }
            }
            return arg_max;
        }
    };
}

test "vec length" {
    const vec_type = @Vector(3, f32);
    const vec3f = Vector(vec_type);
    const rt2_2 = std.math.sqrt(2.0) / 2.0;
    const vec: @Vector(3, f32) = .{ rt2_2, rt2_2, 0 };
    const len = vec3f.length(vec);
    try std.testing.expect(std.math.approxEqAbs(f32, len, 1.0, 0.000001));
}
test "vec normalize" {
    const vec_type = @Vector(3, f32);
    const vec3f = Vector(vec_type);
    const rt2_2 = std.math.sqrt(2.0) / 2.0;
    const expected: vec_type = .{ rt2_2, 0, rt2_2 };
    const vec: vec_type = .{ 10.0, 0, 10.0 };
    const normed = vec3f.normalize(vec);

    const len = @typeInfo(vec_type).Vector.len;
    inline for (0..len) |i| {
        try std.testing.expect(std.math.approxEqAbs(f32, normed[i], expected[i], 0.000001));
    }
}
test "vec dot" {
    const vec_type = @Vector(3, f32);
    const vec3f = Vector(vec_type);
    const a: vec_type = .{ 1.0, 2.0, 3.0 };
    const b: vec_type = .{ 6.0, 5.0, 4.0 };
    const dot_prod = vec3f.dot(a, b);
    try std.testing.expect(std.math.approxEqAbs(f32, dot_prod, 28.0, 0.000001));
}

test "vec cross" {
    const vec_type = @Vector(3, f32);
    const vec3f = Vector(vec_type);
    const a: vec_type = .{ 1.0, 2.0, 3.0 };
    const b: vec_type = .{ 6.0, 5.0, 4.0 };
    const cross_prod = vec3f.cross(a, b);
    const expected: vec_type = .{ -7.0, 14.0, -7.0 };

    const len = @typeInfo(vec_type).Vector.len;
    inline for (0..len) |i| {
        try std.testing.expect(std.math.approxEqAbs(f32, cross_prod[i], expected[i], 0.000001));
    }
}
