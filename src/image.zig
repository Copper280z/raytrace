const std = @import("std");

const log_level: std.log.Level = .debug;
const log = std.log.scoped(.image);

pub const ImgType = enum {
    ppm,
    png,
    jpeg,
};

pub const Image_s = struct {
    arr: std.ArrayList(@Vector(3, u8)),
    ny: usize = 0,
    nx: usize = 0,
};

pub fn write_ppm_image(fname: []const u8, img: Image_s) !void {
    const fs = std.fs;
    const file = try fs.cwd().createFile(
        fname,
        .{ .read = true },
    );
    defer file.close();
    var BW = std.io.bufferedWriter(file.writer());
    const writer = BW.writer(); // output was black image, don't know why
    // const writer = file.writer();
    try std.fmt.format(writer, "P3\n{} {}\n255\n", .{ img.nx, img.ny });

    var img_arr = img.arr;
    log.info("Starting file save", .{});
    for (try img_arr.toOwnedSlice()) |rgb_pix| {
        try std.fmt.format(writer, "{} {} {}\n", .{ rgb_pix[0], rgb_pix[1], rgb_pix[2] });
    }
    try BW.flush();
    log.info("Done", .{});
}
