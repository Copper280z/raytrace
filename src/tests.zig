const std = @import("std");
comptime {
    _ = @import("vector.zig");
}

test {
    std.testing.refAllDecls(@This());
}
