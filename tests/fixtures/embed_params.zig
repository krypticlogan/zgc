const std = @import("std");

const w1 = [_]f32{
    2,   -1,
    0.5, 3,
};
const b1 = [_]f32{ 1, -2 };

pub const weights = [_][]const u8{
    std.mem.asBytes(&w1),
};

pub const biases = [_][]const u8{
    std.mem.asBytes(&b1),
};
