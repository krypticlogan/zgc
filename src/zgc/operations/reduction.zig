pub const Kind = enum { sum, mean, minimum, maximum };
pub const Combine = enum { sum, minimum, maximum };
pub const Finalize = enum { identity, mean };

pub const Descriptor = struct {
    kind: Kind,
    axes: u64,
    keep_dims: bool,
};

pub fn fromCompute(compute: anytype) ?Descriptor {
    return switch (compute) {
        .sum => |attrs| .{ .kind = .sum, .axes = attrs.axes, .keep_dims = attrs.keep_dims },
        .mean => |attrs| .{ .kind = .mean, .axes = attrs.axes, .keep_dims = attrs.keep_dims },
        .min => |attrs| .{ .kind = .minimum, .axes = attrs.axes, .keep_dims = attrs.keep_dims },
        .max => |attrs| .{ .kind = .maximum, .axes = attrs.axes, .keep_dims = attrs.keep_dims },
        else => null,
    };
}
