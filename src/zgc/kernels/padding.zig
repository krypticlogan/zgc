const Op = @import("../operations/semantic.zig").Op;

/// Materialize constant padding. Shape, dtype, widths, and output layout are
/// fixed and validated before this kernel is instantiated.
pub fn constant(
    input: anytype,
    fill: anytype,
    output: anytype,
    comptime attrs: Op.Compute.PadAttrs,
) void {
    const fill_value = fill.get(.{});
    for (0..output.len()) |linear_index| {
        output.storage[output.elementOffsetFromLinear(linear_index)] = fill_value;
    }

    for (0..input.len()) |linear_index| {
        var remaining = linear_index;
        var output_offset: isize = @intCast(output.elementOffsetFromLinear(0));
        comptime var axis = @TypeOf(input).rank;
        inline while (axis > 0) {
            axis -= 1;
            const coordinate = remaining % input.shape[axis];
            remaining /= input.shape[axis];
            output_offset += @as(isize, @intCast(coordinate + attrs.before[axis])) *
                output.strides[axis];
        }
        output.storage[@intCast(output_offset)] =
            input.storage[input.elementOffsetFromLinear(linear_index)];
    }
}
