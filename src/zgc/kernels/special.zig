const line = @import("line.zig");

/// Numerically stable softmax over one logical tensor axis.
pub fn softmax(input: anytype, output: anytype, comptime softmax_axis: i8) void {
    const axis: usize = @intCast(softmax_axis);
    const extent = input.shape[axis];
    if (extent == 0) return;

    const slice_count = input.len() / extent;

    for (0..slice_count) |slice_index| {
        const input_line = input.axisSlice(axis, slice_index);
        const output_line = output.axisSlice(axis, slice_index);
        const maximum = line.max(input_line);
        const denominator = line.shiftedExpSum(input_line, maximum);
        line.normalizedExp(input_line, output_line, maximum, denominator);
    }
}
