const accumulation = @import("accumulation.zig");
const Dtype = @import("../dtype.zig").Dtype;
const Op = @import("../op.zig").Op;
const line = @import("line.zig");
const std = @import("std");

pub fn sum(input: anytype, output: anytype, comptime attrs: Op.Compute.ReductionAttrs) void {
    if (comptime @popCount(attrs.axes) == 1 and !attrs.keep_dims) {
        const axis: usize = @intCast(@ctz(attrs.axes));
        for (0..output.len()) |slice_index| {
            const result = line.sum(input.axisSlice(axis, slice_index));
            output.storage[output.elementOffsetFromLinear(slice_index)] =
                accumulation.narrowScalar(@TypeOf(input).dtype, result);
        }
        return;
    }
    reduce(input, output, attrs, Sum);
}

pub fn mean(input: anytype, output: anytype, comptime attrs: Op.Compute.ReductionAttrs) void {
    reduce(input, output, attrs, Mean);
}

pub fn min(input: anytype, output: anytype, comptime attrs: Op.Compute.ReductionAttrs) void {
    reduce(input, output, attrs, Min);
}

pub fn max(input: anytype, output: anytype, comptime attrs: Op.Compute.ReductionAttrs) void {
    reduce(input, output, attrs, Max);
}

fn reduce(input: anytype, output: anytype, comptime attrs: Op.Compute.ReductionAttrs, comptime Operator: type) void {
    const Input = @TypeOf(input);
    const dtype = Input.dtype;
    const reduction_count = reducedElementCount(input, attrs.axes);

    for (0..output.len()) |output_linear| {
        const base_offset = retainedOffset(input, output, output_linear, attrs);
        var accumulator = Operator.identity(dtype);
        for (0..reduction_count) |reduction_linear| {
            var remaining = reduction_linear;
            var input_offset: isize = @intCast(base_offset);
            comptime var axis = Input.rank;
            inline while (axis > 0) {
                axis -= 1;
                if (comptime axisIsReduced(attrs.axes, axis)) {
                    const coordinate = remaining % input.shape[axis];
                    remaining /= input.shape[axis];
                    input_offset += @as(isize, @intCast(coordinate)) * input.strides[axis];
                }
            }
            accumulator = Operator.combine(dtype, accumulator, input.storage[@intCast(input_offset)]);
        }
        output.storage[output.elementOffsetFromLinear(output_linear)] =
            Operator.finish(dtype, accumulator, reduction_count);
    }
}

fn retainedOffset(input: anytype, output: anytype, output_linear: usize, comptime attrs: Op.Compute.ReductionAttrs) usize {
    const Input = @TypeOf(input);
    var remaining = output_linear;
    var offset: isize = @intCast(input.elementOffsetFromLinear(0));
    comptime var input_axis = Input.rank;
    comptime var output_axis = @TypeOf(output).rank;
    inline while (input_axis > 0) {
        input_axis -= 1;
        if (comptime axisIsReduced(attrs.axes, input_axis)) {
            if (comptime attrs.keep_dims) output_axis -= 1;
            continue;
        }
        output_axis -= 1;
        const coordinate = remaining % output.shape[output_axis];
        remaining /= output.shape[output_axis];
        offset += @as(isize, @intCast(coordinate)) * input.strides[input_axis];
    }
    return @intCast(offset);
}

fn reducedElementCount(input: anytype, comptime axes: u64) usize {
    const Input = @TypeOf(input);
    var count: usize = 1;
    comptime var axis = 0;
    inline while (axis < Input.rank) : (axis += 1) {
        if (comptime axisIsReduced(axes, axis)) count *= input.shape[axis];
    }
    return count;
}

fn axisIsReduced(comptime axes: u64, comptime axis: usize) bool {
    return axes & (@as(u64, 1) << @intCast(axis)) != 0;
}

const Sum = struct {
    fn identity(comptime dtype: Dtype) accumulation.AccumulatorScalar(dtype) {
        return 0;
    }
    fn combine(comptime dtype: Dtype, accumulator: accumulation.AccumulatorScalar(dtype), value: dtype.Scalar()) accumulation.AccumulatorScalar(dtype) {
        return accumulator + accumulation.widenScalar(dtype, value);
    }
    fn finish(comptime dtype: Dtype, accumulator: accumulation.AccumulatorScalar(dtype), _: usize) dtype.Scalar() {
        return accumulation.narrowScalar(dtype, accumulator);
    }
};

const Mean = struct {
    fn identity(comptime dtype: Dtype) accumulation.AccumulatorScalar(dtype) {
        return Sum.identity(dtype);
    }
    fn combine(comptime dtype: Dtype, accumulator: accumulation.AccumulatorScalar(dtype), value: dtype.Scalar()) accumulation.AccumulatorScalar(dtype) {
        return Sum.combine(dtype, accumulator, value);
    }
    fn finish(comptime dtype: Dtype, accumulator: accumulation.AccumulatorScalar(dtype), count: usize) dtype.Scalar() {
        const divisor: accumulation.AccumulatorScalar(dtype) = @floatFromInt(count);
        return accumulation.narrowScalar(dtype, accumulator / divisor);
    }
};

const Min = struct {
    fn identity(comptime dtype: Dtype) accumulation.AccumulatorScalar(dtype) {
        return switch (comptime dtype.kind()) {
            .float => std.math.inf(accumulation.AccumulatorScalar(dtype)),
            .signed_integer => std.math.maxInt(accumulation.AccumulatorScalar(dtype)),
            .boolean => @compileError("boolean tensors cannot be reduced with min"),
        };
    }
    fn combine(comptime dtype: Dtype, accumulator: accumulation.AccumulatorScalar(dtype), value: dtype.Scalar()) accumulation.AccumulatorScalar(dtype) {
        return @min(accumulator, accumulation.widenScalar(dtype, value));
    }
    fn finish(comptime dtype: Dtype, accumulator: accumulation.AccumulatorScalar(dtype), _: usize) dtype.Scalar() {
        return accumulation.narrowScalar(dtype, accumulator);
    }
};

const Max = struct {
    fn identity(comptime dtype: Dtype) accumulation.AccumulatorScalar(dtype) {
        return switch (comptime dtype.kind()) {
            .float => -std.math.inf(accumulation.AccumulatorScalar(dtype)),
            .signed_integer => std.math.minInt(accumulation.AccumulatorScalar(dtype)),
            .boolean => @compileError("boolean tensors cannot be reduced with max"),
        };
    }
    fn combine(comptime dtype: Dtype, accumulator: accumulation.AccumulatorScalar(dtype), value: dtype.Scalar()) accumulation.AccumulatorScalar(dtype) {
        return @max(accumulator, accumulation.widenScalar(dtype, value));
    }
    fn finish(comptime dtype: Dtype, accumulator: accumulation.AccumulatorScalar(dtype), _: usize) dtype.Scalar() {
        return accumulation.narrowScalar(dtype, accumulator);
    }
};
