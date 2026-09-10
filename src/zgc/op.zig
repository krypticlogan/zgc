const std = @import("std");
const Tensor = @import("tensor.zig");
const Matmul = @import("matmul.zig");
const kernels = @import("kernels.zig");
const validation = @import("validation.zig");

/// Tensor operation classified by whether it computes new storage or creates
/// another view of existing storage.
pub const Op = union(enum) {
    compute: Compute,
    view: View,

    pub const Kind = enum { compute, view };

    pub const Compute = union(enum) {
        relu,
        exp,
        add,
        sub,
        mul,
        div,
        matmul: MatmulAttrs,
        sum: ReductionAttrs,
        mean: ReductionAttrs,
        min: ReductionAttrs,
        max: ReductionAttrs,
        concat: ConcatAttrs,
        softmax: SoftmaxAttrs,

        pub const ReductionAttrs = struct {
            axes: u64,
            keep_dims: bool = false,
        };
        pub const SoftmaxAttrs = struct { axis: i8 };
        pub const ConcatAttrs = struct { axis: i8 };
        pub const MatmulAttrs = Matmul.Plan;

        pub fn execute(
            comptime op: Compute,
            inputs: anytype,
            output: anytype,
        ) void {
            kernels.execute(op, inputs, output);
        }

        pub fn inferRank(op: Compute, inputs: anytype) usize {
            return switch (op) {
                .relu => inferUnaryRank("relu", inputs),
                .exp => inferFloatUnaryRank("exp", inputs),
                .add => inferAddRank(inputs),
                .sub => inferBinaryElementwiseRank("sub", inputs),
                .mul => inferBinaryElementwiseRank("mul", inputs),
                .div => blk: {
                    const rank = inferBinaryElementwiseRank("div", inputs);
                    validation.requireDtypeKind("div", inputs[0], .float);
                    break :blk rank;
                },
                .matmul => inferMatmulRank(inputs),
                .sum => |attrs| inferReductionRank("sum", inputs, attrs),
                .mean => |attrs| blk: {
                    validation.requireDtypeKind("mean", inputs[0], .float);
                    break :blk inferReductionRank("mean", inputs, attrs);
                },
                .min => |attrs| inferReductionRank("min", inputs, attrs),
                .max => |attrs| inferReductionRank("max", inputs, attrs),
                .concat => |attrs| inferConcatRank(inputs, attrs.axis),
                .softmax => |attrs| blk: {
                    const rank = inferFloatUnaryRank("softmax", inputs);
                    validation.requireAxis("softmax", inputs[0], attrs.axis);
                    break :blk rank;
                },
            };
        }

        pub fn inferShape(
            comptime op: Compute,
            comptime inputs: anytype,
            comptime max_rank: usize,
        ) Tensor.Shape(max_rank) {
            return switch (op) {
                .relu => inferUnaryShape("relu", inputs, max_rank),
                .exp => blk: {
                    _ = inferFloatUnaryRank("exp", inputs);
                    break :blk inferUnaryShape("exp", inputs, max_rank);
                },
                .add => inferAddShape(inputs, max_rank),
                .sub => inferBinaryElementwiseShape("sub", inputs, max_rank),
                .mul => inferBinaryElementwiseShape("mul", inputs, max_rank),
                .div => blk: {
                    validation.requireDtypeKind("div", inputs[0], .float);
                    break :blk inferBinaryElementwiseShape("div", inputs, max_rank);
                },
                .matmul => inferMatmulShape(inputs, max_rank),
                .sum => |attrs| inferReductionShape("sum", inputs, attrs, max_rank),
                .mean => |attrs| blk: {
                    validation.requireDtypeKind("mean", inputs[0], .float);
                    break :blk inferReductionShape("mean", inputs, attrs, max_rank);
                },
                .min => |attrs| inferReductionShape("min", inputs, attrs, max_rank),
                .max => |attrs| inferReductionShape("max", inputs, attrs, max_rank),
                .concat => |attrs| inferConcatShape(inputs, attrs.axis, max_rank),
                .softmax => |attrs| blk: {
                    _ = inferFloatUnaryRank("softmax", inputs);
                    const shape = inferUnaryShape("softmax", inputs, max_rank);
                    validation.requireAxis("softmax", inputs[0], attrs.axis);
                    break :blk shape;
                },
            };
        }
    };

    pub const View = union(enum) {
        transpose: TransposeAttrs,
        reshape,
        flatten: FlattenAttrs,
        squeeze: AxisAttrs,
        unsqueeze: AxisAttrs,
        slice: SliceAttrs,
        broadcast,

        pub const TransposeAttrs = struct { axis_a: i8, axis_b: i8 };
        pub const FlattenAttrs = struct { start_axis: i8, end_axis: i8 };
        pub const AxisAttrs = struct { axis: i8 };
        pub const SliceAttrs = struct {
            axis: i8,
            start: usize,
            length: usize,
            step: usize,
        };
    };

    pub fn kind(op: Op) Kind {
        return switch (op) {
            .compute => .compute,
            .view => .view,
        };
    }

    pub fn execute(comptime op: Op, inputs: anytype, output: anytype) void {
        switch (op) {
            .compute => |compute| compute.execute(inputs, output),
            .view => @compileError("view operations do not execute a runtime kernel"),
        }
    }
};

fn inferUnaryRank(comptime operation: []const u8, inputs: anytype) usize {
    validation.requireInputCount(operation, inputs, 1);
    return validation.rankOf(inputs[0]);
}

fn inferUnaryShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    validation.requireInputCount(operation, inputs, 1);
    return inputs[0].shape;
}

fn inferFloatUnaryRank(comptime operation: []const u8, inputs: anytype) usize {
    const rank = inferUnaryRank(operation, inputs);
    validation.requireDtypeKind(operation, inputs[0], .float);
    return rank;
}

fn inferReductionRank(
    comptime operation: []const u8,
    inputs: anytype,
    comptime attrs: Op.Compute.ReductionAttrs,
) usize {
    const rank = inferUnaryRank(operation, inputs);
    validation.requireReductionAxes(operation, inputs[0], attrs.axes);
    return if (attrs.keep_dims) rank else rank - @popCount(attrs.axes);
}

fn inferReductionShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime attrs: Op.Compute.ReductionAttrs,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    _ = inferReductionRank(operation, inputs, attrs);
    var shape = Tensor.Shape(max_rank){ .rank = 0, .dims = @splat(0) };
    for (inputs[0].shape.slice(), 0..) |extent, axis| {
        const reduced = attrs.axes & (@as(u64, 1) << @intCast(axis)) != 0;
        if (reduced and !attrs.keep_dims) continue;
        shape.dims[shape.rank] = if (reduced) 1 else extent;
        shape.rank += 1;
    }
    return shape;
}

fn inferBinaryElementwiseRank(
    comptime operation: []const u8,
    inputs: anytype,
) usize {
    validation.requireInputCount(operation, inputs, 2);
    validation.requireMatchingDtypes(operation, inputs);
    return @max(validation.rankOf(inputs[0]), validation.rankOf(inputs[1]));
}

fn inferConcatRank(inputs: anytype, comptime axis: i8) usize {
    if (inputs.len == 0) @compileError("concat requires at least one input");
    validation.requireMatchingRanks("concat", inputs);
    validation.requireMatchingDtypes("concat", inputs);
    validation.requireAxis("concat", inputs[0], axis);
    return validation.rankOf(inputs[0]);
}

fn inferConcatShape(
    comptime inputs: anytype,
    comptime axis: i8,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    const rank = inferConcatRank(inputs, axis);
    const concat_axis: usize = @intCast(axis);
    var result = inputs[0].shape;
    var concat_extent: usize = 0;
    for (inputs) |input| {
        for (0..rank) |current_axis| {
            if (current_axis == concat_axis) continue;
            if (input.shape.at(current_axis) != inputs[0].shape.at(current_axis)) {
                @compileError("concat input extents must match outside the concatenation axis");
            }
        }
        concat_extent = std.math.add(usize, concat_extent, input.shape.at(concat_axis)) catch
            @compileError("concat axis extent exceeds usize");
    }
    result.dims[concat_axis] = concat_extent;
    return result;
}

fn inferBinaryElementwiseShape(
    comptime operation: []const u8,
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    const result_rank = inferBinaryElementwiseRank(operation, inputs);
    const lhs_shape = inputs[0].shape.slice();
    const rhs_shape = inputs[1].shape.slice();
    var result = Tensor.Shape(max_rank){
        .rank = result_rank,
        .dims = @splat(0),
    };

    for (0..result_rank) |axis_from_end| {
        const lhs_extent = if (axis_from_end < lhs_shape.len)
            lhs_shape[lhs_shape.len - 1 - axis_from_end]
        else
            1;
        const rhs_extent = if (axis_from_end < rhs_shape.len)
            rhs_shape[rhs_shape.len - 1 - axis_from_end]
        else
            1;

        if (!validation.extentsBroadcast(lhs_extent, rhs_extent)) {
            @compileError(std.fmt.comptimePrint(
                "{s} cannot broadcast extents {d} and {d} at aligned axis {d}",
                .{ operation, lhs_extent, rhs_extent, result_rank - 1 - axis_from_end },
            ));
        }

        result.dims[result_rank - 1 - axis_from_end] =
            if (lhs_extent == 1) rhs_extent else lhs_extent;
    }
    return result;
}

fn inferAddRank(inputs: anytype) usize {
    return inferBinaryElementwiseRank("add", inputs);
}

fn inferAddShape(
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    return inferBinaryElementwiseShape("add", inputs, max_rank);
}

fn inferMatmulRank(inputs: anytype) usize {
    validation.requireInputCount("matmul", inputs, 2);
    validation.requireRanks("matmul", inputs, &.{ 2, 2 });
    validation.requireMatchingDtypes("matmul", inputs);
    validation.requireDtype("matmul", inputs[0], .f32);
    return 2;
}

fn inferMatmulShape(
    comptime inputs: anytype,
    comptime max_rank: usize,
) Tensor.Shape(max_rank) {
    _ = inferMatmulRank(inputs);
    validation.requireMatchingExtents("matmul", inputs[0], 1, inputs[1], 0);

    return Tensor.Shape(max_rank).init(&.{
        inputs[0].shape.at(0),
        inputs[1].shape.at(1),
    });
}
