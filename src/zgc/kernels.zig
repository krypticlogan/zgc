const Op = @import("op.zig").Op;
const contraction = @import("kernels/contraction.zig");
const concatenation = @import("kernels/concatenation.zig");
const elementwise = @import("kernels/elementwise.zig");
const materialization = @import("kernels/materialization.zig");
const padding = @import("kernels/padding.zig");
const predicate = @import("kernels/predicate.zig");
const reduction = @import("kernels/reduction.zig");
const shifting = @import("kernels/shifting.zig");
const special = @import("kernels/special.zig");

/// Route graph operations to a kernel family.
pub fn execute(comptime op: Op.Compute, inputs: anytype, output: anytype) void {
    switch (op) {
        .relu => elementwise.relu(inputs[0], output),
        .exp => elementwise.exp(inputs[0], output),
        .neg => elementwise.neg(inputs[0], output),
        .abs => elementwise.abs(inputs[0], output),
        .sqrt => elementwise.sqrt(inputs[0], output),
        .log => elementwise.log(inputs[0], output),
        .reciprocal => elementwise.reciprocal(inputs[0], output),
        .add => elementwise.add(inputs[0], inputs[1], output),
        .sub => elementwise.sub(inputs[0], inputs[1], output),
        .mul => elementwise.mul(inputs[0], inputs[1], output),
        .div => elementwise.div(inputs[0], inputs[1], output),
        .minimum => elementwise.minimum(inputs[0], inputs[1], output),
        .maximum => elementwise.maximum(inputs[0], inputs[1], output),
        .clamp => elementwise.clamp(inputs[0], inputs[1], inputs[2], output),
        .equal => predicate.equal(inputs[0], inputs[1], output),
        .not_equal => predicate.notEqual(inputs[0], inputs[1], output),
        .less_than => predicate.lessThan(inputs[0], inputs[1], output),
        .less_equal => predicate.lessEqual(inputs[0], inputs[1], output),
        .greater_than => predicate.greaterThan(inputs[0], inputs[1], output),
        .greater_equal => predicate.greaterEqual(inputs[0], inputs[1], output),
        .logical_not => predicate.logicalNot(inputs[0], output),
        .logical_and => predicate.logicalAnd(inputs[0], inputs[1], output),
        .logical_or => predicate.logicalOr(inputs[0], inputs[1], output),
        .where => predicate.where(inputs[0], inputs[1], inputs[2], output),
        .copy, .contiguous => materialization.copy(inputs[0], output),
        .pad => |attrs| padding.constant(inputs[0], inputs[1], output, attrs),
        .shift => |attrs| shifting.shift(inputs, output, attrs),
        .matmul => |plan| contraction.matmulWithPlan(
            plan.strategy,
            inputs[0],
            inputs[1],
            output,
        ),
        .sum => |attrs| reduction.sum(inputs[0], output, attrs),
        .mean => |attrs| reduction.mean(inputs[0], output, attrs),
        .min => |attrs| reduction.min(inputs[0], output, attrs),
        .max => |attrs| reduction.max(inputs[0], output, attrs),
        .concat => |attrs| concatenation.concat(inputs, output, attrs.axis),
        .softmax => |attrs| special.softmax(inputs[0], output, attrs.axis),
    }
}
