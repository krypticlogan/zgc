const accumulation = @import("../accumulation.zig");
const Dtype = @import("../../dtype.zig").Dtype;
const ReductionPlan = @import("../../execution/kernel_plan.zig").ReductionPlan;
const Expression = @import("../../optimization/fusion/expression.zig").Program;
const elementwise = @import("../elementwise_operation.zig");

/// Execute one statically planned multi-accumulator reduction region.
pub fn execute(comptime plan: ReductionPlan, inputs: anytype, outputs: anytype) void {
    if (outputs.len == 0) @compileError("reduction plan requires an output");
    const Output = @TypeOf(outputs[0]);
    const dtype = Output.dtype;
    const Accumulator = accumulation.AccumulatorScalar(dtype);
    const rank = plan.region.domain_shape.len;
    const domain_shape: [rank]usize = plan.region.domain_shape[0..rank].*;
    const reduction_count = comptime reducedElementCount(domain_shape, plan.region.reduction_axes);
    const output_count = outputs[0].len();

    for (0..output_count) |output_linear| {
        var accumulators: [plan.region.accumulators.len]Accumulator = undefined;
        inline for (plan.region.accumulators, 0..) |accumulator, index| {
            accumulators[index] = accumulation.identity(dtype, accumulator.combine);
        }

        for (0..reduction_count) |reduction_linear| {
            const coordinates = domainCoordinates(
                domain_shape,
                plan.region.reduction_axes,
                output_linear,
                reduction_linear,
            );
            var values: [plan.region.expressions.instructions.len]dtype.Scalar() = undefined;
            inline for (plan.region.expressions.instructions, 0..) |instruction, instruction_index| {
                var params: [instruction.operation.arity()]dtype.Scalar() = undefined;
                inline for (instruction.args[0..params.len], 0..) |reference, param_index| {
                    params[param_index] = resolve(
                        dtype,
                        rank,
                        domain_shape,
                        reference,
                        inputs,
                        &values,
                        coordinates,
                    );
                }
                values[instruction_index] = elementwise.evaluateScalar(dtype, instruction.operation, params);
            }

            inline for (plan.region.accumulators, 0..) |accumulator, accumulator_index| {
                const value = resolve(
                    dtype,
                    rank,
                    domain_shape,
                    accumulator.update,
                    inputs,
                    &values,
                    coordinates,
                );
                accumulators[accumulator_index] = accumulation.combine(
                    dtype,
                    accumulator.combine,
                    accumulators[accumulator_index],
                    value,
                );
            }
        }

        inline for (plan.region.stores) |store| {
            const accumulator_index = switch (store.value) {
                .accumulator => |index| index,
                .input, .instruction => @compileError("reduction stores must reference an accumulator"),
            };
            const accumulator = plan.region.accumulators[accumulator_index];
            const value = accumulation.finish(
                dtype,
                accumulator.finalize,
                accumulators[accumulator_index],
                reduction_count,
            );
            outputs[store.output].storage[outputs[store.output].elementOffsetFromLinear(output_linear)] = value;
        }
    }
}

fn resolve(
    comptime dtype: Dtype,
    comptime rank: usize,
    comptime domain_shape: [rank]usize,
    comptime reference: Expression.ValueRef,
    inputs: anytype,
    values: anytype,
    coordinates: [rank]usize,
) dtype.Scalar() {
    return switch (reference) {
        .input => |input_index| blk: {
            const view = inputs[input_index].broadcastTo(rank, domain_shape);
            break :blk view.storage[view.elementOffset(coordinates)];
        },
        .instruction => |instruction_index| values[instruction_index],
        .accumulator => @compileError("reduction body expressions cannot read accumulators"),
    };
}

fn reducedElementCount(comptime shape: anytype, comptime axes: u64) usize {
    var count: usize = 1;
    for (shape, 0..) |extent, axis| {
        if (axes & (@as(u64, 1) << @intCast(axis)) != 0) count *= extent;
    }
    return count;
}

fn domainCoordinates(
    comptime shape: anytype,
    comptime reduction_axes: u64,
    output_linear: usize,
    reduction_linear: usize,
) [shape.len]usize {
    var coordinates: [shape.len]usize = @splat(0);
    var outer_remaining = output_linear;
    var reduction_remaining = reduction_linear;
    var axis = shape.len;
    while (axis > 0) {
        axis -= 1;
        if (reduction_axes & (@as(u64, 1) << @intCast(axis)) != 0) {
            coordinates[axis] = reduction_remaining % shape[axis];
            reduction_remaining /= shape[axis];
        } else {
            coordinates[axis] = outer_remaining % shape[axis];
            outer_remaining /= shape[axis];
        }
    }
    return coordinates;
}
