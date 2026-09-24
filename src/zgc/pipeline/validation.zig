const std = @import("std");
const Graph = @import("../graph.zig");
const Op = @import("../operations/semantic.zig").Op;
const Tensor = @import("../tensor.zig");
const layout_ops = @import("../kernels/layout.zig");
const Plan = @import("../execution/kernel_plan.zig");
const fusion = @import("../optimization/fusion/expression.zig");
const matmul = @import("../optimization/matmul.zig");

/// Validates the constructed semantic graph before analysis.
pub fn Validation(comptime capacity: Graph.Capacity) type {
    return struct {
        pub fn validate(comptime lowered_graph: Graph.Graph(capacity, Op)) type {
            inline for (0..lowered_graph.node_ct) |node_id| {
                const node = lowered_graph.nodes[node_id].?;
                const output = lowered_graph.tensors[node.result].?;
                const Inputs = [node.input_count]Graph.Graph(capacity, Op).TensorInfo;
                var inputs: Inputs = undefined;
                inline for (0..node.input_count) |input_index| {
                    const tensor_id = lowered_graph.input_refs[node.input_start + input_index].?;
                    inputs[input_index] = lowered_graph.tensors[tensor_id].?;
                }

                switch (node.op) {
                    .view => |view| {
                        const expected = layout_ops.infer(view, &inputs, output.shape, capacity.max_rank);
                        if (!std.mem.eql(usize, expected.shape.slice(), output.shape.slice()) or
                            expected.layout.offset != output.layout.offset or
                            !std.mem.eql(
                                isize,
                                expected.layout.strides[0..output.shape.rank],
                                output.layout.strides[0..output.shape.rank],
                            ) or expected.storage_tensor != output.storage_tensor)
                        {
                            @compileError("lowered view metadata does not match its inferred alias");
                        }
                        if (output.dtype != inputs[0].dtype) {
                            @compileError("view output dtype does not match its input dtype");
                        }
                    },
                    .compute => |compute| {
                        const expected_shape = compute.inferShape(&inputs, capacity.max_rank);
                        if (!std.mem.eql(usize, expected_shape.slice(), output.shape.slice())) {
                            @compileError("lowered operation output shape does not match its inferred shape");
                        }
                        const expected_dtype = compute.inferDtype(&inputs);
                        if (output.dtype != expected_dtype) {
                            @compileError("lowered operation output dtype does not match its inferred dtype");
                        }
                    },
                }
            }
            return struct {
                pub const graph = lowered_graph;

                pub fn View(comptime tensor_id: Tensor.Id) type {
                    const info = tensorInfo(tensor_id);
                    return Tensor.StaticView(
                        info.dtype.Scalar(),
                        info.shape.dims[0..info.shape.rank].*,
                        info.layout.strides[0..info.shape.rank].*,
                        info.layout.offset,
                    );
                }

                pub fn ConstView(comptime tensor_id: Tensor.Id) type {
                    const info = tensorInfo(tensor_id);
                    return Tensor.StaticConstView(
                        info.dtype.Scalar(),
                        info.shape.dims[0..info.shape.rank].*,
                        info.layout.strides[0..info.shape.rank].*,
                        info.layout.offset,
                    );
                }

                fn tensorInfo(comptime tensor_id: Tensor.Id) Graph.Graph(capacity, Op).TensorInfo {
                    if (tensor_id >= graph.tensor_ct) {
                        @compileError("tensor id is outside the validated graph");
                    }
                    return graph.tensors[tensor_id].?;
                }
            };
        }
    };
}

/// Validates optimizer output before lifetime analysis and model generation.
/// Failures here indicate an invalid compiler rewrite or execution plan.
pub fn FinalValidation(comptime capacity: Graph.Capacity) type {
    return struct {
        pub fn validate(comptime executable: anytype) type {
            const Program = @TypeOf(executable);
            inline for (0..executable.node_ct) |node_id| {
                const node = executable.nodes[node_id].?;
                const Inputs = [node.input_count]Program.TensorInfo;
                var inputs: Inputs = undefined;
                inline for (0..node.input_count) |input_index| {
                    const tensor_id = executable.input_refs[node.input_start + input_index].?;
                    inputs[input_index] = executable.tensors[tensor_id].?;
                }
                const Outputs = [node.output_count]Program.TensorInfo;
                var outputs: Outputs = undefined;
                inline for (0..node.output_count) |output_index| {
                    const tensor_id = executable.output_refs[node.output_start + output_index].?;
                    outputs[output_index] = executable.tensors[tensor_id].?;
                }

                switch (node.op) {
                    .view => |view| {
                        requireOutputCount(node.output_count, 1);
                        validateView(view, &inputs, outputs[0]);
                    },
                    .compute => |compute| switch (compute) {
                        .direct => |semantic| {
                            requireOutputCount(node.output_count, 1);
                            validateSemantic(semantic, &inputs, outputs[0]);
                        },
                        .kernel => |kernel_plan| switch (kernel_plan) {
                            .map => |plan| validateMapPlan(plan, &inputs, &outputs),
                            .reduction => |plan| validateReductionPlan(plan, &inputs, &outputs),
                            .contraction => |plan| {
                                requireOutputCount(node.output_count, 1);
                                validateSemantic(.matmul, &inputs, outputs[0]);
                                matmul.validate(plan, inputs[0], inputs[1], outputs[0]);
                            },
                        },
                    },
                }
            }

            return struct {
                pub const graph = executable;

                pub fn View(comptime tensor_id: Tensor.Id) type {
                    const info = tensorInfo(tensor_id);
                    return Tensor.StaticView(
                        info.dtype.Scalar(),
                        info.shape.dims[0..info.shape.rank].*,
                        info.layout.strides[0..info.shape.rank].*,
                        info.layout.offset,
                    );
                }

                pub fn ConstView(comptime tensor_id: Tensor.Id) type {
                    const info = tensorInfo(tensor_id);
                    return Tensor.StaticConstView(
                        info.dtype.Scalar(),
                        info.shape.dims[0..info.shape.rank].*,
                        info.layout.strides[0..info.shape.rank].*,
                        info.layout.offset,
                    );
                }

                fn tensorInfo(comptime tensor_id: Tensor.Id) Program.TensorInfo {
                    if (tensor_id >= graph.tensor_ct) @compileError("tensor id is outside the final validated graph");
                    return graph.tensors[tensor_id].?;
                }
            };
        }

        fn validateView(comptime view: Op.View, comptime inputs: anytype, comptime output: anytype) void {
            const expected = layout_ops.infer(view, inputs, output.shape, capacity.max_rank);
            if (!std.mem.eql(usize, expected.shape.slice(), output.shape.slice()) or
                expected.layout.offset != output.layout.offset or
                !std.mem.eql(isize, expected.layout.strides[0..output.shape.rank], output.layout.strides[0..output.shape.rank]) or
                expected.storage_tensor != output.storage_tensor)
            {
                @compileError("optimized view metadata does not match its inferred alias");
            }
            if (output.dtype != inputs[0].dtype) @compileError("optimized view output dtype does not match its input");
        }

        fn validateSemantic(comptime compute: Op.Compute, comptime inputs: anytype, comptime output: anytype) void {
            const expected_shape = compute.inferShape(inputs, capacity.max_rank);
            if (!std.mem.eql(usize, expected_shape.slice(), output.shape.slice())) {
                @compileError("optimized operation output shape does not match semantic inference");
            }
            if (output.dtype != compute.inferDtype(inputs)) {
                @compileError("optimized operation output dtype does not match semantic inference");
            }
        }

        fn requireOutputCount(comptime actual: usize, comptime expected: usize) void {
            if (actual != expected) @compileError("executable operation has an invalid output count");
        }

        fn validateMapPlan(comptime plan: Plan.MapPlan, comptime inputs: anytype, comptime outputs: anytype) void {
            if (plan.region.stores.len != outputs.len) @compileError("map stores must match invocation outputs");
            if (outputs.len != 1) @compileError("multi-store map execution is not implemented");
            validateElementwiseProgram(plan.region.expressions, inputs, outputs[0]);
            const store = plan.region.stores[0];
            if (store.output != 0) @compileError("single-output map store must target output zero");
            switch (store.value) {
                .instruction => |index| if (index != plan.region.expressions.instructions.len - 1) {
                    @compileError("map executor requires its store to reference the final instruction");
                },
                .input, .accumulator => @compileError("map store must reference an expression instruction"),
            }
            if (plan.traversal_plan.vector_width == 0 or plan.traversal_plan.unroll == 0) {
                @compileError("map traversal factors must be nonzero");
            }
            if (plan.traversal_plan.axis_order.len != outputs[0].shape.rank) {
                @compileError("map traversal plan must order every output axis");
            }
            var seen: [capacity.max_rank]bool = @splat(false);
            for (plan.traversal_plan.axis_order) |axis| {
                if (axis >= outputs[0].shape.rank or seen[axis]) @compileError("map axis order is invalid");
                seen[axis] = true;
            }
            if (plan.traversal_plan.vector_axis) |axis| {
                if (axis >= outputs[0].shape.rank) @compileError("map vector axis is outside the output rank");
            } else if (plan.traversal_plan.vector_width != 1) {
                @compileError("scalar map traversal plans must have vector width one");
            }
        }

        fn validateReductionPlan(comptime plan: Plan.ReductionPlan, comptime inputs: anytype, comptime outputs: anytype) void {
            const region = plan.region;
            const traversal_plan = plan.traversal_plan;
            const rank = region.domain_shape.len;

            if (outputs.len == 0) @compileError("reduction plan requires an output");
            if (plan.region.stores.len != outputs.len) @compileError("reduction stores must match invocation outputs");
            if (plan.region.accumulators.len == 0) @compileError("reduction plan requires an accumulator");
            if (rank > 64 or region.reduction_axes == 0 or
                (rank < 64 and region.reduction_axes >= (@as(u64, 1) << @intCast(rank))))
            {
                @compileError("reduction plan axes are outside its domain rank");
            }
            for (region.domain_shape) |extent| {
                if (extent == 0) @compileError("reduction domain extents must be nonzero");
            }

            const dtype = outputs[0].dtype;
            if (dtype.kind() == .boolean) @compileError("reduction plans require a numeric dtype");
            for (inputs) |input| {
                if (input.dtype != dtype) @compileError("reduction expression inputs must match the output dtype");
                if (!broadcastsToShape(input, region.domain_shape)) {
                    @compileError("reduction expression input does not broadcast to the reduction domain");
                }
            }
            for (outputs) |output| {
                if (output.dtype != dtype) @compileError("reduction outputs must have matching dtypes");
                if (!isReductionOutputShape(region.domain_shape, region.reduction_axes, region.keep_dims, output)) {
                    @compileError("reduction output shape does not match its domain and axes");
                }
            }

            validateReductionExpressions(region.expressions, inputs.len, dtype);

            for (region.accumulators) |accumulator| {
                validateReductionValueRef(
                    accumulator.update,
                    inputs.len,
                    region.expressions.instructions.len,
                    0,
                );
                if (accumulator.finalize == .mean and dtype.kind() != .float) {
                    @compileError("mean reduction finalization requires a floating-point dtype");
                }
            }

            var stored_outputs: [outputs.len]bool = @splat(false);
            var stored_accumulators: [region.accumulators.len]bool = @splat(false);
            for (region.stores) |store| {
                if (store.output >= outputs.len) @compileError("reduction store refers to an unknown output");
                if (stored_outputs[store.output]) @compileError("reduction output has more than one store");
                stored_outputs[store.output] = true;
                const accumulator_index = switch (store.value) {
                    .accumulator => |index| index,
                    .input, .instruction => @compileError("reduction stores must refer to accumulators"),
                };
                if (accumulator_index >= region.accumulators.len) {
                    @compileError("reduction store refers to an unknown accumulator");
                }
                stored_accumulators[accumulator_index] = true;
            }
            for (stored_outputs) |stored| {
                if (!stored) @compileError("reduction output is missing a store");
            }
            for (stored_accumulators) |stored| {
                if (!stored) @compileError("reduction plan contains an unused accumulator");
            }

            if (traversal_plan.vector_width == 0 or traversal_plan.accumulator_lanes == 0 or traversal_plan.unroll == 0) {
                @compileError("reduction traversal factors must be nonzero");
            }
            if (traversal_plan.vector_axis == null and traversal_plan.vector_width != 1) {
                @compileError("a scalar reduction traversal plan must have vector width one");
            }
            if (traversal_plan.vector_axis) |axis| {
                if (axis >= rank) @compileError("reduction vector axis is outside the domain rank");
                if (region.reduction_axes & (@as(u64, 1) << @intCast(axis)) == 0) {
                    @compileError("reduction vector axis must be a reduced axis");
                }
                if (region.domain_shape[axis] < traversal_plan.vector_width) {
                    @compileError("reduction vector axis is shorter than its vector width");
                }
                for (inputs) |input| {
                    if (!supportsReductionVectorAxis(input, region.domain_shape, axis)) {
                        @compileError("reduction input is neither contiguous nor broadcast on its vector axis");
                    }
                }
            }
            validateReductionAxisOrder(
                rank,
                region.reduction_axes,
                traversal_plan.outer_axis_order,
                traversal_plan.reduction_axis_order,
            );
        }

        fn supportsReductionVectorAxis(comptime input: anytype, comptime domain_shape: []const usize, comptime axis: usize) bool {
            const leading_axes = domain_shape.len - input.shape.rank;
            if (axis < leading_axes) return true;
            const input_axis = axis - leading_axes;
            if (input.shape.at(input_axis) == 1 and domain_shape[axis] != 1) return true;
            return input.layout.strides[input_axis] == 1;
        }

        fn validateReductionExpressions(
            comptime program: fusion.Program,
            comptime input_count: usize,
            comptime dtype: @import("../dtype.zig").Dtype,
        ) void {
            for (program.instructions, 0..) |instruction, instruction_index| {
                if (instruction.dtype != dtype) {
                    @compileError("reduction expression instruction dtype must match the reduction dtype");
                }
                if (!instruction.operation.acceptsDtype(dtype)) {
                    @compileError("reduction expression instruction does not accept the reduction dtype");
                }
                for (instruction.args[0..instruction.operation.arity()]) |reference| {
                    validateReductionValueRef(reference, input_count, instruction_index, 0);
                }
            }
        }

        fn validateReductionValueRef(
            comptime reference: fusion.Program.ValueRef,
            comptime input_count: usize,
            comptime instruction_limit: usize,
            comptime accumulator_count: usize,
        ) void {
            switch (reference) {
                .input => |index| if (index >= input_count) {
                    @compileError("reduction expression refers to an unknown input");
                },
                .instruction => |index| if (index >= instruction_limit) {
                    @compileError("reduction expression must refer only to prior instructions");
                },
                .accumulator => |index| if (index >= accumulator_count) {
                    @compileError("reduction expression cannot read an accumulator at this stage");
                },
            }
        }

        fn validateReductionAxisOrder(
            comptime rank: usize,
            comptime reduction_axes: u64,
            comptime outer_order: []const u8,
            comptime reduction_order: []const u8,
        ) void {
            if (outer_order.len + reduction_order.len != rank) {
                @compileError("reduction traversal axes must partition the domain");
            }
            var seen: [rank]bool = @splat(false);
            for (outer_order) |axis| {
                if (axis >= rank or seen[axis]) @compileError("reduction outer-axis order is invalid");
                if (reduction_axes & (@as(u64, 1) << @intCast(axis)) != 0) {
                    @compileError("reduction outer-axis order contains a reduced axis");
                }
                seen[axis] = true;
            }
            for (reduction_order) |axis| {
                if (axis >= rank or seen[axis]) @compileError("reduction axis order is invalid");
                if (reduction_axes & (@as(u64, 1) << @intCast(axis)) == 0) {
                    @compileError("reduction axis order contains a retained axis");
                }
                seen[axis] = true;
            }
        }

        fn isReductionOutputShape(
            comptime domain_shape: []const usize,
            comptime reduction_axes: u64,
            comptime keep_dims: bool,
            comptime output: anytype,
        ) bool {
            const expected_rank = if (keep_dims) domain_shape.len else domain_shape.len - @popCount(reduction_axes);
            if (output.shape.rank != expected_rank) return false;
            var output_axis: usize = 0;
            for (domain_shape, 0..) |extent, domain_axis| {
                const reduced = reduction_axes & (@as(u64, 1) << @intCast(domain_axis)) != 0;
                if (reduced and !keep_dims) continue;
                const expected_extent: usize = if (reduced) 1 else extent;
                if (output.shape.at(output_axis) != expected_extent) return false;
                output_axis += 1;
            }
            return true;
        }

        fn broadcastsToShape(comptime input: anytype, comptime shape: []const usize) bool {
            if (input.shape.rank > shape.len) return false;
            for (0..input.shape.rank) |axis_from_end| {
                const input_extent = input.shape.at(input.shape.rank - 1 - axis_from_end);
                const output_extent = shape[shape.len - 1 - axis_from_end];
                if (input_extent != 1 and input_extent != output_extent) return false;
            }
            return true;
        }

        fn validateElementwiseProgram(comptime program: fusion.Program, comptime inputs: anytype, comptime output: anytype) void {
            if (program.instructions.len == 0) @compileError("fused elementwise program must contain an instruction");
            for (inputs) |input| {
                if (!broadcastsTo(input, output)) @compileError("fused elementwise input does not broadcast to its output shape");
            }
            for (program.instructions, 0..) |instruction, instruction_index| {
                var operand_dtypes: [instruction.operation.arity()]@import("../dtype.zig").Dtype = undefined;
                for (instruction.args[0..instruction.operation.arity()], 0..) |reference, operand_index| {
                    switch (reference) {
                        .input => |input_index| {
                            if (input_index >= inputs.len) {
                                @compileError("fused elementwise instruction refers to an unknown input");
                            }
                            operand_dtypes[operand_index] = inputs[input_index].dtype;
                        },
                        .instruction => |prior| {
                            if (prior >= instruction_index) {
                                @compileError("fused elementwise instruction must refer only to prior instructions");
                            }
                            operand_dtypes[operand_index] = program.instructions[prior].dtype;
                        },
                        .accumulator => @compileError("map expressions cannot refer to accumulators"),
                    }
                }
                if (!instruction.operation.acceptsOperands(&operand_dtypes)) {
                    @compileError("fused pointwise instruction has invalid operand dtypes");
                }
                if (instruction.dtype != instruction.operation.inferDtype(&operand_dtypes)) {
                    @compileError("fused pointwise instruction result dtype is invalid");
                }
            }
            if (program.instructions[program.instructions.len - 1].dtype != output.dtype) {
                @compileError("fused pointwise program result dtype does not match its output");
            }
        }

        fn broadcastsTo(input: anytype, output: anytype) bool {
            if (input.shape.rank > output.shape.rank) return false;
            for (0..input.shape.rank) |axis_from_end| {
                const input_extent = input.shape.at(input.shape.rank - 1 - axis_from_end);
                const output_extent = output.shape.at(output.shape.rank - 1 - axis_from_end);
                if (input_extent != 1 and input_extent != output_extent) return false;
            }
            return true;
        }
    };
}
