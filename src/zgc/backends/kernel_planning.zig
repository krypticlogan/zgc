const std = @import("std");
const Execution = @import("../execution.zig");
const ExecutableProgram = @import("../execution/program.zig");
const Expression = @import("../optimization/fusion/expression.zig");
const Graph = @import("../graph.zig");
const Plan = @import("../execution/kernel_plan.zig");
const Region = @import("../optimization/fusion/region.zig");
const Semantic = @import("../operations/semantic.zig");
const Elementwise = @import("../operations/elementwise.zig");
const Reduction = @import("../operations/reduction.zig");
const matmul = @import("../optimization/matmul.zig");

/// Convert a layout-planned semantic graph and selected logical regions into
/// a sequential executable program with concrete, data-only kernel plans.
pub fn KernelPlanningBackend(comptime capacity: Graph.Capacity) type {
    return struct {
        pub fn plan(comptime Validated: type, comptime Fused: type) ExecutableProgram.Program(capacity, Execution.Op) {
            const source_graph = Validated.graph;
            var program: ExecutableProgram.Program(capacity, Execution.Op) = .init();

            for (0..source_graph.tensor_ct) |tensor_id| {
                const info = source_graph.tensors[tensor_id].?;
                const inserted = program.insertTensor(info);
                if (inserted != tensor_id) @compileError("kernel planning changed tensor order");
                program.materialized[tensor_id] = switch (info.origin) {
                    .source, .literal => true,
                    .node => false,
                };
            }
            for (0..source_graph.sources.len) |source_index| {
                if (source_graph.sources[source_index]) |source| program.insertSource(source_index, source);
            }

            inline for (0..source_graph.node_ct) |node_id| {
                if (Fused.regions.node_region[node_id]) |region| {
                    switch (region) {
                        .reduction => |region_id| {
                            const group = Fused.regions.reduction_storage[region_id].?;
                            if (node_id != group.emit_node) continue;
                            const Planned = PlannedReduction(source_graph, group);
                            insertInvocation(
                                &program,
                                .{ .compute = .{ .kernel = .{ .reduction = Planned.plan } } },
                                &Planned.input_ids,
                                &Planned.output_ids,
                            );
                        },
                        .map => |map_id| {
                            const group = Fused.regions.map_storage[map_id].?;
                            if (node_id != group.root_node) continue;
                            const Planned = PlannedMap(source_graph, group);
                            insertInvocation(
                                &program,
                                .{ .compute = .{ .kernel = .{ .map = Planned.plan } } },
                                &Planned.input_ids,
                                &Planned.output_ids,
                            );
                        },
                    }
                    continue;
                }

                const node = source_graph.nodes[node_id].?;
                const executable: Execution.Op = switch (node.op) {
                    .view => |view| .{ .view = view },
                    .compute => |compute| .{ .compute = planCompute(compute, &source_graph, node) },
                };
                var inputs: [node.input_count]usize = undefined;
                inline for (0..node.input_count) |input_index| {
                    inputs[input_index] = source_graph.input_refs[node.input_start + input_index].?;
                }
                insertInvocation(&program, executable, &inputs, &.{node.result});
            }

            for (0..source_graph.output_ct) |output_index| {
                program.insertOutput(source_graph.outputs[output_index].?);
            }
            return program;
        }

        fn insertInvocation(
            program: *ExecutableProgram.Program(capacity, Execution.Op),
            comptime op: Execution.Op,
            comptime inputs: []const usize,
            comptime outputs: []const usize,
        ) void {
            for (inputs) |tensor_id| program.insertInputRef(tensor_id);
            for (outputs) |tensor_id| {
                program.insertOutputRef(tensor_id);
                program.materialized[tensor_id] = true;
                program.tensors[tensor_id].?.origin = .{ .node = program.node_ct };
            }
            program.insertInvocation(.{
                .op = op,
                .input_start = program.input_ref_ct - inputs.len,
                .input_count = inputs.len,
                .output_start = program.output_ref_ct - outputs.len,
                .output_count = outputs.len,
            });
        }

        fn planCompute(
            comptime compute: Semantic.Op.Compute,
            comptime graph: anytype,
            comptime node: anytype,
        ) Execution.ExecutableCompute {
            return switch (compute) {
                .matmul => blk: {
                    const lhs_id = graph.input_refs[node.input_start].?;
                    const rhs_id = graph.input_refs[node.input_start + 1].?;
                    break :blk .{ .kernel = .{ .contraction = matmul.plan(
                        capacity,
                        graph.tensors[lhs_id].?,
                        graph.tensors[rhs_id].?,
                        graph.tensors[node.result].?,
                    ) } };
                },
                else => .{ .direct = compute },
            };
        }

        fn PlannedReduction(comptime graph: anytype, comptime group: anytype) type {
            const built = comptime buildReduction(graph, group);
            return struct {
                const input_ids = built.inputs[0..built.input_count].*;
                const output_ids = built.outputs[0..built.output_count].*;
                const instructions = built.instructions[0..built.instruction_count].*;
                const accumulators = built.accumulators[0..built.accumulator_count].*;
                const stores = built.stores[0..built.store_count].*;
                const domain_shape = built.domain_shape[0..built.domain_rank].*;
                const outer_axes = built.outer_axes[0..built.outer_axis_count].*;
                const reduction_axes = built.reduction_axis_order[0..built.reduction_axis_count].*;

                pub const plan: Plan.ReductionPlan = .{
                    .region = .{
                        .expressions = .{ .instructions = &instructions },
                        .domain_shape = &domain_shape,
                        .reduction_axes = group.descriptor.axes,
                        .keep_dims = group.descriptor.keep_dims,
                        .accumulators = &accumulators,
                        .stores = &stores,
                    },
                    .schedule = .{
                        .outer_axis_order = &outer_axes,
                        .reduction_axis_order = &reduction_axes,
                        .vector_axis = null,
                        .vector_width = 1,
                        .accumulator_lanes = 1,
                    },
                };
            };
        }

        fn PlannedMap(comptime graph: anytype, comptime group: anytype) type {
            const built = comptime buildMap(graph, group);
            return struct {
                const input_ids = built.inputs[0..built.input_count].*;
                const output_ids = built.outputs;
                const instructions = built.instructions[0..built.instruction_count].*;
                const stores = built.stores;
                const axis_order = built.axis_order[0..built.axis_count].*;

                pub const plan: Plan.MapPlan = .{
                    .region = .{
                        .expressions = .{ .instructions = &instructions },
                        .stores = &stores,
                    },
                    .schedule = .{
                        .axis_order = &axis_order,
                        .traversal = built.traversal,
                        .vector_axis = if (built.vector_width > 1 and built.axis_count > 0)
                            @intCast(built.axis_count - 1)
                        else
                            null,
                        .vector_width = built.vector_width,
                    },
                };
            };
        }

        fn MapBuildResult(comptime graph: anytype) type {
            return struct {
                inputs: [graph.tensor_ct]usize = undefined,
                input_count: usize = 0,
                outputs: [1]usize = undefined,
                instructions: [graph.node_ct]Expression.Program.Instruction = undefined,
                instruction_count: usize = 0,
                stores: [1]Region.Store = undefined,
                values: [graph.tensor_ct]?Expression.Program.ValueRef = @splat(null),
                axis_order: [capacity.max_rank]u8 = @splat(0),
                axis_count: usize = 0,
                traversal: Plan.MapPlan.Traversal = .strided,
                vector_width: usize = 1,
            };
        }

        fn buildMap(comptime graph: anytype, comptime group: anytype) MapBuildResult(graph) {
            var built: MapBuildResult(graph) = .{};
            const root = graph.nodes[group.root_node].?;
            const root_value = buildExpressionValue(graph, group.nodes, root.result, &built);
            built.outputs[0] = root.result;
            built.stores[0] = .{ .output = 0, .value = root_value };

            const output = graph.tensors[root.result].?;
            built.axis_count = output.shape.rank;
            for (0..output.shape.rank) |axis| built.axis_order[axis] = @intCast(axis);
            built.traversal = if (isContiguous(output)) .contiguous else .strided;
            var vector_dtype = output.dtype;
            for (built.inputs[0..built.input_count]) |input_id| {
                const dtype = graph.tensors[input_id].?.dtype;
                if (dtype.kind() != .boolean) {
                    vector_dtype = dtype;
                    break;
                }
            }
            built.vector_width = if (output.shape.rank == 0)
                1
            else
                std.simd.suggestVectorLength(vector_dtype.Scalar()) orelse 1;
            return built;
        }

        fn isContiguous(comptime info: anytype) bool {
            var expected: isize = 1;
            var axis = info.shape.rank;
            while (axis > 0) {
                axis -= 1;
                if (info.shape.at(axis) > 1 and info.layout.strides[axis] != expected) return false;
                expected *= @intCast(info.shape.at(axis));
            }
            return true;
        }

        fn BuildResult(comptime graph: anytype) type {
            return struct {
                inputs: [graph.tensor_ct]usize = undefined,
                input_count: usize = 0,
                outputs: [graph.node_ct]usize = undefined,
                output_count: usize = 0,
                instructions: [graph.node_ct]Expression.Program.Instruction = undefined,
                instruction_count: usize = 0,
                accumulators: [graph.node_ct]Region.Reduction.Accumulator = undefined,
                accumulator_count: usize = 0,
                stores: [graph.node_ct]Region.Store = undefined,
                store_count: usize = 0,
                values: [graph.tensor_ct]?Expression.Program.ValueRef = @splat(null),
                domain_shape: [capacity.max_rank]usize = @splat(0),
                domain_rank: usize = 0,
                outer_axes: [capacity.max_rank]u8 = @splat(0),
                outer_axis_count: usize = 0,
                reduction_axis_order: [capacity.max_rank]u8 = @splat(0),
                reduction_axis_count: usize = 0,
            };
        }

        fn buildReduction(comptime graph: anytype, comptime group: anytype) BuildResult(graph) {
            var built: BuildResult(graph) = .{};
            const domain = graph.tensors[group.domain_tensor].?;
            built.domain_rank = domain.shape.rank;
            for (0..domain.shape.rank) |axis| {
                built.domain_shape[axis] = domain.shape.at(axis);
                if (group.descriptor.axes & (@as(u64, 1) << @intCast(axis)) != 0) {
                    built.reduction_axis_order[built.reduction_axis_count] = @intCast(axis);
                    built.reduction_axis_count += 1;
                } else {
                    built.outer_axes[built.outer_axis_count] = @intCast(axis);
                    built.outer_axis_count += 1;
                }
            }

            for (group.reduction_nodes[0..group.reduction_count]) |maybe_node_id| {
                const node_id = maybe_node_id.?;
                const node = graph.nodes[node_id].?;
                const descriptor = Reduction.fromCompute(node.op.compute).?;
                const input_id = graph.input_refs[node.input_start].?;
                const accumulator_index = built.accumulator_count;
                built.accumulators[accumulator_index] = .{
                    .combine = switch (descriptor.kind) {
                        .sum, .mean => .sum,
                        .minimum => .minimum,
                        .maximum => .maximum,
                    },
                    .update = buildExpressionValue(graph, group.nodes, input_id, &built),
                    .finalize = if (descriptor.kind == .mean) .mean else .identity,
                };
                built.accumulator_count += 1;
                built.outputs[built.output_count] = node.result;
                built.output_count += 1;
                built.stores[built.store_count] = .{
                    .output = built.store_count,
                    .value = .{ .accumulator = accumulator_index },
                };
                built.store_count += 1;
            }
            return built;
        }

        fn buildExpressionValue(
            comptime graph: anytype,
            comptime included_nodes: anytype,
            comptime tensor_id: usize,
            built: anytype,
        ) Expression.Program.ValueRef {
            if (built.values[tensor_id]) |value| return value;
            const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                .node => |id| id,
                .source, .literal => return addExpressionInput(tensor_id, built),
            };
            if (!included_nodes[producer_id]) return addExpressionInput(tensor_id, built);

            const producer = graph.nodes[producer_id].?;
            const operation = Elementwise.fromCompute(producer.op.compute).?;
            var args: [3]Expression.Program.ValueRef = @splat(.{ .input = 0 });
            for (0..producer.input_count) |input_index| {
                args[input_index] = buildExpressionValue(
                    graph,
                    included_nodes,
                    graph.input_refs[producer.input_start + input_index].?,
                    built,
                );
            }
            const value: Expression.Program.ValueRef = .{ .instruction = built.instruction_count };
            built.instructions[built.instruction_count] = .{
                .operation = operation,
                .dtype = graph.tensors[tensor_id].?.dtype,
                .args = args,
            };
            built.instruction_count += 1;
            built.values[tensor_id] = value;
            return value;
        }

        fn addExpressionInput(comptime tensor_id: usize, built: anytype) Expression.Program.ValueRef {
            for (built.inputs[0..built.input_count], 0..) |existing, index| {
                if (existing == tensor_id) return .{ .input = index };
            }
            const index = built.input_count;
            built.inputs[index] = tensor_id;
            built.input_count += 1;
            const value: Expression.Program.ValueRef = .{ .input = index };
            built.values[tensor_id] = value;
            return value;
        }
    };
}
