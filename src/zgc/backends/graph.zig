const std = @import("std");
const Graph = @import("../graph.zig");
const Matmul = @import("../matmul.zig");
const Tensor = @import("../tensor.zig");
const layout_ops = @import("../kernels/layout.zig");

/// Lowers a completed definition into the concrete graph sized by the counting
/// backend.
pub fn GraphBackend(
    comptime Definition: type,
    comptime capacity: Graph.Capacity,
) type {
    return struct {
        pub fn build(comptime definition: Definition) Graph.Graph(capacity) {
            var graph: Graph.Graph(capacity) = .init();
            const TensorInfo = Tensor.Info(capacity.max_rank);

            inline for (0..definition.tensor_count) |tensor_id| {
                const record = definition.tensors[tensor_id];
                const shape: Tensor.Shape(capacity.max_rank) = .init(record.value.shape.slice());

                switch (record.origin) {
                    .source => |source_index| {
                        const info: TensorInfo = .{
                            .dtype = record.value.dtype,
                            .shape = shape,
                            .origin = .{ .source = source_index },
                            .layout = .contiguous(shape),
                            .storage_tensor = tensor_id,
                        };
                        graph.insertSource(source_index, .{
                            .kind = record.source_kind.?,
                            .tensor = tensor_id,
                        });
                        const inserted = graph.insertTensor(info);
                        if (inserted != tensor_id) @compileError("definition tensor order was not preserved");
                    },
                    .literal => |value| {
                        const info: TensorInfo = .{
                            .dtype = record.value.dtype,
                            .shape = shape,
                            .origin = .{ .literal = value },
                            .layout = .contiguous(shape),
                            .storage_tensor = tensor_id,
                        };
                        const inserted = graph.insertTensor(info);
                        if (inserted != tensor_id) @compileError("definition tensor order was not preserved");
                    },
                    .node => |node_id| {
                        const node = definition.nodes[node_id];
                        const InputInfos = [node.input_count]TensorInfo;
                        var input_infos: InputInfos = undefined;
                        inline for (0..node.input_count) |input_index| {
                            const input_id = definition.input_refs[node.input_start + input_index];
                            input_infos[input_index] = graph.tensors[input_id].?;
                        }

                        const info: TensorInfo = switch (node.op) {
                            .compute => |compute| .{
                                .dtype = record.value.dtype,
                                .shape = shape,
                                .origin = .{ .node = node_id },
                                .layout = computeLayout(
                                    compute,
                                    &graph,
                                    definition.input_refs[node.input_start..][0..node.input_count],
                                    shape,
                                ),
                                .storage_tensor = tensor_id,
                            },
                            .view => |view| blk: {
                                const result = layout_ops.infer(view, &input_infos, shape, capacity.max_rank);
                                break :blk .{
                                    .dtype = record.value.dtype,
                                    .shape = result.shape,
                                    .origin = .{ .node = node_id },
                                    .layout = result.layout,
                                    .storage_tensor = result.storage_tensor,
                                };
                            },
                        };
                        const inserted = graph.insertTensor(info);
                        if (inserted != tensor_id) @compileError("definition tensor order was not preserved");

                        inline for (0..node.input_count) |input_index| {
                            graph.insertRef(definition.input_refs[node.input_start + input_index]);
                        }
                        graph.insertNode(.{
                            .op = lowerOperation(
                                node.op,
                                &graph,
                                definition.input_refs[node.input_start..][0..node.input_count],
                                info,
                            ),
                            .kind = node.op.kind(),
                            .input_start = graph.input_ref_ct - node.input_count,
                            .input_count = node.input_count,
                            .result = tensor_id,
                        });
                    },
                }
            }

            inline for (0..definition.output_count) |output_index| {
                graph.insertOutput(definition.outputs[output_index]);
            }
            return graph;
        }

        fn computeLayout(
            comptime op: @import("../op.zig").Op.Compute,
            graph: *Graph.Graph(capacity),
            comptime input_ids: []const Tensor.Id,
            shape: Tensor.Shape(capacity.max_rank),
        ) Tensor.Layout(capacity.max_rank) {
            return switch (op) {
                .matmul => matmulLayout(graph, input_ids[0], input_ids[1], shape),
                .relu, .exp, .neg, .abs, .sqrt, .log, .reciprocal, .softmax, .logical_not => preserveBatchLayout(graph, input_ids[0], shape),
                .add,
                .sub,
                .mul,
                .div,
                .minimum,
                .maximum,
                .equal,
                .not_equal,
                .less_than,
                .less_equal,
                .greater_than,
                .greater_equal,
                .logical_and,
                .logical_or,
                => preserveBatchLayout(graph, input_ids[0], shape),
                .clamp => preserveBatchLayout(graph, input_ids[0], shape),
                .where => preserveBatchLayout(graph, input_ids[1], shape),
                .copy => preserveBatchLayout(graph, input_ids[0], shape),
                .contiguous => .contiguous(shape),
                .pad, .shift => .contiguous(shape),
                .sum, .mean, .min, .max => .contiguous(shape),
                .concat => .contiguous(shape),
            };
        }

        fn lowerOperation(
            op: @import("../op.zig").Op,
            graph: *const Graph.Graph(capacity),
            comptime input_ids: []const Tensor.Id,
            result: Tensor.Info(capacity.max_rank),
        ) @import("../op.zig").Op {
            return switch (op) {
                .view => op,
                .compute => |compute| switch (compute) {
                    .matmul => .{ .compute = .{ .matmul = .{
                        .strategy = selectMatmulStrategy(
                            graph.tensors[input_ids[0]].?,
                            graph.tensors[input_ids[1]].?,
                            result,
                        ),
                    } } },
                    else => op,
                },
            };
        }

        fn selectMatmulStrategy(
            lhs: Tensor.Info(capacity.max_rank),
            rhs: Tensor.Info(capacity.max_rank),
            output: Tensor.Info(capacity.max_rank),
        ) Matmul.Strategy {
            if (rhs.layout.strides[1] == 1 and output.layout.strides[1] == 1) {
                return .output_columns;
            }
            if (lhs.layout.strides[1] == 1 and rhs.layout.strides[0] == 1) {
                return .contracted_axis;
            }
            if (lhs.layout.strides[0] == 1 and output.layout.strides[0] == 1) {
                return .output_rows;
            }
            return .scalar;
        }

        fn matmulLayout(
            graph: *Graph.Graph(capacity),
            comptime lhs_id: Tensor.Id,
            comptime rhs_id: Tensor.Id,
            shape: Tensor.Shape(capacity.max_rank),
        ) Tensor.Layout(capacity.max_rank) {
            packMatmulWeights(graph, rhs_id);

            const vector_len = std.simd.suggestVectorLength(f32) orelse
                return .contiguous(shape);
            if (shape.rank != 2 or shape.at(0) < vector_len) return .contiguous(shape);

            var lhs = &graph.tensors[lhs_id].?;
            if (isBatchLayout(lhs.*)) return .firstAxisContiguous(shape);

            const can_relayout = switch (lhs.origin) {
                .source => |source_id| graph.sources[source_id].?.kind == .input,
                .node => lhs.storage_tensor == lhs_id,
                .literal => false,
            };
            if (!can_relayout) return .contiguous(shape);

            lhs.layout = .firstAxisContiguous(lhs.shape);
            return .firstAxisContiguous(shape);
        }

        /// Pack logical [K, N] parameter and constant sources as physical
        /// [N, K]. The graph retains its logical shape; only its strides
        /// change, allowing both contracted-axis and batched-row kernels to
        /// consume each output column contiguously.
        fn packMatmulWeights(
            graph: *Graph.Graph(capacity),
            comptime rhs_id: Tensor.Id,
        ) void {
            var rhs = &graph.tensors[rhs_id].?;
            if (rhs.shape.rank != 2 or rhs.storage_tensor != rhs_id) return;
            const source_id = switch (rhs.origin) {
                .source => |id| id,
                .node, .literal => return,
            };
            switch (graph.sources[source_id].?.kind) {
                .parameter, .constant => rhs.layout = .firstAxisContiguous(rhs.shape),
                .input, .state => {},
            }
        }

        fn preserveBatchLayout(
            graph: *const Graph.Graph(capacity),
            comptime input_id: Tensor.Id,
            output_shape: Tensor.Shape(capacity.max_rank),
        ) Tensor.Layout(capacity.max_rank) {
            const input = graph.tensors[input_id].?;
            if (output_shape.rank == 2 and
                input.shape.rank == 2 and
                input.storage_tensor == input_id and
                input.shape.at(0) == output_shape.at(0) and
                input.shape.at(1) == output_shape.at(1) and
                isBatchLayout(input))
            {
                return .firstAxisContiguous(output_shape);
            }
            return .contiguous(output_shape);
        }

        fn isBatchLayout(info: Tensor.Info(capacity.max_rank)) bool {
            return info.shape.rank == 2 and
                info.layout.offset == 0 and
                info.layout.strides[0] == 1 and
                info.layout.strides[1] == @as(isize, @intCast(info.shape.at(0)));
        }
    };
}
