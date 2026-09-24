const Graph = @import("../graph.zig");
const Semantic = @import("../operations/semantic.zig");
const Tensor = @import("../tensor.zig");
const layout_ops = @import("../kernels/layout.zig");
const matmul = @import("../optimization/matmul.zig");

/// Select physical tensor layouts and source packing without choosing kernels.
pub fn LayoutPlanningBackend(comptime capacity: Graph.Capacity) type {
    return struct {
        pub fn plan(comptime Validated: type, comptime analysis: anytype) Graph.Graph(capacity, Semantic.Op) {
            const raw = Validated.graph;
            var graph: Graph.Graph(capacity, Semantic.Op) = .init();

            for (0..raw.tensor_ct) |tensor_id| {
                const inserted = graph.insertTensor(raw.tensors[tensor_id].?);
                if (inserted != tensor_id) @compileError("optimization changed tensor order");
            }
            for (0..raw.sources.len) |source_index| {
                if (raw.sources[source_index]) |source| graph.insertSource(source_index, source);
            }

            for (0..raw.node_ct) |node_id| {
                const node = raw.nodes[node_id].?;
                const input_ids = raw.input_refs[node.input_start..][0..node.input_count];
                var result = graph.tensors[node.result].?;

                const planned: Semantic.Op = switch (node.op) {
                    .view => |view| blk: {
                        const InputInfos = [node.input_count]Graph.Graph(capacity, Semantic.Op).TensorInfo;
                        var inputs: InputInfos = undefined;
                        inline for (0..node.input_count) |input_index| {
                            inputs[input_index] = graph.tensors[input_ids[input_index].?].?;
                        }
                        const inferred = layout_ops.infer(view, &inputs, result.shape, capacity.max_rank);
                        result.shape = inferred.shape;
                        result.layout = inferred.layout;
                        result.storage_tensor = inferred.storage_tensor;
                        break :blk .{ .view = view };
                    },
                    .compute => |compute| blk: {
                        result.layout = computeLayout(compute, &graph, input_ids, result.shape, analysis);
                        break :blk .{ .compute = compute };
                    },
                };
                graph.tensors[node.result] = result;

                inline for (0..node.input_count) |input_index| {
                    graph.insertRef(input_ids[input_index].?);
                }
                graph.insertNode(.{
                    .op = planned,
                    .input_start = graph.input_ref_ct - node.input_count,
                    .input_count = node.input_count,
                    .result = node.result,
                });
            }

            for (0..raw.output_ct) |output_index| graph.insertOutput(raw.outputs[output_index].?);
            return graph;
        }

        fn computeLayout(
            comptime op: Semantic.Op.Compute,
            graph: *Graph.Graph(capacity, Semantic.Op),
            comptime input_ids: []const ?Tensor.Id,
            shape: Tensor.Shape(capacity.max_rank),
            comptime analysis: anytype,
        ) Tensor.Layout(capacity.max_rank) {
            return switch (op) {
                .matmul => matmul.selectOutputLayout(capacity, graph, input_ids[0].?, input_ids[1].?, shape, analysis),
                .relu, .exp, .neg, .abs, .sqrt, .log, .reciprocal, .softmax, .logical_not => preserveBatchLayout(graph, input_ids[0].?, shape),
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
                => preserveBatchLayout(graph, input_ids[0].?, shape),
                .clamp => preserveBatchLayout(graph, input_ids[0].?, shape),
                .where => preserveBatchLayout(graph, input_ids[1].?, shape),
                .copy => preserveBatchLayout(graph, input_ids[0].?, shape),
                .contiguous, .pad, .shift, .sum, .mean, .min, .max, .concat => .contiguous(shape),
            };
        }

        fn preserveBatchLayout(
            graph: *const Graph.Graph(capacity, Semantic.Op),
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
