const std = @import("std");
const Graph = @import("../graph.zig");
const Tensor = @import("../tensor.zig");
const layout_ops = @import("../kernels/layout.zig");

/// Validates the fully lowered graph before model types and executable kernels
/// are instantiated. Kernels may rely on these contracts without repeating
/// shape, dtype, rank, or layout checks.
pub fn ValidationBackend(comptime capacity: Graph.Capacity) type {
    return struct {
        pub fn validate(comptime lowered_graph: Graph.Graph(capacity)) type {
            inline for (0..lowered_graph.node_ct) |node_id| {
                const node = lowered_graph.nodes[node_id].?;
                const output = lowered_graph.tensors[node.result].?;
                const Inputs = [node.input_count]Graph.Graph(capacity).TensorInfo;
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
                        const expected_dtype = inputs[0].dtype;
                        if (output.dtype != expected_dtype) {
                            @compileError("lowered operation output dtype does not match its inferred dtype");
                        }
                        switch (compute) {
                            .matmul => |plan| validateMatmulPlan(plan.strategy, inputs[0], inputs[1], output),
                            else => {},
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

                fn tensorInfo(comptime tensor_id: Tensor.Id) Graph.Graph(capacity).TensorInfo {
                    if (tensor_id >= graph.tensor_ct) {
                        @compileError("tensor id is outside the validated graph");
                    }
                    return graph.tensors[tensor_id].?;
                }
            };
        }

        fn validateMatmulPlan(
            comptime strategy: @import("../matmul.zig").Strategy,
            comptime lhs: Graph.Graph(capacity).TensorInfo,
            comptime rhs: Graph.Graph(capacity).TensorInfo,
            comptime output: Graph.Graph(capacity).TensorInfo,
        ) void {
            const compatible = switch (strategy) {
                .output_columns => rhs.layout.strides[1] == 1 and output.layout.strides[1] == 1,
                .contracted_axis => lhs.layout.strides[1] == 1 and rhs.layout.strides[0] == 1,
                .output_rows => lhs.layout.strides[0] == 1 and output.layout.strides[0] == 1,
                .scalar => true,
            };
            if (!compatible) {
                @compileError("lowered matmul strategy is incompatible with its tensor layouts");
            }
        }
    };
}
