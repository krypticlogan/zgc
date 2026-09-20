const std = @import("std");
const Graph = @import("../graph.zig");
const Op = @import("../operations/semantic.zig").Op;
const Tensor = @import("../tensor.zig");
const layout_ops = @import("../kernels/layout.zig");

/// Validates the raw semantic graph before analysis and optimization.
pub fn ValidationBackend(comptime capacity: Graph.Capacity) type {
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
