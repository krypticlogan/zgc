const Elementwise = @import("../operations/elementwise.zig");

/// Compile-time facts used by optimization policy. This pass does not mutate
/// the graph or decide which legal opportunities should be selected.
pub fn GraphAnalysis() type {
    return struct {
        pub fn analyze(comptime Validated: type) Analysis(Validated.graph.tensor_ct, Validated.graph.input_ref_ct) {
            const graph = Validated.graph;
            var result: Analysis(graph.tensor_ct, graph.input_ref_ct) = .{};

            for (0..graph.node_ct) |node_id| {
                const node = graph.nodes[node_id].?;
                for (0..node.input_count) |input_index| {
                    const ref_index = node.input_start + input_index;
                    const tensor_id = graph.input_refs[ref_index].?;
                    result.use_counts[tensor_id] += 1;
                }
            }
            for (0..graph.output_ct) |output_index| {
                result.is_output[graph.outputs[output_index].?] = true;
            }

            for (0..graph.node_ct) |node_id| {
                const consumer = graph.nodes[node_id].?;
                const consumer_compute = switch (consumer.op) {
                    .compute => |compute| compute,
                    .view => continue,
                };
                if (Elementwise.fromCompute(consumer_compute) == null) continue;

                for (0..consumer.input_count) |input_index| {
                    const ref_index = consumer.input_start + input_index;
                    const tensor_id = graph.input_refs[ref_index].?;
                    if (result.use_counts[tensor_id] != 1 or result.is_output[tensor_id]) continue;
                    const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                        .node => |id| id,
                        .source, .literal => continue,
                    };
                    const producer_compute = switch (graph.nodes[producer_id].?.op) {
                        .compute => |compute| compute,
                        .view => continue,
                    };
                    if (Elementwise.fromCompute(producer_compute) != null) {
                        result.fusible_input_refs[ref_index] = true;
                    }
                }
            }
            return result;
        }
    };
}

pub fn Analysis(comptime tensor_count: usize, comptime input_ref_count: usize) type {
    return struct {
        use_counts: [tensor_count]usize = @splat(0),
        is_output: [tensor_count]bool = @splat(false),
        fusible_input_refs: [input_ref_count]bool = @splat(false),

        pub fn hasFusibleElementwiseEdges(comptime analysis: @This()) bool {
            for (analysis.fusible_input_refs) |fusible| if (fusible) return true;
            return false;
        }
    };
}
