/// Half-open execution interval for one tensor's backing storage.
/// A null end marks storage that must remain reserved for the model lifetime.
pub const Lifetime = struct {
    begin_node: usize,
    end_node_exclusive: ?usize,

    pub fn isPersistent(lifetime: Lifetime) bool {
        return lifetime.end_node_exclusive == null;
    }
};

pub fn Analysis(comptime tensor_count: usize) type {
    return struct {
        tensor_lifetimes: [tensor_count]Lifetime,
    };
}

/// Computes storage lifetimes for a validated, sequential graph. Aliasing
/// tensors receive the consolidated lifetime of their storage root.
pub fn LifetimeAnalysis() type {
    return struct {
        pub fn analyze(
            comptime Validated: type,
        ) Analysis(Validated.graph.tensor_ct) {
            const graph = Validated.graph;
            var storage_lifetimes: [graph.tensor_ct]?Lifetime = @splat(null);

            for (0..graph.tensor_ct) |tensor_id| {
                const info = graph.tensors[tensor_id].?;
                if (info.storage_tensor != tensor_id) continue;

                storage_lifetimes[tensor_id] = switch (info.origin) {
                    .source => .{
                        .begin_node = 0,
                        .end_node_exclusive = null,
                    },
                    .node => |node_id| .{
                        .begin_node = node_id,
                        .end_node_exclusive = node_id + 1,
                    },
                    .literal => .{
                        .begin_node = 0,
                        .end_node_exclusive = null,
                    },
                };
            }

            for (0..graph.node_ct) |node_id| {
                const node = graph.nodes[node_id].?;
                for (0..node.input_count) |input_index| {
                    const tensor_id = graph.input_refs[node.input_start + input_index].?;
                    const storage_tensor = graph.tensors[tensor_id].?.storage_tensor;
                    if (storage_lifetimes[storage_tensor].?.end_node_exclusive) |end_node| {
                        storage_lifetimes[storage_tensor].?.end_node_exclusive = @max(
                            end_node,
                            node_id + 1,
                        );
                    }
                }
            }

            for (0..graph.output_ct) |output_index| {
                const tensor_id = graph.outputs[output_index].?;
                const storage_tensor = graph.tensors[tensor_id].?.storage_tensor;
                storage_lifetimes[storage_tensor].?.end_node_exclusive = null;
            }

            var result: Analysis(graph.tensor_ct) = undefined;
            for (0..graph.tensor_ct) |tensor_id| {
                const storage_tensor = graph.tensors[tensor_id].?.storage_tensor;
                result.tensor_lifetimes[tensor_id] = storage_lifetimes[storage_tensor].?;
            }
            return result;
        }
    };
}
