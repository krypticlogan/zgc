const Graph = @import("../graph.zig");

/// Derives the exact graph capacities required by a completed definition.
pub fn CountingBackend(comptime Definition: type) type {
    return struct {
        pub fn count(comptime definition: Definition) Graph.Capacity {
            var capacity: Graph.Capacity = .{
                .max_nodes = definition.node_count,
                .max_input_refs = definition.input_ref_count,
                .max_tensors = definition.tensor_count,
                .max_outputs = definition.output_count,
            };

            for (definition.tensors[0..definition.tensor_count]) |record| {
                capacity.max_rank = @max(capacity.max_rank, record.value.shape.rank);
                switch (record.origin) {
                    .source => |source_index| {
                        capacity.max_sources = @max(capacity.max_sources, source_index + 1);
                    },
                    .node, .literal => {},
                }
            }
            return capacity;
        }
    };
}
