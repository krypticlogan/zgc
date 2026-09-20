const Elementwise = @import("../operations/elementwise.zig");
const Reduction = @import("../operations/reduction.zig");

/// Select conservative logical reduction and pointwise map regions. Selected
/// regions retain semantic tensor ids until physical kernel planning.
pub fn FusionBackend() type {
    return struct {
        pub fn form(comptime Validated: type, comptime analysis: anytype) type {
            const source_graph = Validated.graph;
            const selection = comptime select(source_graph, analysis);
            return struct {
                pub const graph = Validated.graph;
                pub const regions = selection;
            };
        }

        fn select(comptime graph: anytype, comptime analysis: anytype) Selection(graph.node_ct) {
            var result: Selection(graph.node_ct) = .{};

            for (0..graph.node_ct) |node_id| {
                const descriptor = reductionForNode(graph, node_id) orelse continue;
                const input_id = graph.input_refs[graph.nodes[node_id].?.input_start].?;
                const anchor = fusionAnchor(graph, analysis, input_id);

                var selected_group: ?usize = null;
                for (0..result.region_count) |region_id| {
                    const group = result.reduction_storage[region_id].?;
                    if (!compatible(graph, group, descriptor, input_id, anchor)) continue;
                    if (!canDelayExistingOutputs(graph, group, node_id)) continue;
                    selected_group = region_id;
                    break;
                }

                const region_id = selected_group orelse blk: {
                    const id = result.region_count;
                    result.reduction_storage[id] = .{
                        .descriptor = descriptor,
                        .domain_tensor = input_id,
                        .anchor_tensor = anchor,
                    };
                    result.region_count += 1;
                    break :blk id;
                };
                var group = &result.reduction_storage[region_id].?;
                group.reduction_nodes[group.reduction_count] = node_id;
                group.reduction_count += 1;
                group.emit_node = node_id;
            }

            for (0..result.region_count) |region_id| {
                var group = &result.reduction_storage[region_id].?;
                var has_producer = false;
                for (group.reduction_nodes[0..group.reduction_count]) |maybe_node_id| {
                    const node_id = maybe_node_id.?;
                    const node = graph.nodes[node_id].?;
                    const input_id = graph.input_refs[node.input_start].?;
                    has_producer = markFoldedProducers(graph, analysis, input_id, &group.nodes) or has_producer;
                }

                group.active = group.reduction_count > 1 or has_producer;
                if (!group.active) continue;
                for (group.reduction_nodes[0..group.reduction_count]) |maybe_node_id| {
                    result.node_region[maybe_node_id.?] = .{ .reduction = region_id };
                }
                for (group.nodes, 0..) |folded, node_id| {
                    if (folded) result.node_region[node_id] = .{ .reduction = region_id };
                }
            }

            for (0..graph.node_ct) |node_id| {
                if (result.node_region[node_id] != null or pointwiseForNode(graph, node_id) == null) continue;
                if (hasUnassignedPointwiseConsumer(graph, analysis, result, node_id)) continue;

                var group: MapGroup(graph.node_ct) = .{ .root_node = node_id };
                group.nodes[node_id] = true;
                group.node_count = 1;
                const root = graph.nodes[node_id].?;
                for (0..root.input_count) |input_index| {
                    markMapProducers(
                        graph,
                        analysis,
                        result,
                        graph.input_refs[root.input_start + input_index].?,
                        &group,
                    );
                }
                if (group.node_count < 2) continue;

                const map_id = result.map_count;
                result.map_storage[map_id] = group;
                result.map_count += 1;
                for (group.nodes, 0..) |included, included_node| {
                    if (included) result.node_region[included_node] = .{ .map = map_id };
                }
            }
            return result;
        }

        fn pointwiseForNode(comptime graph: anytype, comptime node_id: usize) ?Elementwise.Operation {
            return switch (graph.nodes[node_id].?.op) {
                .compute => |compute| Elementwise.fromCompute(compute),
                .view => null,
            };
        }

        fn hasUnassignedPointwiseConsumer(
            comptime graph: anytype,
            comptime analysis: anytype,
            comptime selection: Selection(graph.node_ct),
            comptime producer_id: usize,
        ) bool {
            const result_id = graph.nodes[producer_id].?.result;
            if (analysis.use_counts[result_id] != 1 or analysis.is_output[result_id]) return false;
            for (producer_id + 1..graph.node_ct) |consumer_id| {
                if (selection.node_region[consumer_id] != null or pointwiseForNode(graph, consumer_id) == null) continue;
                const consumer = graph.nodes[consumer_id].?;
                for (0..consumer.input_count) |input_index| {
                    if (graph.input_refs[consumer.input_start + input_index].? == result_id) return true;
                }
            }
            return false;
        }

        fn markMapProducers(
            comptime graph: anytype,
            comptime analysis: anytype,
            comptime selection: Selection(graph.node_ct),
            comptime tensor_id: usize,
            group: *MapGroup(graph.node_ct),
        ) void {
            if (analysis.use_counts[tensor_id] != 1 or analysis.is_output[tensor_id]) return;
            const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                .node => |id| id,
                .source, .literal => return,
            };
            if (selection.node_region[producer_id] != null or pointwiseForNode(graph, producer_id) == null) return;
            if (group.nodes[producer_id]) return;

            group.nodes[producer_id] = true;
            group.node_count += 1;
            const producer = graph.nodes[producer_id].?;
            for (0..producer.input_count) |input_index| {
                markMapProducers(
                    graph,
                    analysis,
                    selection,
                    graph.input_refs[producer.input_start + input_index].?,
                    group,
                );
            }
        }

        fn reductionForNode(comptime graph: anytype, comptime node_id: usize) ?Reduction.Descriptor {
            return switch (graph.nodes[node_id].?.op) {
                .compute => |compute| Reduction.fromCompute(compute),
                .view => null,
            };
        }

        fn compatible(
            comptime graph: anytype,
            comptime group: Group(graph.node_ct),
            comptime descriptor: Reduction.Descriptor,
            comptime input_id: usize,
            comptime anchor: usize,
        ) bool {
            if (group.descriptor.axes != descriptor.axes or
                group.descriptor.keep_dims != descriptor.keep_dims or
                group.anchor_tensor != anchor)
            {
                return false;
            }
            const expected = graph.tensors[group.domain_tensor].?;
            const candidate = graph.tensors[input_id].?;
            if (expected.dtype != candidate.dtype or expected.shape.rank != candidate.shape.rank) return false;
            for (0..expected.shape.rank) |axis| {
                if (expected.shape.at(axis) != candidate.shape.at(axis)) return false;
            }
            return true;
        }

        fn canDelayExistingOutputs(comptime graph: anytype, comptime group: Group(graph.node_ct), comptime candidate_node: usize) bool {
            for (group.reduction_nodes[0..group.reduction_count]) |maybe_node_id| {
                const reduction_node = maybe_node_id.?;
                const result_id = graph.nodes[reduction_node].?.result;
                for (reduction_node + 1..candidate_node + 1) |consumer_id| {
                    const consumer = graph.nodes[consumer_id].?;
                    for (0..consumer.input_count) |input_index| {
                        if (graph.input_refs[consumer.input_start + input_index].? == result_id) return false;
                    }
                }
            }
            return true;
        }

        fn fusionAnchor(comptime graph: anytype, comptime analysis: anytype, comptime tensor_id: usize) usize {
            const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                .node => |id| id,
                .source, .literal => return tensor_id,
            };
            if (analysis.use_counts[tensor_id] != 1 or analysis.is_output[tensor_id]) return tensor_id;
            const producer = graph.nodes[producer_id].?;
            const compute = switch (producer.op) {
                .compute => |value| value,
                .view => return tensor_id,
            };
            const operation = Elementwise.fromCompute(compute) orelse return tensor_id;
            if (!operation.isReductionCompatible() or producer.input_count == 0) return tensor_id;
            return fusionAnchor(graph, analysis, graph.input_refs[producer.input_start].?);
        }

        fn markFoldedProducers(
            comptime graph: anytype,
            comptime analysis: anytype,
            comptime tensor_id: usize,
            folded: *[graph.node_ct]bool,
        ) bool {
            const producer_id = switch (graph.tensors[tensor_id].?.origin) {
                .node => |id| id,
                .source, .literal => return false,
            };
            if (analysis.use_counts[tensor_id] != 1 or analysis.is_output[tensor_id]) return false;
            const producer = graph.nodes[producer_id].?;
            const compute = switch (producer.op) {
                .compute => |value| value,
                .view => return false,
            };
            const operation = Elementwise.fromCompute(compute) orelse return false;
            if (!operation.isReductionCompatible()) return false;

            folded[producer_id] = true;
            for (0..producer.input_count) |input_index| {
                _ = markFoldedProducers(
                    graph,
                    analysis,
                    graph.input_refs[producer.input_start + input_index].?,
                    folded,
                );
            }
            return true;
        }
    };
}

pub fn Selection(comptime node_count: usize) type {
    return struct {
        reduction_storage: [node_count]?Group(node_count) = .{null} ** node_count,
        map_storage: [node_count]?MapGroup(node_count) = .{null} ** node_count,
        node_region: [node_count]?RegionRef = .{null} ** node_count,
        region_count: usize = 0,
        map_count: usize = 0,
    };
}

pub const RegionRef = union(enum) { reduction: usize, map: usize };

pub fn MapGroup(comptime node_count: usize) type {
    return struct {
        root_node: usize,
        nodes: [node_count]bool = @splat(false),
        node_count: usize = 0,
    };
}

pub fn Group(comptime node_count: usize) type {
    return struct {
        descriptor: Reduction.Descriptor,
        domain_tensor: usize,
        anchor_tensor: usize,
        reduction_nodes: [node_count]?usize = .{null} ** node_count,
        reduction_count: usize = 0,
        nodes: [node_count]bool = @splat(false),
        emit_node: usize = 0,
        active: bool = false,
    };
}
