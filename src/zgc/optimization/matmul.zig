const std = @import("std");
const ContractionPlan = @import("../execution/kernel_plan.zig").ContractionPlan;
const Graph = @import("../graph.zig");
const Tensor = @import("../tensor.zig");

/// Fully selected matmul execution policy. No geometry-dependent selection is
/// performed when the model runs.
pub const Plan = ContractionPlan;

pub fn plan(
    comptime capacity: Graph.Capacity,
    lhs: Tensor.Info(capacity.max_rank),
    rhs: Tensor.Info(capacity.max_rank),
    output: Tensor.Info(capacity.max_rank),
) Plan {
    return .{ .strategy = selectStrategy(capacity, lhs, rhs, output) };
}

pub fn selectOutputLayout(
    comptime capacity: Graph.Capacity,
    graph: anytype,
    comptime lhs_id: Tensor.Id,
    comptime rhs_id: Tensor.Id,
    shape: Tensor.Shape(capacity.max_rank),
    comptime analysis: anytype,
) Tensor.Layout(capacity.max_rank) {
    packWeights(capacity, graph, rhs_id, analysis);

    const vector_len = std.simd.suggestVectorLength(f32) orelse return .contiguous(shape);
    if (shape.rank != 2 or shape.at(0) < vector_len) return .contiguous(shape);

    var lhs = &graph.tensors[lhs_id].?;
    if (isBatchLayout(capacity, lhs.*)) return .firstAxisContiguous(shape);
    if (analysis.use_counts[lhs_id] != 1) return .contiguous(shape);

    const can_relayout = switch (lhs.origin) {
        .source => |source_id| graph.sources[source_id].?.kind == .input,
        .node => lhs.storage_tensor == lhs_id,
        .literal => false,
    };
    if (!can_relayout) return .contiguous(shape);

    lhs.layout = .firstAxisContiguous(lhs.shape);
    return .firstAxisContiguous(shape);
}

pub fn validate(
    comptime plan_value: Plan,
    comptime lhs: anytype,
    comptime rhs: anytype,
    comptime output: anytype,
) void {
    const compatible = switch (plan_value.strategy) {
        .output_columns => rhs.layout.strides[1] == 1 and output.layout.strides[1] == 1,
        .contracted_axis => lhs.layout.strides[1] == 1 and rhs.layout.strides[0] == 1,
        .output_rows => lhs.layout.strides[0] == 1 and output.layout.strides[0] == 1,
        .scalar => true,
    };
    if (!compatible) @compileError("optimized matmul strategy is incompatible with its layouts");
}

fn selectStrategy(
    comptime capacity: Graph.Capacity,
    lhs: Tensor.Info(capacity.max_rank),
    rhs: Tensor.Info(capacity.max_rank),
    output: Tensor.Info(capacity.max_rank),
) ContractionPlan.Strategy {
    if (rhs.layout.strides[1] == 1 and output.layout.strides[1] == 1) return .output_columns;
    if (lhs.layout.strides[1] == 1 and rhs.layout.strides[0] == 1) return .contracted_axis;
    if (lhs.layout.strides[0] == 1 and output.layout.strides[0] == 1) return .output_rows;
    return .scalar;
}

fn packWeights(
    comptime capacity: Graph.Capacity,
    graph: anytype,
    comptime rhs_id: Tensor.Id,
    comptime analysis: anytype,
) void {
    _ = capacity;
    if (analysis.use_counts[rhs_id] != 1) return;
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

fn isBatchLayout(
    comptime capacity: Graph.Capacity,
    info: Tensor.Info(capacity.max_rank),
) bool {
    return info.shape.rank == 2 and
        info.layout.offset == 0 and
        info.layout.strides[0] == 1 and
        info.layout.strides[1] == @as(isize, @intCast(info.shape.at(0)));
}
