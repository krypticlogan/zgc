const std = @import("std");
const zgc = @import("zgc");

const ProducerSources = enum(usize) { lhs, rhs };
const ProducerDefinition = zgc.DefinitionBackend(ProducerSources, .{
    .max_rank = 2,
    .max_nodes = 2,
    .max_tensors = 4,
    .max_input_refs = 3,
    .max_outputs = 1,
});
const ProducerReductionModel = model: {
    var builder = ProducerDefinition.init();
    const lhs = builder.input(.lhs, .f32, &.{ 2, 3 });
    const rhs = builder.input(.rhs, .f32, &.{ 2, 3 });
    builder.output(builder.sum(builder.mul(lhs, rhs), .{ .axes = &.{1} }));
    break :model builder.finish().model();
};

const SiblingSources = enum(usize) { input, weights };
const SiblingDefinition = zgc.DefinitionBackend(SiblingSources, .{
    .max_rank = 2,
    .max_nodes = 3,
    .max_tensors = 5,
    .max_input_refs = 4,
    .max_outputs = 2,
});
const SiblingReductionModel = model: {
    var builder = SiblingDefinition.init();
    const input = builder.input(.input, .f32, &.{ 2, 3 });
    const weights = builder.parameter(.weights, .f32, &.{3});
    builder.output(builder.sum(input, .{ .axes = &.{1} }));
    builder.output(builder.max(builder.mul(input, weights), .{ .axes = &.{1} }));
    break :model builder.finish().model();
};

const TypedPointwiseSources = enum(usize) { input, threshold };
const TypedPointwiseDefinition = zgc.DefinitionBackend(TypedPointwiseSources, .{
    .max_rank = 1,
    .max_nodes = 3,
    .max_tensors = 5,
    .max_input_refs = 6,
    .max_outputs = 1,
});
const TypedPointwiseModel = model: {
    var builder = TypedPointwiseDefinition.init();
    const input = builder.input(.input, .f32, &.{8});
    const threshold = builder.input(.threshold, .f32, &.{8});
    const positive = builder.greaterThan(input, threshold);
    builder.output(builder.where(positive, input, builder.neg(input)));
    break :model builder.finish().model();
};

test "elementwise producer folds into its reduction" {
    const Model = ProducerReductionModel;

    try std.testing.expectEqual(@as(usize, 2), Model.semantic_graph.node_ct);
    try std.testing.expectEqual(@as(usize, 1), Model.build_graph.node_ct);
    try std.testing.expect(!Model.build_graph.materialized[2]);
    try std.testing.expect(Model.memory_plan.tensor_regions[2] == null);
    switch (Model.build_graph.nodes[0].?.op.compute.kernel) {
        .reduction => |plan| {
            try std.testing.expectEqual(@as(usize, 1), plan.region.expressions.instructions.len);
            try std.testing.expectEqual(@as(usize, 1), plan.region.accumulators.len);
        },
        else => return error.TestUnexpectedResult,
    }

    var model = Model.init();
    try model.copyInput(.lhs, &.{ 1, 2, 3, 4, 5, 6 });
    try model.copyInput(.rhs, &.{ 2, 3, 4, 5, 6, 7 });
    model.run();
    try std.testing.expectEqualSlices(f32, &.{ 20, 92 }, model.outputView(0).contiguousSlice().?);
}

test "reductions over a shared domain execute as one multi-output region" {
    const Model = SiblingReductionModel;

    try std.testing.expectEqual(@as(usize, 3), Model.semantic_graph.node_ct);
    try std.testing.expectEqual(@as(usize, 1), Model.build_graph.node_ct);
    const invocation = Model.build_graph.nodes[0].?;
    try std.testing.expectEqual(@as(usize, 2), invocation.output_count);
    switch (invocation.op.compute.kernel) {
        .reduction => |plan| {
            try std.testing.expectEqual(@as(usize, 1), plan.region.expressions.instructions.len);
            try std.testing.expectEqual(@as(usize, 2), plan.region.accumulators.len);
            try std.testing.expectEqual(@as(usize, 2), plan.region.stores.len);
        },
        else => return error.TestUnexpectedResult,
    }

    var model = Model.init();
    try model.copyInput(.input, &.{ 1, 2, 3, 4, 5, 6 });
    try model.copySource(.weights, &.{ 1, 2, 3 });
    model.run();
    try std.testing.expectEqualSlices(f32, &.{ 6, 15 }, model.outputView(0).contiguousSlice().?);
    try std.testing.expectEqualSlices(f32, &.{ 9, 18 }, model.outputView(1).contiguousSlice().?);
}

test "mixed boolean and numeric pointwise expressions fuse without predicate storage" {
    const Model = TypedPointwiseModel;

    try std.testing.expectEqual(@as(usize, 3), Model.semantic_graph.node_ct);
    try std.testing.expectEqual(@as(usize, 1), Model.build_graph.node_ct);
    try std.testing.expect(!Model.build_graph.materialized[2]);
    try std.testing.expect(!Model.build_graph.materialized[3]);
    try std.testing.expect(Model.memory_plan.tensor_regions[2] == null);
    try std.testing.expect(Model.memory_plan.tensor_regions[3] == null);
    switch (Model.build_graph.nodes[0].?.op.compute.kernel) {
        .map => |plan| {
            try std.testing.expectEqual(@as(usize, 3), plan.region.expressions.instructions.len);
            try std.testing.expectEqual(zgc.Dtype.bool, plan.region.expressions.instructions[0].dtype);
            try std.testing.expectEqual(zgc.Dtype.f32, plan.region.expressions.instructions[1].dtype);
            try std.testing.expectEqual(zgc.Dtype.f32, plan.region.expressions.instructions[2].dtype);
        },
        else => return error.TestUnexpectedResult,
    }

    var model = Model.init();
    try model.copyInput(.input, &.{ -1, 2, -3, 4, -5, 6, -7, 8 });
    try model.copyInput(.threshold, &.{ 0, 0, 0, 0, 0, 0, 0, 0 });
    model.run();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, model.outputView(0).contiguousSlice().?);
}
