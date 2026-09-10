const std = @import("std");
const zgc = @import("zgc");

const Sources = enum(usize) { lhs, rhs, auxiliary };
const Definition = zgc.DefinitionBackend(Sources, .{
    .max_rank = 4,
    .max_nodes = 6,
    .max_tensors = 8,
    .max_input_refs = 8,
    .max_outputs = 2,
});

const matmul_model = model: {
    var builder = Definition.init();
    const lhs = builder.parameter(.lhs, .f32, &.{ 3, 4 });
    const rhs = builder.parameter(.rhs, .f32, &.{ 4, 7 });
    builder.output(builder.relu(builder.matmul(lhs, rhs)));
    break :model builder.finish().model();
};

test "definition counting and lowering preserve exact graph contracts" {
    const counts = matmul_model.internal_capacity;
    const graph = matmul_model.build_graph;

    try std.testing.expectEqual(@as(usize, 2), counts.max_nodes);
    try std.testing.expectEqual(@as(usize, 4), counts.max_tensors);
    try std.testing.expectEqual(@as(usize, 3), counts.max_input_refs);
    try std.testing.expectEqual(@as(usize, 1), counts.max_outputs);
    try std.testing.expectEqual(@as(usize, 2), counts.max_sources);
    try std.testing.expectEqual(@as(usize, 2), counts.max_rank);
    try std.testing.expectEqualSlices(usize, &.{ 3, 7 }, graph.tensors[2].?.shape.slice());
    try std.testing.expectEqual([2]isize{ 7, 1 }, graph.tensors[2].?.layout.strides);
    try std.testing.expectEqual(zgc.Matmul.Strategy.contracted_axis, graph.nodes[0].?.op.compute.matmul.strategy);
    try std.testing.expectEqual(@as(usize, 3), graph.outputs[0].?);
}

const batch_model = model: {
    const batch = std.simd.suggestVectorLength(f32) orelse 4;
    var builder = Definition.init();
    const input = builder.input(.lhs, .f32, &.{ batch, 4 });
    const weights = builder.parameter(.rhs, .f32, &.{ 4, 3 });
    const bias = builder.parameter(.auxiliary, .f32, &.{3});
    builder.output(builder.relu(builder.add(builder.matmul(input, weights), bias)));
    break :model builder.finish().model();
};

test "lowering fixes batch-oriented layouts and matmul strategy" {
    const graph = batch_model.build_graph;
    const batch = std.simd.suggestVectorLength(f32) orelse 4;
    const expected = [2]isize{ 1, batch };

    try std.testing.expectEqual(expected, graph.tensors[0].?.layout.strides);
    try std.testing.expectEqual(expected, graph.tensors[3].?.layout.strides);
    try std.testing.expectEqual(expected, graph.tensors[4].?.layout.strides);
    try std.testing.expectEqual(expected, graph.tensors[5].?.layout.strides);
    try std.testing.expectEqual(zgc.Matmul.Strategy.output_rows, graph.nodes[0].?.op.compute.matmul.strategy);
}

const reduction_model = model: {
    var builder = Definition.init();
    const input = builder.input(.lhs, .f32, &.{ 2, 3, 4, 5 });
    const bias = builder.parameter(.rhs, .f32, &.{ 1, 4, 1 });
    const biased = builder.add(input, bias);
    builder.output(builder.mean(biased, .{ .axes = &.{ -3, -2 }, .keep_dims = true }));
    builder.output(builder.max(input, zgc.ReductionOptions{}));
    break :model builder.finish().model();
};

test "broadcasting and reductions normalize compile-time geometry" {
    const graph = reduction_model.build_graph;

    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4, 5 }, graph.tensors[2].?.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 1, 5 }, graph.tensors[3].?.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{}, graph.tensors[4].?.shape.slice());
    try std.testing.expectEqual(@as(u64, (1 << 1) | (1 << 2)), graph.nodes[1].?.op.compute.mean.axes);
    try std.testing.expect(graph.nodes[1].?.op.compute.mean.keep_dims);
    try std.testing.expectEqual(@as(u64, 0b1111), graph.nodes[2].?.op.compute.max.axes);
}

const structural_model = model: {
    var builder = Definition.init();
    const input = builder.input(.lhs, .f32, &.{ 2, 1, 3, 4 });
    const squeezed = builder.squeeze(input, 1);
    const flattened = builder.flatten(squeezed, .{ .start_axis = 1 });
    const expanded = builder.unsqueeze(flattened, -1);
    const reshaped = builder.reshape(expanded, &.{ 4, 6 });
    const permuted = builder.permute(reshaped, &.{ 1, 0 });
    builder.output(builder.slice(permuted, .{ .axis = 0, .start = 1, .end = 6, .step = 2 }));
    break :model builder.finish().model();
};

test "structural operations lower to one static alias chain" {
    const graph = structural_model.build_graph;
    const output = graph.tensors[6].?;

    try std.testing.expectEqualSlices(usize, &.{ 3, 4 }, output.shape.slice());
    try std.testing.expectEqual([2]isize{ 2, 6 }, output.layout.strides[0..2].*);
    try std.testing.expectEqual(@as(usize, 1), output.layout.offset);
    for (1..7) |tensor_id| {
        try std.testing.expectEqual(graph.tensors[0].?.storage_tensor, graph.tensors[tensor_id].?.storage_tensor);
    }
}

test "scalar and full definitions retain immutable literal geometry" {
    const definition = comptime blk: {
        var builder = Definition.init();
        builder.output(builder.scalar(.f32, 2.5));
        builder.output(builder.full(.i8, &.{ 2, 3 }, -4));
        break :blk builder.finish();
    };

    try std.testing.expectEqualSlices(usize, &.{}, definition.tensors[0].value.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, definition.tensors[2].value.shape.slice());
    switch (definition.tensors[0].origin) {
        .literal => |value| try std.testing.expectEqual(@as(f32, 2.5), value.get(.f32)),
        else => return error.TestUnexpectedResult,
    }
    switch (definition.tensors[1].origin) {
        .literal => |value| try std.testing.expectEqual(@as(i8, -4), value.get(.i8)),
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expectEqual(@as(usize, 1), definition.node_count);
}
