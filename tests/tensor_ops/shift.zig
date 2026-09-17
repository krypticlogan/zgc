const std = @import("std");
const zgc = @import("zgc");

const Sources = enum(usize) { input, fill };
const Definition = zgc.DefinitionBackend(Sources, .{
    .max_rank = 2,
    .max_nodes = 4,
    .max_tensors = 6,
    .max_input_refs = 5,
    .max_outputs = 4,
});

const boundary_model = model: {
    var b = Definition.init();
    const input = b.input(.input, .f32, &.{ 2, 3 });
    const fill = b.input(.fill, .f32, &.{});
    b.output(b.shift(input, &.{ 1, -1 }, .wrap));
    b.output(b.shift(input, &.{ 1, -1 }, .edge));
    b.output(b.shift(input, &.{ 1, -1 }, .reflect));
    b.output(b.shift(input, &.{ 1, -1 }, .{ .constant = fill }));
    break :model b.finish().model();
};

test "shift maps positive and negative offsets across every boundary mode" {
    var model = boundary_model.init();
    try model.copyInput(.input, &.{ 1, 2, 3, 4, 5, 6 });
    try model.copyInput(.fill, &.{-9});
    model.run();
    try std.testing.expectEqualSlices(f32, &.{ 5, 6, 4, 2, 3, 1 }, model.outputView(0).contiguousSlice().?);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 3, 2, 3, 3 }, model.outputView(1).contiguousSlice().?);
    try std.testing.expectEqualSlices(f32, &.{ 5, 6, 5, 2, 3, 2 }, model.outputView(2).contiguousSlice().?);
    try std.testing.expectEqualSlices(f32, &.{ -9, -9, -9, 2, 3, -9 }, model.outputView(3).contiguousSlice().?);

    try model.copyInput(.fill, &.{7});
    model.run();
    try std.testing.expectEqualSlices(f32, &.{ 7, 7, 7, 2, 3, 7 }, model.outputView(3).contiguousSlice().?);
}

const StridedDefinition = zgc.DefinitionBackend(enum(usize) { input }, .{
    .max_rank = 2,
    .max_nodes = 2,
    .max_tensors = 3,
    .max_input_refs = 2,
    .max_outputs = 1,
});

const strided_model = model: {
    var b = StridedDefinition.init();
    const input = b.input(.input, .f32, &.{ 2, 3 });
    const transposed = b.transpose(input, 0, 1);
    b.output(b.shift(transposed, &.{ 0, 1 }, .wrap));
    break :model b.finish().model();
};

test "shift reads a strided source view in logical coordinates" {
    var model = strided_model.init();
    try model.copyInput(.input, &.{ 1, 2, 3, 4, 5, 6 });
    model.run();
    try std.testing.expectEqualSlices(f32, &.{ 4, 1, 5, 2, 6, 3 }, model.outputView(0).contiguousSlice().?);
}

const SingletonDefinition = zgc.DefinitionBackend(enum(usize) { input }, .{
    .max_rank = 1,
    .max_nodes = 1,
    .max_tensors = 2,
    .max_input_refs = 1,
    .max_outputs = 1,
});

const singleton_model = model: {
    var b = SingletonDefinition.init();
    const input = b.input(.input, .i8, &.{1});
    b.output(b.shift(input, &.{101}, .reflect));
    break :model b.finish().model();
};

test "reflection of a singleton axis stays on its only element" {
    var model = singleton_model.init();
    try model.copyInput(.input, &.{42});
    model.run();
    try std.testing.expectEqualSlices(i8, &.{42}, model.outputView(0).contiguousSlice().?);
}
