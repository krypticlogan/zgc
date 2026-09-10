const zgc = @import("zgc");

const Sources = enum(usize) { input };
const Definition = zgc.DefinitionBackend(Sources, .{
    .max_rank = 1,
    .max_nodes = 1,
    .max_tensors = 2,
    .max_input_refs = 1,
    .max_outputs = 1,
});

pub const Model = model: {
    var builder = Definition.init();
    const input = builder.input(.input, .f32, &.{4});
    builder.output(builder.relu(input));
    break :model builder.finish().model();
};
