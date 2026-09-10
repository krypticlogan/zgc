const zgc = @import("zgc");

pub const batch_size = 1;
pub const input_size = 8;

pub const Sources = enum(usize) {
    input,
};

pub const Definition = zgc.DefinitionBackend(Sources, .{
    .max_rank = 2,
    .max_nodes = 12,
    .max_tensors = 19,
    .max_input_refs = 18,
    .max_outputs = 1,
});

fn defineGraph(builder: *Definition) void {
    const input = builder.input(.input, .f32, &.{ batch_size, input_size });
    builder.output(builder.relu(input));
}

pub const definition = blk: {
    var builder = Definition.init();
    defineGraph(&builder);
    break :blk builder.finish();
};

pub const Model = definition.modelWith(.{
    .input = zgc.Source.bound,
});
