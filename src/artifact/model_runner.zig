const std = @import("std");
const generated_model = @import("model");

const Model = generated_model.Model;

fn zgc_run_model(model: *Model) callconv(.c) void {
    model.run();
}

fn zgc_model_size() callconv(.c) usize {
    return @sizeOf(Model);
}

fn zgc_model_alignment() callconv(.c) usize {
    return @alignOf(Model);
}

fn zgc_model_mutable_bytes() callconv(.c) usize {
    return Model.memory_plan.byte_count;
}

comptime {
    @export(&zgc_run_model, .{ .name = "zgc_run_model" });
    @export(&zgc_model_size, .{ .name = "zgc_model_size" });
    @export(&zgc_model_alignment, .{ .name = "zgc_model_alignment" });
    @export(&zgc_model_mutable_bytes, .{ .name = "zgc_model_mutable_bytes" });
}

/// The executable is an artifact container. Applications provide model
/// initialization, source binding, and output handling separately.
pub fn main() void {
    std.mem.doNotOptimizeAway(&zgc_run_model);
    std.mem.doNotOptimizeAway(&zgc_model_size);
    std.mem.doNotOptimizeAway(&zgc_model_alignment);
    std.mem.doNotOptimizeAway(&zgc_model_mutable_bytes);
}
