test {
    _ = @import("network.zig");
    _ = @import("meta.zig");
    _ = @import("definition.zig");
    _ = @import("model.zig");
    _ = @import("inspect.zig");
    _ = @import("lifetime_analysis.zig");
    _ = @import("memory_reuse.zig");
    _ = @import("tensor_view.zig");
    _ = @import("tensor_ops/tests.zig");
    _ = @import("validation.zig");
}
