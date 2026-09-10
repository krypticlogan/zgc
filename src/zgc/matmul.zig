/// Compile-time matmul traversal selected by graph lowering.
pub const Strategy = enum {
    output_columns,
    contracted_axis,
    output_rows,
    scalar,
};

/// Lowered matmul configuration. Blocking and specialized small-batch
/// parameters can be added here without changing the operation dispatch
/// boundary.
pub const Plan = struct {
    strategy: Strategy,
};
