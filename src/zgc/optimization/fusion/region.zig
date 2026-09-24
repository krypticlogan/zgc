const Expression = @import("expression.zig").Program;
const ReductionOperation = @import("../../operations/reduction.zig");

/// Logical computation assigned to one kernel invocation. Regions
/// describe what is computed together, without choosing a traversal plan.
pub const Region = union(enum) {
    map: Map,
    reduction: Reduction,
    contraction: Contraction,
};

pub const Map = struct {
    expressions: Expression,
    stores: []const Store,
};

pub const Reduction = struct {
    expressions: Expression,
    domain_shape: []const usize,
    reduction_axes: u64,
    keep_dims: bool,
    accumulators: []const Accumulator,
    stores: []const Store,

    pub const Accumulator = struct {
        combine: Combine,
        update: Expression.ValueRef,
        finalize: Finalize,
    };

    pub const Combine = ReductionOperation.Combine;
    pub const Finalize = ReductionOperation.Finalize;
};

pub const Contraction = struct {
    epilogue: ?Expression = null,
    stores: []const Store,
};

pub const Store = struct {
    output: usize,
    value: Expression.ValueRef,
};
