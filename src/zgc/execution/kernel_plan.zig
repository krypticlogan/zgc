const Region = @import("../optimization/fusion/region.zig");

/// Data-only physical plan selected after fusion and layout planning.
pub const KernelPlan = union(enum) {
    map: MapPlan,
    reduction: ReductionPlan,
    contraction: ContractionPlan,
};

pub const MapPlan = struct {
    region: Region.Map,
    schedule: Schedule,

    pub const Schedule = struct {
        axis_order: []const u8,
        traversal: Traversal,
        vector_axis: ?u8,
        vector_width: usize,
        unroll: usize = 1,
    };

    pub const Traversal = enum { contiguous, strided };
};

pub const ReductionPlan = struct {
    region: Region.Reduction,
    schedule: Schedule,

    pub const Schedule = struct {
        outer_axis_order: []const u8,
        reduction_axis_order: []const u8,
        vector_axis: ?u8,
        vector_width: usize,
        accumulator_lanes: usize,
        unroll: usize = 1,
    };
};

pub const ContractionPlan = struct {
    strategy: Strategy,

    pub const Strategy = enum {
        output_columns,
        contracted_axis,
        output_rows,
        scalar,
    };
};
