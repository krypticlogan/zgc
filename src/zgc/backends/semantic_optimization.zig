/// Apply graph-level semantic rewrites before fusion. This stage currently
/// preserves the validated graph; every rewrite added here must preserve
/// observable tensor values and source/output contracts.
pub fn SemanticOptimizationBackend() type {
    return struct {
        pub fn optimize(comptime Validated: type, comptime analysis: anytype) type {
            _ = analysis;
            return struct {
                pub const graph = Validated.graph;
            };
        }
    };
}
