const CountingBackend = @import("counting.zig").CountingBackend;
const GraphBackend = @import("graph.zig").GraphBackend;
const LifetimeAnalysis = @import("lifetime.zig").LifetimeAnalysis;
const ValidationBackend = @import("validation.zig").ValidationBackend;
const Model = @import("../model.zig").Model;
const Source = @import("../source.zig");

/// Specializes a definition through counting, lowering, validation, lifetime
/// analysis, source planning, and model generation.
pub fn model(
    comptime Definition: type,
    comptime definition: Definition,
    comptime source_configuration: anytype,
) type {
    const capacity = CountingBackend(Definition).count(definition);
    const lowered_graph = GraphBackend(Definition, capacity).build(definition);
    const Validated = ValidationBackend(capacity).validate(lowered_graph);
    const graph = Validated.graph;
    const lifetime_analysis = LifetimeAnalysis().analyze(Validated);
    const SourcePlan = Source.Plan(
        Definition.Source,
        capacity,
        graph,
        source_configuration,
    );
    return Model(
        Definition.Source,
        capacity,
        Validated,
        lifetime_analysis,
        SourcePlan,
    );
}
