const CountingBackend = @import("counting.zig").CountingBackend;
const GraphBackend = @import("graph.zig").GraphBackend;
const GraphAnalysis = @import("analysis.zig").GraphAnalysis;
const SemanticOptimizationBackend = @import("semantic_optimization.zig").SemanticOptimizationBackend;
const FusionBackend = @import("fusion.zig").FusionBackend;
const LayoutPlanningBackend = @import("layout_planning.zig").LayoutPlanningBackend;
const KernelPlanningBackend = @import("kernel_planning.zig").KernelPlanningBackend;
const FinalValidationBackend = @import("final_validation.zig").FinalValidationBackend;
const LifetimeAnalysis = @import("lifetime.zig").LifetimeAnalysis;
const ValidationBackend = @import("validation.zig").ValidationBackend;
const Model = @import("../model.zig").Model;
const Source = @import("../source.zig");

/// Specializes a definition through semantic optimization, fusion, layout and
/// kernel planning, final validation, lifetime analysis, and model generation.
pub fn model(
    comptime Definition: type,
    comptime definition: Definition,
    comptime source_configuration: anytype,
) type {
    const compile_work = 10_000 + definition.node_count *
        (definition.tensor_count + definition.input_ref_count + Definition.max_rank + 16) * 64;
    @setEvalBranchQuota(compile_work);
    const capacity = CountingBackend(Definition).count(definition);
    const raw_graph = GraphBackend(Definition, capacity).build(definition);
    const EarlyValidated = ValidationBackend(capacity).validate(raw_graph);
    const initial_analysis = GraphAnalysis().analyze(EarlyValidated);
    const SemanticOptimized = SemanticOptimizationBackend().optimize(EarlyValidated, initial_analysis);
    const SemanticValidated = ValidationBackend(capacity).validate(SemanticOptimized.graph);
    const graph_analysis = GraphAnalysis().analyze(SemanticValidated);
    const Fused = FusionBackend().form(SemanticValidated, graph_analysis);
    const layout_graph = LayoutPlanningBackend(capacity).plan(Fused, graph_analysis);
    const LayoutValidated = ValidationBackend(capacity).validate(layout_graph);
    const executable_program = KernelPlanningBackend(capacity).plan(LayoutValidated, Fused);
    const FinalValidated = FinalValidationBackend(capacity).validate(executable_program);
    const graph = FinalValidated.graph;
    const lifetime_analysis = LifetimeAnalysis().analyze(FinalValidated);
    const SourcePlan = Source.Plan(
        Definition.Source,
        capacity,
        graph,
        source_configuration,
    );
    return Model(
        Definition.Source,
        capacity,
        EarlyValidated,
        SemanticValidated,
        graph_analysis,
        FinalValidated,
        lifetime_analysis,
        SourcePlan,
    );
}
