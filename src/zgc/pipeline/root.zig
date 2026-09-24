const Construction = @import("construction.zig");
const Analysis = @import("analysis.zig");
const Planning = @import("executable_planning.zig");
const validation = @import("validation.zig");
const Model = @import("../model.zig").Model;
const Source = @import("../source.zig");

/// Run the compilation steps from a completed definition to a generated model.
pub fn model(
    comptime Definition: type,
    comptime definition: Definition,
    comptime source_configuration: anytype,
) type {
    const compile_work = 10_000 + definition.node_count *
        (definition.tensor_count + definition.input_ref_count + Definition.max_rank + 16) * 64;
    @setEvalBranchQuota(compile_work);
    const capacity = Construction.count(Definition, definition);
    const raw_graph = Construction.GraphConstruction(Definition, capacity).build(definition);
    const SemanticValidated = validation.Validation(capacity).validate(raw_graph);
    const semantic_analysis = Analysis.SemanticAnalysis().analyze(SemanticValidated);
    const fusion_candidates = Analysis.FusionAnalysis(capacity).analyze(
        SemanticValidated.graph,
        semantic_analysis,
    );
    const layout_candidates = Analysis.LayoutAnalysis(capacity).analyze(
        SemanticValidated.graph,
        semantic_analysis,
    );

    const executable_search = Planning.ExecutablePlanning(capacity).search(
        Definition.Source,
        SemanticValidated,
        semantic_analysis,
        fusion_candidates,
        layout_candidates,
        source_configuration,
    );
    const selected_candidate = executable_search.selected();
    const executable = selected_candidate.executable;
    const FinalValidated = validation.FinalValidation(capacity).validate(executable);
    const graph = FinalValidated.graph;

    const lifetime_analysis = Planning.LifetimeAnalysis().analyze(FinalValidated);
    const SourcePlan = Source.Plan(
        Definition.Source,
        capacity,
        graph,
        source_configuration,
    );
    return Model(
        Definition.Source,
        capacity,
        raw_graph,
        SemanticValidated,
        semantic_analysis,
        executable_search,
        FinalValidated,
        lifetime_analysis,
        SourcePlan,
    );
}
