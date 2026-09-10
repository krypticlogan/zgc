# Documentation

These documents describe the public model-construction flow, compiler pipeline,
runtime storage, supported operations, constraints, and tooling.

- [Architecture](architecture.md) — definition, counting, graph lowering,
  memory planning, and generated model execution.
- [Design constraints](design-constraints.md) — compile-time specialization,
  source ownership, execution, and memory invariants.
- [Operation semantics](operations.md) — shape, dtype, broadcasting,
  reduction, concatenation, and structural-view contracts.
- [Model inspection](inspection.md) — writer-based representations and the
  model-specific inspection CLI.
- [Generated model artifacts](model-artifacts.md) — minimal model executables
  and their stable exported symbols.
- [Development state](development-state.md) — implemented capabilities,
  limitations, and tests.
- [Repository README](../README.md) — public API example and common commands.
- [Sandbox](../sandbox/README.md) — model consumption, interactive inference,
  inspection, and generated-code analysis.
- [Benchmarks](../benchmarks/README.md) — cases, methodology, and recorded
  results.
