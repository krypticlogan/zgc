# Architecture

ZGC specializes a statically defined tensor program into an executable Zig
model type. Definition bounds make compile-time storage possible; a counting
pass then removes unused capacity before the graph and model are materialized.

```text
typed model function
        │
        ▼
DefinitionBackend ──► completed definition
                            │
                            ▼
                     CountingBackend
                     exact capacities
                            │
                            ▼
                       GraphBackend
                    raw semantic graph
                            │
                            ▼
                    ValidationBackend
                    early validation
                            │
                            ▼
                      GraphAnalysis
                  uses and fusion edges
                            │
                            ▼
             SemanticOptimizationBackend
                 semantic graph rewrites
                            │
                            ▼
                    ValidationBackend
                 semantic revalidation
                            │
                            ▼
                      GraphAnalysis
               optimized graph use facts
                            │
                            ▼
                     FusionBackend
                logical fusion regions
                            │
                            ▼
                LayoutPlanningBackend
                  physical tensor layouts
                            │
                            ▼
                    ValidationBackend
                   layout validation
                            │
                            ▼
                KernelPlanningBackend
                 executable kernel plans
                            │
                            ▼
                FinalValidationBackend
             checked executable graph/views
                            │
                            ▼
                    LifetimeAnalysis
                 storage-live intervals
                            │
                            ▼
                       MemoryPlan
                reusable aligned regions
                            │
                            ▼
                     executable Model type
```

Calling `definition.model()` performs every stage after definition. Counting
and compiler-analysis backends are internal implementation details and are not
exported as public model-building APIs.

## Definition

`DefinitionBackend(SourceKey, limits)` is the typed front end. `SourceKey` must
be an enum, giving every input, parameter, or constant a stable compile-time
index. Its operation methods consume and return one concrete tensor-value type
whose metadata contains an ID, dtype, and bounded shape.

Internal namespaces such as `zgc.nn` and `zgc.img` build on this same front end.
Neural-network layers expand into core sources and operations, while image
helpers declare core inputs with explicit rank-4 layout conventions. They do
not own a separate runtime or storage representation.

The definition records:

- source tensors and their source kinds;
- compute and view operations;
- flattened input references;
- inferred dtype and shape metadata;
- graph outputs.

Operation inputs are validated while operations are added. `finish()` returns
the completed immutable definition value.

Definition limits cover maximum rank, nodes, tensors, input
references, and outputs. Defaults support small models; larger definitions can
override individual fields. Exceeding a bound is a compile error.

## Counting, graph construction, and optimization

The counting backend reads the completed definition and derives exact graph
capacities. In particular, the graph's rank capacity is the largest rank
actually used, rather than the definition's rank bound. Source storage uses
direct enum indexing, so its capacity is the highest referenced source index
plus one.

The graph backend records a raw semantic graph in definition order. Compute
results initially use canonical dense storage. View operations preserve their
source storage tensor and derive aliasing shape and strides from that semantic
layout. Early validation checks operation shapes, dtypes, and view aliases
before optimization relies on them.

Graph analysis records tensor use counts, graph outputs, and legal
single-consumer elementwise fusion edges. Semantic optimization owns
value-preserving graph rewrites. Fusion selects pointwise map regions,
producer-to-reduction regions, and compatible sibling reductions. Layout
planning rebuilds the semantic graph with selected physical layouts and packed
parameters. Kernel planning converts that result into an executable program
with concrete, data-only kernel plans. A rank-2 matmul retains the logical
contract `[M, K] * [K, N]` while eligible parameter and constant right-hand
sides use physical strides `[1, K]`. Batch-oriented layouts propagate through
compatible operations. Final validation checks layouts and execution plans
before kernels are instantiated.

Semantic compute nodes use the operation representation in `operations/`.
Executable compute nodes retain unchanged operations as `direct` semantic
operations. Specialized computation uses a `KernelPlan` classified as map,
reduction, or contraction. Plans contain compile-time data only;
`ExecutableCompute` dispatches them to their kernel family. Matmul lowers to a
contraction plan. Map regions combine single-consumer pointwise
expressions into one traversal. Reduction regions combine compatible
pointwise producers and sibling accumulators over a shared domain.

The semantic graph stores fixed arrays of nodes, tensor metadata, flattened
input references, outputs, and sources. The executable program stores a fixed
sequence of invocations with flattened input and output references. An
invocation may name multiple outputs, allowing sibling reductions to share one
traversal without representing secondary stores as no-op nodes.

The generated model retains `raw_graph`, `semantic_graph`, and the executable
`optimized_graph` as compile-time inspection metadata. `build_graph` refers to
the executable program. A final validated program provides mutable and
read-only tensor view types
whose shape, strides, base offset, element count, and layout traits are
compile-time properties.

## Memory planning and model generation

`MemoryPlan` assigns one aligned byte region to each storage-owning tensor.
Aliasing views point at their root storage tensor's region. The generated model
contains one inline byte array sized and aligned by that plan. Model-owned
sources and compute results receive regions in this array; embedded parameters,
embedded constants, and runtime-bound inputs remain external to it.

Lifetime analysis records a half-open node interval for each storage root and
propagates alias uses to that root. Model-owned sources and output roots remain
persistent. The planner releases expired intermediate regions, coalesces
adjacent free spans, and places new tensors into the smallest aligned span that
fits. Oversized spans are split around the allocation. If no span fits, the
planner extends the model's storage high-water mark. Execution does not
allocate.

The model API provides:

- `init()` to zero-initialize model memory;
- `copyInput(key, values)` to pack a logical row-major runtime input into owned storage;
- `copySource(key, values)` to pack logical row-major values into any model-owned source;
- `bindInput(key, values)` to borrow input already stored in the compiled physical layout;
- `sourceLayout(key)` to query that source layout;
- `run()` to execute compute nodes in graph order;
- `outputView(index)` to retrieve a typed read-only view.

`zgc.Inspect` consumes the model's compile-time graph and memory-plan metadata
without adding rendering responsibilities to the model, graph, operation,
tensor, or storage types. It also renders bounded mutable memory from a model
instance when requested.

`zgc_model_runner` specializes a minimal executable around a consumer-provided
model module. The generated artifact exports a stable execution symbol and
model layout metadata while leaving initialization, runtime source binding, and
output handling to the application.

View nodes do not execute kernels. Their result layouts are resolved during
graph construction, and downstream compute kernels receive static-geometry
views into the aliased storage.

`definition.modelWith(...)` accepts a typed slice of source-enum tags and
storage bindings to select non-default storage.
`zgc.Source.embed(bytes)` accepts logical row-major parameter or constant bytes
and compile-time packs them into the lowered source layout.
`zgc.Source.embedPacked(bytes)` accepts bytes already in that physical layout.
Both place the resulting storage in read-only program data.
`zgc.Source.bound` makes an input borrow storage supplied to each model
instance. Dtype is enforced by the typed copy/bind APIs, and element or byte
counts are checked before a source is accepted.

## Kernel dispatch

Each compute node resolves prevalidated static input and output view types and
dispatches through its executable operation. Optimization selects physical
layouts, while kernels traverse contiguous axes in target-native SIMD chunks
with scalar tails. Runtime view state contains storage and any cursor offset
introduced by runtime-selected subviews; fixed tensor geometry is carried by
the type.

Matmul kernel planning records a concrete traversal strategy in a data-only
contraction plan. Generated models dispatch directly to that strategy and do not
branch over layout metadata at runtime. Direct semantic matmul execution uses
the general scalar kernel.

Map execution uses a compile-time instruction program built from
backend-independent elementwise descriptors. Instruction arity and accepted
dtypes belong to semantic operations; instruction references and traversal are
executable-graph details. The instruction sequence is unrolled at compile time,
so a fused kernel performs one output traversal without runtime opcode dispatch
or storage for instruction results.

Logical fusion regions and physical kernel plans are separate. Regions contain
expressions, reduction axes, accumulators, and stores; plans add traversal and
vectorization schedules after layout selection. Neither representation owns
execution behavior.

Shape, dtype, rank, axis, and plan compatibility checks belong to semantic and
final validation. Execution kernels assume those contracts.
Dynamic `Tensor.View` and `Tensor.ConstView` types remain available when a
low-level caller intentionally supplies runtime geometry.

Kernels are grouped by family:

| Family | Implemented operations |
| --- | --- |
| Literals | Rank-zero scalar values and zero-stride filled-tensor expansion |
| Elementwise | Numeric operations, comparisons, strict boolean logic, and conditional selection |
| Fused elementwise | Compile-time typed instruction programs with contiguous SIMD and static-stride fallback traversal |
| Materialization | Logical copy, row-major conversion, and constant padding |
| Shifting | Shape-preserving translation with wrap, edge, reflect, or constant boundaries |
| Contraction | Rank-2 matmul |
| Reduction | Sum, mean, min, and max over compile-time axis sets |
| Special | Softmax over one axis |
| Concatenation | Materialized output with contiguous block-copy and static strided paths |
| Layout | Compile-time transpose, permutation, reshape, flatten, squeeze, unsqueeze, slicing, broadcasting, and overlapping-window inference |

Binary arithmetic aligns shapes from the trailing axis. Equal extents are
paired directly, singleton extents broadcast with zero strides, and absent
leading axes behave as singleton dimensions. Reduction axes are normalized,
deduplicated, and encoded at definition time. `keep_dims` retains reduced axes
as singleton dimensions so reduction outputs can broadcast back over inputs.

Comparisons return boolean tensors. Logical operations accept only boolean
tensors, and `where` requires a boolean condition. Numeric tensors are never
interpreted through implicit truthiness rules.

Scalar literals are immutable, source-free rank-zero tensors embedded in the
generated program. They do not reserve model memory or execute a kernel. A
filled tensor is a scalar literal followed by a zero-stride broadcast view, so
its storage remains one element regardless of logical shape.

Structural operations create aliases and do not execute kernels. Squeeze and
unsqueeze preserve arbitrary source strides. Flatten requires its selected
axis range to be logically contiguous. General reshape requires a
logically row-major contiguous source because it must preserve element order
without copying. Permutation lowers to transpose aliases. Slicing uses
compile-time positive bounds and steps to produce an offset strided alias.
Windows append static neighborhood axes and may overlap within the same
storage root.

Elementwise kernels use SIMD for row-major tensors and matching dense axis
permutations. Trailing-vector binary arithmetic also vectorizes across a contiguous first
axis, covering bias operations on batch-oriented matmul results. Selected
reduction and contraction paths use SIMD. Generic view traversal handles
offsets and positive or negative strides where the relevant kernel supports
them.
