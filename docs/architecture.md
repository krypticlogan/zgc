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
                 concrete graph and layouts
                            │
                            ▼
                    ValidationBackend
              checked graph and static views
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

Internal namespaces such as `zgc.nn` and `zgc.img` build on this same front end. i.e. Neural
network layers expand into core sources and operations, while image helpers
declare core inputs with explicit rank-4 layout conventions. They do not own a
separate runtime or storage representation.

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

## Counting and graph lowering

The counting backend reads the completed definition and derives exact graph
capacities. In particular, the graph's rank capacity is the largest rank
actually used, rather than the definition's rank bound. Source storage uses
direct enum indexing, so its capacity is the highest referenced source index
plus one.

The graph backend then lowers tensors in definition order. Compute results own
dense storage. A rank-2 matmul keeps the public logical contract
`[M, K] * [K, N]`. Parameter and constant right-hand sides are packed with
physical strides `[1, K]`, making each logical output column contiguous.
Matmuls whose leading dimension can fill a target SIMD vector also select a
first-axis-contiguous lhs and result layout; that result layout propagates
through compatible add, ReLU, exp, and softmax results. Other compute results
use row-major storage. Transpose view operations preserve their source storage
tensor and produce an aliasing layout with adjusted shape and strides.

The concrete graph stores fixed arrays of nodes, tensor metadata, flattened
input references, outputs, and sources. Node order is execution order.

The validation backend checks the lowered graph's inferred output shapes and
dtypes. It also verifies that each matmul traversal plan is compatible with its
selected layouts. A validated program provides mutable and read-only tensor
view types whose shape, strides, base offset, element count, and layout traits
are compile-time properties.

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

`definition.modelWith(...)` selects non-default storage by source-enum tag.
`zgc.Source.embed(bytes)` accepts logical row-major parameter or constant bytes
and compile-time packs them into the lowered source layout.
`zgc.Source.embedPacked(bytes)` accepts bytes already in that physical layout.
Both place the resulting storage in read-only program data.
`zgc.Source.bound` makes an input borrow storage supplied to each model
instance. Dtype is enforced by the typed copy/bind APIs, and element or byte
counts are checked before a source is accepted.

## Kernel dispatch

Each compute node resolves prevalidated static input and output view types and
dispatches through `Op.Compute.execute`. Graph lowering selects physical
layouts, while kernels traverse contiguous axes in target-native SIMD chunks
with scalar tails. Runtime view state contains storage and any cursor offset
introduced by runtime-selected subviews; fixed tensor geometry is carried by
the type.

Matmul lowering also records a concrete traversal strategy in the operation's
compile-time plan. Generated models dispatch directly to that strategy and do
not branch over layout metadata at runtime. Direct low-level operation calls
must provide a concrete strategy. The plan contains the traversal strategy and
is the configuration boundary for strategy-specific kernels.

Shape, dtype, rank, axis, and lowered-plan compatibility checks belong to the
definition and validation backends. Execution kernels assume those contracts.
Dynamic `Tensor.View` and `Tensor.ConstView` types remain available when a
low-level caller intentionally supplies runtime geometry.

Kernels are grouped by family:

| Family | Implemented operations |
| --- | --- |
| Literals | Rank-zero scalar values and zero-stride filled-tensor expansion |
| Elementwise | ReLU, exp, add, sub, mul, div; trailing-axis broadcasting for binary operations |
| Contraction | Rank-2 matmul |
| Reduction | Sum, mean, min, and max over compile-time axis sets |
| Special | Softmax over one axis |
| Concatenation | Materialized output with contiguous block-copy and static strided paths |
| Layout | Compile-time transpose, permutation, reshape, flatten, squeeze, unsqueeze, and slicing inference |

Binary arithmetic aligns shapes from the trailing axis. Equal extents are
paired directly, singleton extents broadcast with zero strides, and absent
leading axes behave as singleton dimensions. Reduction axes are normalized,
deduplicated, and encoded at definition time. `keep_dims` retains reduced axes
as singleton dimensions so reduction outputs can broadcast back over inputs.

Scalar literals are immutable, source-free rank-zero tensors embedded in the
generated program. They do not reserve model memory or execute a kernel. A
filled tensor is a scalar literal followed by a zero-stride broadcast view, so
its storage remains one element regardless of logical shape.

Structural operations create aliases and do not execute kernels. Squeeze and
unsqueeze preserve arbitrary source strides. Flatten requires its selected
axis range to be logically contiguous. General reshape currently requires a
logically row-major contiguous source because it must preserve element order
without copying. Permutation lowers to transpose aliases. Slicing uses
compile-time positive bounds and steps to produce an offset strided alias.

Elementwise kernels use SIMD for row-major tensors and matching dense axis
permutations. Trailing-vector binary arithmetic also vectorizes across a contiguous first
axis, covering bias operations on batch-oriented matmul results. Selected
reduction and contraction paths use SIMD. Generic view traversal handles
offsets and positive or negative strides where the relevant kernel supports
them.
