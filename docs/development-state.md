# Development state

ZGC is functional but pre-release. The compile-time definition-to-model path,
runtime execution, tests, examples, and operation benchmark suite are working
on Zig 0.16.0. The API should still be expected to change.

## Implementation roadmap

1. Add dtype conversion and settle the tensor-index dtype.
2. Build argmin/argmax and runtime-indexed gather operations.
3. Generalize matmul to broadcastable batch dimensions with compile-time
   traversal plans.
4. Build convolution, pooling, and specialized stencil lowering over static
   padding and window geometry.
5. Implement constant folding, common-subexpression elimination, dead-node
   elimination, algebraic simplification, and store/epilogue sinking.
6. Introduce explicitly bounded runtime extents where they preserve static
   kernel specialization and memory planning.

## Implemented

### Definition and compilation

- A concrete, typed `DefinitionBackend` is the public model-builder.
- Models are defined once and compiled with `definition.model()`.
- Internal counting derives exact capacities within configurable definition
  bounds.
- Raw graph construction preserves stable tensor IDs, source indices, operation
  order, outputs, shapes, and dtypes.
- Semantic optimization, fusion, layout planning, and kernel planning have
  distinct pass boundaries. Final validation checks executable layouts and
  kernel-plan contracts.
- Executable programs use multi-output invocation records. Compatible sibling
  reductions share one invocation and write independent results.
- Generated model types retain raw and optimized graphs for inspection. Use
  counts, output markers, and legal elementwise fusion edges are available from
  graph analysis.
- Lifetime analysis consolidates alias uses onto storage roots and provides
  half-open intervals to memory planning.
- Invalid ranks, shapes, axes, dtypes, input counts, and broadcasting are
  rejected at compile time where the relevant metadata is static.

### Tensors and layouts

- Static-geometry mutable/read-only views for generated models and
  runtime-geometry views for explicit low-level use.
- Rank-zero through bounded-rank shapes.
- Contiguous layouts, offsets, arbitrary strides, negative strides, transpose
  aliases, axis slices, and broadcast views.
- `f32`, `f16`, `i8`, and `bool` dtypes. Numeric dtypes provide scalar/vector
  mappings and accumulation helpers; booleans remain strict predicates.

### Operations

| Operation | Support |
| --- | --- |
| Scalar/full | Source-free rank-zero literals and zero-stride filled-tensor aliases |
| ReLU | SIMD over matching dense layouts and generic strided traversal; float and signed integer |
| Exp | Floating-point tensors; SIMD over matching dense layouts and strided traversal |
| Unary math | Floating-point negation, absolute value, square root, logarithm, and reciprocal |
| Add/sub/mul/div | Matching numeric dtypes and trailing-axis broadcasting; div is floating-point |
| Minimum/maximum/clamp | Elementwise numeric bounds with trailing-axis broadcasting |
| Comparisons | Broadcast equality and ordered comparisons producing boolean tensors |
| Boolean/selection | Strict logical operations and broadcast `where` selection |
| Copy/contiguous | Fresh storage with lowering-selected or logical row-major layout |
| Pad/shift/windows | Materialized constant padding, shifts with wrap/edge/reflect/constant boundaries, and zero-copy overlapping trailing-axis windows |
| Matmul | Rank-2 tensors with contiguous and strided inputs/outputs; packed right-hand parameters and compile-time-selected native-width SIMD traversal |
| Sum/mean/min/max | Compile-time single- or multi-axis reduction, optional retained dimensions, and strided traversal; mean is floating-point |
| Softmax | Stable single-axis floating-point implementation, including strided axes |
| Concat | Matching-rank and matching-dtype inputs materialized along one compile-time axis |
| Structural views | Transpose, permutation, reshape, flatten, squeeze, unsqueeze, slicing, and explicit broadcasting aliases with no runtime kernels |

### Domain abstractions

- `zgc.nn.Dense` declares weights and bias sources and expands to matmul, add,
  and an optional core activation.
- `zgc.nn.Sequential` composes graph-layer definitions in order.
- Dense layers accept `[input, output]` or `[output, input]` parameter storage.
- `zgc.img.Dimensions` and `zgc.img.input` provide channel-first and
  channel-last rank-4 input conventions.

### Model and storage

- One inline, aligned memory allocation per model instance with lifetime-based
  intermediate-region reuse.
- Compile-time memory plan shared by runtime instances.
- Typed logical source/input packing, borrowed runtime inputs, logical or
  prepacked embedded read-only parameters/constants, sequential graph
  execution, and typed output views.
- Views alias their root tensor's storage without adding another allocation.
- Writer-based capacity, graph, tree, memory-plan, and bounded model-memory
  inspection through `zgc.Inspect`.

### Tooling

- Unit and end-to-end tests through `zig build test`.
- Compile-only check through `zig build check`.
- Maintained root benchmark suite with shape and layout comparisons.
- Reusable model-specific inspection CLI and generated-model runner modules.
- Standalone example packages with interactive applications and model artifact
  tooling.

## Important limitations

- Shapes and extents are currently compile-time fixed; bounded runtime extents
  are a design goal, not an implemented feature.
- Runtime-bound inputs currently report a missing binding when their view is
  first resolved during execution rather than through a separate run preflight.
- Layout selection is limited to packed matmul right-hand parameters, the
  matmul batch heuristic, and compatible result propagation. Fusion forms
  single-consumer pointwise maps, producer-to-reduction regions, and compatible
  sibling reductions. Constant folding, common-subexpression elimination,
  dead-node elimination, and general cost-based planning are not implemented.
- Softmax and reductions traverse propagated layouts correctly, but do not yet
  have a dedicated batch-oriented lowering and kernel strategy for every axis.
- Matmul is a direct specialized kernel, not a tuned BLAS replacement.
- No training, automatic differentiation, dynamic control flow, or device/GPU
  backend exists.
- External parameter packs and memory-mapped parameter bindings are not yet
  implemented; parameters can currently be owned or compile-time embedded.
- Public naming and module boundaries remain subject to change before a stable
  release.

## Test coverage

The test suite exercises:

- typed definition construction and exact capacity counting;
- source indexing, graph materialization, and multiple ranks/outputs;
- shape, dtype, axis, rank, and broadcasting validation;
- aligned memory planning and source loading;
- model execution and typed output access;
- contiguous, offset, broadcast, transposed, and negative-stride views;
- unary math, arithmetic, comparisons, boolean logic, selection, matmul,
  reductions, softmax, padding, and overlapping-window behavior;
- transpose, permutation, reshape, flatten, squeeze, unsqueeze, and slicing alias behavior;
- SIMD tails and strided fallbacks;
- end-to-end execution across graph-produced aliasing views.
