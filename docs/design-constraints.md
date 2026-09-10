# Design constraints

ZGC specializes a model from compile-time graph metadata while keeping runtime
data ownership explicit. The following constraints define the current model
construction and execution flow.

## Compile-time model structure

- Architecture, tensor ranks, shapes, dtypes, layouts, and operation order are
  known at compile time.
- Tensor dimensions are positive; zero-length dimensions are rejected during
  definition.
- Definition bounds limit front-end construction. The counting pass derives the
  exact capacities used by graph lowering and memory planning.
- Shape and operation compatibility errors are reported during compilation when
  their inputs are statically known.
- The validation backend checks the lowered graph before executable model types
  are instantiated.
- The generated model type contains a fixed graph and memory plan; execution
  does not interpret or allocate graph nodes.

## Source ownership

Every graph source has one storage policy selected by `definition.modelWith`:

- Owned sources receive an aligned region in model memory and are populated
  through `copyInput` or `copySource`. Values use logical row-major order and
  are packed into the compiled physical layout.
- Bound inputs borrow a caller-owned slice through `bindInput`. The slice must
  use the layout reported by `sourceLayout`, and remain valid and unchanged for
  the duration of `run()`.
- Embedded parameters and constants use read-only program data.
  `Source.embed`, commonly used with `@embedFile`, accepts logical row-major
  bytes and compile-time packs them into the selected layout;
  `Source.embedPacked` accepts bytes already in that physical layout.

Unspecified policies are owned. Bound and embedded sources do not consume space
in the model's mutable memory plan.

## Views and execution

- Operations receive read-only input views and write only to their designated
  output storage.
- Layout-changing graph operations such as transpose create aliases rather than
  copying tensor data.
- Reshape, flatten, squeeze, and unsqueeze are aliasing views. Reshape requires
  logical row-major contiguity, while flatten requires contiguity only within
  its collapsed axis range.
- Axis permutation lowers to transpose aliases. Static slices use positive
  bounds and steps and retain the source storage root.
- Generated-model views carry shape, strides, base offset, element count, and
  layout traits in their types. Their runtime state contains storage and any
  cursor offset introduced by runtime-selected subviews.
- Dynamic views retain runtime geometry for explicit low-level use.
- Lowering may choose a first-axis-contiguous physical layout for eligible
  rank-2 matmuls and propagate it through compatible dense operations.
- Matmul parameter and constant right-hand sides retain logical `[K, N]` shape
  while lowering may store them output-major with physical strides `[1, K]`.
- Generated matmuls carry a compile-time traversal plan selected from concrete
  graph layouts. Direct low-level calls must select a concrete strategy.
- Concatenation materializes distinct contiguous output storage. Its axis and
  input geometry are validated and specialized before execution.
- Filled tensors alias one scalar storage element through zero strides. They do
  not allocate or initialize storage proportional to their logical shape.
- `run()` executes the fixed operation list sequentially. Runtime input values
  may change between runs without rebuilding the model type.

## Memory

- A model instance owns one inline, aligned mutable byte array.
- The compile-time memory plan assigns regions to owned source and result
  tensors; aliases do not receive separate regions.
- Model-owned sources and graph outputs retain persistent regions.
- Intermediate regions may overlap when their validated half-open lifetimes do
  not overlap. Free spans are alignment-aware, split when partially consumed,
  and coalesced when adjacent.
- Heap allocation is not required for model initialization or execution.

See [architecture](architecture.md) for the compilation pipeline and
[development state](development-state.md) for supported operations and current
limitations.
