# Operation semantics

All model shapes, axes, dtypes, and operation attributes are compile-time
values. Inputs are read-only and compute operations write a distinct output.
View operations alias an existing storage root and execute no runtime kernel.

Tensor dimensions must be positive. Operations preserve their input dtype
unless stated otherwise.

## Scalars and filled tensors

`scalar(dtype, value)` creates an immutable, source-free rank-zero tensor whose
value is embedded in the generated program. It reserves no mutable model memory
and executes no runtime kernel.

`full(dtype, shape, value)` creates that scalar and broadcasts it to the target
shape. The result is a read-only zero-stride alias backed by one scalar element,
not a materialized array. `broadcastTo(tensor, shape)` exposes the same
trailing-axis expansion for any broadcast-compatible tensor.

## Elementwise operations

`relu`, `exp`, `add`, `sub`, `mul`, and `div` operate elementwise. `exp` and
`div` require floating-point tensors. The remaining operations support the
implemented floating-point and signed-integer dtypes.

Binary operands must have matching dtypes and use trailing-axis broadcasting:

- equal aligned extents are paired directly;
- an extent of one broadcasts across the other operand;
- absent leading axes behave as singleton dimensions;
- rank-zero tensors broadcast as scalars.

Broadcasting is represented with zero input strides and does not materialize
expanded operands.

## Matrix multiplication

`matmul` accepts two rank-two `f32` tensors with logical shapes `[M, K]` and
`[K, N]`, producing `[M, N]`. Lowering selects the traversal strategy and may
pack parameter or constant right-hand operands into an output-major physical
layout.

## Reductions

`sum`, `mean`, `min`, and `max` accept one or more axes. `null` selects every
axis. Negative axes are normalized relative to the input rank; duplicate,
empty, and out-of-range axis sets are rejected. Reduced axes are removed by
default or retained with extent one when `keep_dims` is true.

`mean` requires a floating-point tensor. `sum`, `min`, and `max` support the
implemented floating-point and signed-integer dtypes. Reduction geometry and
axis traversal are fixed in the generated kernel.

`softmax` accepts a floating-point tensor and one axis. It preserves shape and
uses a numerically stable shifted exponential calculation.

## Concatenation

`concat` accepts one or more tensors with matching ranks and dtypes. All
extents outside the selected axis must match. The output extent on that axis is
the sum of the corresponding input extents.

Concatenation materializes a new contiguous tensor. Generated contiguous views
use fixed block copies; other layouts use static strided traversal. Inputs are
never expanded or copied into temporary tensors.

## Structural views

Structural operations alias their source storage:

- `transpose` exchanges two axes and their strides.
- `permute` validates a complete unique axis ordering and lowers it to
  transpose aliases.
- `reshape` preserves element count and requires a logically row-major
  contiguous input.
- `flatten` collapses an inclusive axis range and requires only that range to
  be logically contiguous.
- `squeeze` removes a selected extent-one axis.
- `unsqueeze` inserts an extent-one axis at the selected position.
- `slice` selects a non-empty range with compile-time positive bounds and
  step, producing an offset strided alias.
- `broadcastTo` introduces leading or singleton expansion axes with zero
  strides.

Negative axes are accepted by definition-builder methods and normalized before
graph validation. Structural views receive no independent memory-plan region.
