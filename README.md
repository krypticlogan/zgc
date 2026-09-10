# Zig Graph Compiler

ZGC is an allocation-free inference graph compiler for Zig. A model
architecture is defined at compile time, lowered to a fixed execution graph,
assigned an inline memory plan, and emitted as a specialized Zig type.
The binary is the model: graph traversal, tensor ranks, shapes, dtypes, layouts,
and kernel selection are compile-time-known.

The project targets Zig 0.16.0.

## Capabilities

- Typed, front-facing `DefinitionBackend` with enum-indexed sources.
- Counting, graph-lowering, and validation passes through `definition.model()`.
- Exact graph capacities derived from user-configurable definition bounds.
- Lifetime-planned, reusable inline model memory with no heap allocation during
  execution.
- Static-geometry model views and dynamic low-level views for contiguous,
  offset, broadcast, transposed, and generally strided layouts.
- Multiple graph inputs, parameters, constants, and outputs.
- Source-free scalar literals and storage-efficient zero-stride filled tensors.
- `f32`, `f16`, and `i8` tensor metadata and elementwise kernels where valid.
- ReLU, exp, add, sub, mul, div, matmul, sum, mean, min, max, softmax, and concatenation compute operations.
- Transpose, permutation, reshape, flatten, squeeze, unsqueeze, and static slicing view operations.
- Trailing-axis broadcasting for binary arithmetic and compile-time single- or multi-axis reductions.
- SIMD fast paths for contiguous kernels and generic strided traversal.
- Core-backed dense and sequential graph layers through `zgc.nn`.
- Rank-4 image input conventions through `zgc.img`.
- Operation and generated-model benchmarks, plus a standalone sandbox package.

See [development state](docs/development-state.md) for precise limitations and
[architecture](docs/architecture.md) for the compilation pipeline.

## Defining a model

`DefinitionBackend` is the model-building surface. Source keys and tensor
values are concrete types; user code does not run separately against counting
and graph builders.

```zig
const std = @import("std");
const zgc = @import("zgc");

const Sources = enum(usize) { input, weights };
const Definition = zgc.DefinitionBackend(Sources, .{ .max_rank = 2 });

fn define(builder: *Definition) void {
    const input = builder.input(.input, .f32, &.{ 4, 8 });
    const weights = builder.parameter(.weights, .f32, &.{ 8, 16 });
    builder.output(builder.relu(builder.matmul(input, weights)));
}

const definition = blk: {
    var builder = Definition.init();
    define(&builder);
    break :blk builder.finish();
};

const MyModel = definition.model();

pub fn main() !void {
    var model = MyModel.init();

    const input_values: [4 * 8]f32 = @splat(1);
    const weight_values: [8 * 16]f32 = @splat(0.25);
    try model.copyInput(.input, &input_values);
    try model.copySource(.weights, &weight_values);
    model.run();

    const output = model.outputView(0);
    std.debug.print("shape={any} data={any}\n", .{ output.shape, output.storage });
}
```

Definition limits have defaults and may be overridden at compile time. These
are front-end bounds, not final allocation sizes. The counting pass derives the
exact node, tensor, reference, output, source, and rank capacities before graph
construction and memory planning.

## Neural-network layers

`zgc.nn` composes higher-level layers through `DefinitionBackend`; it does not
provide a separate tensor runtime.

```zig
const Dense = zgc.nn.Dense(Sources);
const Classifier = zgc.nn.Sequential(&[_]Dense{
    .{ .weights = .w1, .bias = .b1, .output_size = 16, .activation = .relu },
    .{ .weights = .w2, .bias = .b2, .output_size = 10, .activation = .softmax },
});

const input = builder.input(.input, .f32, &.{ batch_size, input_size });
builder.output(Classifier.apply(builder, input));
```

Dense weights use logical `[input, output]` storage by default. Set
`.weight_layout = .output_input` for sources stored as `[output, input]`; the
layer adds an aliasing transpose before matmul.

`zgc.img.Dimensions` defines channel-first or channel-last rank-4 shapes, and
`zgc.img.input` declares a core graph input with that convention.

## Source storage

Sources use model-owned storage by default. Runtime inputs can be copied into
that storage with `copyInput`, while owned parameters and constants use the
typed `copySource` API shown above.

Parameters and constants may instead be embedded directly into the program:

```zig
const EmbeddedModel = definition.modelWith(.{
    .weights = zgc.Source.embed(@embedFile("weights.bin")),
});
```

The required byte length is derived from the source tensor's compile-time dtype
and shape and checked during compilation. `Source.embed` accepts raw logical
row-major, native-endian tensor data and packs it at compile time when lowering
selects another physical layout. `Source.embedPacked` accepts bytes already in
the layout reported by `Model.sourceLayout`. Embedded values remain read-only
and do not receive a region in the model's mutable memory plan.

Inputs can also borrow caller-owned runtime storage without a copy:

```zig
const BorrowingModel = definition.modelWith(.{
    .input = zgc.Source.bound,
    .weights = zgc.Source.embed(@embedFile("weights.bin")),
});

var model = BorrowingModel.init();
try model.bindInput(.input, runtime_values);
model.run();
```

The bound slice must:

- use the physical order reported by `BorrowingModel.sourceLayout(.input)`
- remain alive and unchanged while `run()` is executing.

It may be updated or rebound between runs. `copyInput` accepts logical
row-major values and packs them when lowering selects another layout.

Unspecified sources use model-owned storage, so `definition.model()` is
equivalent to an all-owned source plan.

## Build and test

```sh
zig build test
zig build check
```

The package exports a module named `zgc`. The [sandbox](sandbox/README.md) shows
how a separate Zig package consumes it through `b.dependency("zgc", ...)`.

## Benchmarks

Run the complete suite in `ReleaseFast`:

```sh
zig build benchmark -Doptimize=ReleaseFast
```

Select an individual case with `-Dop`, for example:

```sh
zig build benchmark -Dop=matmul-rhs-strided -Doptimize=ReleaseFast
```

See the [benchmark dashboard](benchmarks/README.md) for selectors, methodology,
and recorded results.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/zgc/backends/` | Definition, exact counting, graph lowering, validation, and pipeline orchestration |
| `src/zgc/kernels/` | Elementwise, reduction, contraction, layout, and special kernels |
| `src/zgc/` | Graph, tensor/view, operation, storage, and executable-model machinery |
| `src/cli/` | Model-specific command-line entry points |
| `src/artifact/` | Generated-model artifact entry points |
| `src/extensions/` | Core-backed domain abstractions exported as `zgc.nn` and `zgc.img` |
| `tests/` | Compile-time graph, runtime model, validation, view, and kernel coverage |
| `benchmarks/` | Operation and generated-model benchmark harness, with recorded results |
| `sandbox/` | Standalone model definitions, interactive inference, inspection, and artifact analysis |
| `docs/` | Architecture, design constraints, capabilities, and limitations |

## Documentation

- [Documentation index](docs/README.md)
- [Architecture and compilation pipeline](docs/architecture.md)
- [Design constraints](docs/design-constraints.md)
- [Model inspection](docs/inspection.md)
- [Generated model artifacts](docs/model-artifacts.md)
- [Development state](docs/development-state.md)
- [Sandbox and binary inspection](sandbox/README.md)
- [Benchmarks](benchmarks/README.md)
