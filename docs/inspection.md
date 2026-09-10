# Model inspection

Inspection is separate from graph construction, memory planning, kernel
dispatch, and model execution. The public `zgc.Inspect` module renders model
metadata to any `std.Io.Writer` without changing the generated model type.

## Programmatic API

Every generated model exposes the compile-time metadata consumed by the
inspector:

```zig
var buffer: [4096]u8 = undefined;
var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
defer stdout_writer.interface.flush() catch {};

try zgc.Inspect.writeModel(MyModel, &stdout_writer.interface, .{});
```

`writeModel` can include or omit capacity, graph, tree, and memory-plan
sections through `zgc.Inspect.Sections`. The individual renderers are also
public:

- `writeCapacity`
- `writeGraph`
- `writeGraphStructure`
- `writeMemoryPlan`
- `writeModelMemory`

`writeModelMemory` accepts a model instance and a runtime byte limit. The other
renderers inspect compile-time model metadata and do not initialize or execute
the model.

## Model-specific CLI

ZGC exports a `zgc_inspect_cli` module. A consumer supplies a module named
`model` containing `pub const Model`, producing an inspector specialized for
that generated model:

```zig
const zgc_dep = b.dependency("zgc", .{
    .target = target,
    .optimize = optimize,
});

const model_module = b.createModule(.{
    .root_source_file = b.path("src/model.zig"),
    .target = target,
    .optimize = optimize,
    // Add the imports required by the model definition here.
});

const cli_module = zgc_dep.module("zgc_inspect_cli");
cli_module.addImport("model", model_module);

const inspector = b.addExecutable(.{
    .name = "zgc-inspect",
    .root_module = cli_module,
});
b.installArtifact(inspector);
```

The executable supports these commands:

```text
zgc-inspect all
zgc-inspect summary
zgc-inspect graph
zgc-inspect tree
zgc-inspect memory-plan
zgc-inspect help
```

`all` is the default. Each executable is compile-time specialized for the model
module supplied by its build.
