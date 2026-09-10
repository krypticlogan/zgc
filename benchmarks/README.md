# Benchmarks

This directory contains ZGC performance measurements at two levels:

- `ops/` measures individual tensor operations and layouts.
- `models/` measures complete model execution through the generated ZGC
  pipeline.

The model suite covers small, medium, and large dense networks at batch sizes 1
and 32. Recorded measurements and their environment are documented under
[`reference/`](reference/README.md).

## Run

From the repository root:

```sh
zig build benchmark -Dop=relu -Doptimize=ReleaseFast
```

Without `-Dop`, the build runs both tiers (`all`). Run either tier separately:

```sh
zig build benchmark -Dop=ops -Doptimize=ReleaseFast
zig build benchmark -Dop=models -Doptimize=ReleaseFast
```

Run one model case with:

```sh
zig build benchmark -Dop=model -Dmodel=small -Dbatch=2 -Doptimize=ReleaseFast
```

Run every model shape at batch sizes 1 and 32 with:

```sh
./benchmarks/run-models.sh
```

Additional build options are forwarded to each case:

```sh
./benchmarks/run-models.sh -Dwarmup_ms=500 -Dsample_ms=100 -Druns=10
```

Available selectors are:

| Tier/group | `-Dop` values |
| --- | --- |
| Suites | `all`, `ops`, `models` |
| Elementwise | `relu`, `relu-16`, `relu-64`, `relu-256`, `add`, `add-strided`, `add-broadcast`, `exp` |
| Reduction | `sum`, `sum-strided`, `softmax`, `softmax-strided`, `softmax-8`, `softmax-10` |
| Matmul shapes | `matmul-32`, `matmul-64`, `matmul-128`, `matmul-rect`, `matmul-reference-16x8x24`, `matmul-reference-16x32x64` |
| Matmul layouts | `matmul-lhs-strided`, `matmul-rhs-strided`, `matmul-output-strided`, `matmul-batch` |
| Dense model | `model` with `-Dmodel=small\|medium\|large` and any positive `-Dbatch` |
| Dense models, batch 1 | `model-small-b1`, `model-medium-b1`, `model-large-b1` |
| Dense models, batch 32 | `model-small-b32`, `model-medium-b32`, `model-large-b32` |

The model cases are:

| Name | Layer widths | Activations | Parameters | Batches |
| --- | --- | --- | ---: | --- |
| small | 32 → 16 → 8 | ReLU, softmax | 664 | 1, 32 |
| medium | 128 → 64 → 10 | ReLU, softmax | 8,906 | 1, 32 |
| large | 1024 → 256 → 128 → 64 → 2 | ReLU, ReLU, ReLU, softmax | 303,682 | 1, 32 |

The `matmul-reference-*` selectors retain two useful reference shapes in the
active ZGC matmul suite.

By default, each case receives a two-second untimed warmup followed by 30 timed
samples. After warmup, the harness calibrates the case's iteration count against
a 250 ms sample target, providing approximately 7.5 seconds or more of timed
execution per case.
Calibration may select a longer sample when a case's seed iteration count
already exceeds the target.

Use `-Dsample_ms` and `-Dwarmup_ms` to change the duration targets, and `-Druns`
to change the sample count. Explicit `-Diterations` or `-Dwarmup_iterations`
values bypass calibration or duration-based warmup respectively. These
overrides are useful for smoke tests but should be reported whenever their
results are retained. Always report the Zig version, target, optimization mode,
CPU/OS, case shape, and benchmark configuration alongside results.

The timer surrounds a calibrated group of invocations, so its two reads are
amortized. Reported statistics include minimum, median, average, p95, maximum,
standard deviation, and coefficient of variation across independently timed
samples. Each operation case owns its buffers, warms the kernel first, and
prevents the result from being optimized away. Model parameter initialization,
source copying, and input binding happen before calibration, warmup, and
timing. A model invocation is one forward pass over its configured batch. The
`normalized` line reports nanoseconds per inference, and the work throughput
reports inferences per second.
Model inputs, weights, and biases are initialized deterministically before
timing.

## Coverage

| Layer | Cases | Status |
| --- | --- | --- |
| Primitive | ReLU, add, exp | Measured, including strided and broadcast layouts |
| Primitive | matmul | Measured across four shapes, four 64³ layouts, and a batch-contiguous layout |
| Primitive | sum and softmax | Measured across contiguous and strided axes |
| Primitive | sub and remaining layout/elementwise ops | Not covered |
| Fused/operator | dense + activation | Covered by the model tier |
| End to end | small, medium, and large dense networks at batch 1 and 32 | Measured through generated ZGC models |
| End to end | transformer block; sequence sweeps | Not covered |
| Resources | workspace, allocations, binary size | Not covered |

Cases are selected in `main.zig`. Each exposes its name, default iteration
counts, work unit, work and byte counts per invocation, `init`, and `run`.
Model cases may also expose a preparation hook and normalization metadata.
