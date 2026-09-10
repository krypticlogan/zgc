# Benchmark data

The CSV files contain `ReleaseFast` operation and model measurements recorded
on 2026-09-01 with:

- Zig 0.16.0
- `ReleaseFast`
- x86_64 Darwin 24.6.0
- Intel Core i5-8279U at 2.40 GHz
- 2 seconds of untimed warmup per case
- 30 timed samples per case, automatically calibrated to a 250 ms target
- median, average, p95, minimum, maximum, and variability reporting

Results should only be compared directly when the Zig version, target,
optimization mode, processor, operating system, shapes, and benchmark
configuration match.
