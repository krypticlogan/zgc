const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const run_count = build_options.runs;

comptime {
    if (run_count == 0) @compileError("benchmark runs must be greater than zero");
    if (build_options.iterations == 0 and build_options.sample_ms == 0) {
        @compileError("sample_ms must be greater than zero when iterations are calibrated");
    }
    if (build_options.warmup_iterations == 0 and build_options.warmup_ms == 0) {
        @compileError("warmup_ms must be greater than zero when warmup iterations are automatic");
    }
}

pub fn main(init: std.process.Init) void {
    if (comptime std.mem.eql(u8, build_options.op, "all")) {
        runSuite(all_op_benchmarks, init);
        runSuite(model_benchmarks, init);
        return;
    }
    if (comptime std.mem.eql(u8, build_options.op, "ops")) {
        runSuite(all_op_benchmarks, init);
        return;
    }
    if (comptime std.mem.eql(u8, build_options.op, "models")) {
        runSuite(model_benchmarks, init);
        return;
    }
    if (comptime std.mem.eql(u8, build_options.op, "model")) {
        runBenchmark(
            @import("models/dense.zig").select(
                build_options.model,
                build_options.batch,
            ),
            init,
        );
        return;
    }
    runBenchmark(selectedBenchmark(), init);
}

fn runSuite(comptime benchmarks: anytype, init: std.process.Init) void {
    inline for (benchmarks) |Benchmark| runBenchmark(Benchmark, init);
}

fn selectedBenchmark() type {
    const matmul = @import("ops/matmul.zig");
    const elementwise = @import("ops/elementwise.zig");
    const reduction = @import("ops/reduction.zig");
    const models = @import("models/dense.zig");

    if (std.mem.eql(u8, build_options.op, "relu")) return @import("ops/relu.zig").Benchmark;
    if (std.mem.eql(u8, build_options.op, "relu-16")) return @import("ops/relu.zig").Small16;
    if (std.mem.eql(u8, build_options.op, "relu-64")) return @import("ops/relu.zig").Medium64;
    if (std.mem.eql(u8, build_options.op, "relu-256")) return @import("ops/relu.zig").Large256;
    if (std.mem.eql(u8, build_options.op, "add")) return elementwise.AddContiguous;
    if (std.mem.eql(u8, build_options.op, "add-strided")) return elementwise.AddStrided;
    if (std.mem.eql(u8, build_options.op, "add-broadcast")) return elementwise.AddBroadcast;
    if (std.mem.eql(u8, build_options.op, "exp")) return elementwise.ExpContiguous;
    if (std.mem.eql(u8, build_options.op, "sum")) return reduction.SumContiguous;
    if (std.mem.eql(u8, build_options.op, "sum-strided")) return reduction.SumStrided;
    if (std.mem.eql(u8, build_options.op, "softmax")) return reduction.SoftmaxContiguous;
    if (std.mem.eql(u8, build_options.op, "softmax-strided")) return reduction.SoftmaxStrided;
    if (std.mem.eql(u8, build_options.op, "softmax-8")) return reduction.Softmax8;
    if (std.mem.eql(u8, build_options.op, "softmax-10")) return reduction.Softmax10;
    if (std.mem.eql(u8, build_options.op, "matmul-32")) return matmul.Square32;
    if (std.mem.eql(u8, build_options.op, "matmul-64")) return matmul.Square64;
    if (std.mem.eql(u8, build_options.op, "matmul-128")) return matmul.Square128;
    if (std.mem.eql(u8, build_options.op, "matmul-rect")) return matmul.Rectangular;
    if (std.mem.eql(u8, build_options.op, "matmul-lhs-strided")) return matmul.LhsStrided;
    if (std.mem.eql(u8, build_options.op, "matmul-rhs-strided")) return matmul.RhsStrided;
    if (std.mem.eql(u8, build_options.op, "matmul-output-strided")) return matmul.OutputStrided;
    if (std.mem.eql(u8, build_options.op, "matmul-batch")) return matmul.BatchContiguous;
    if (std.mem.eql(u8, build_options.op, "matmul-reference-16x8x24")) return matmul.Reference16x8x24;
    if (std.mem.eql(u8, build_options.op, "matmul-reference-16x32x64")) return matmul.Reference16x32x64;
    if (std.mem.eql(u8, build_options.op, "model-small-b1")) return models.SmallBatch1;
    if (std.mem.eql(u8, build_options.op, "model-medium-b1")) return models.MediumBatch1;
    if (std.mem.eql(u8, build_options.op, "model-large-b1")) return models.LargeBatch1;
    if (std.mem.eql(u8, build_options.op, "model-small-b32")) return models.SmallBatch32;
    if (std.mem.eql(u8, build_options.op, "model-medium-b32")) return models.MediumBatch32;
    if (std.mem.eql(u8, build_options.op, "model-large-b32")) return models.LargeBatch32;
    @compileError("unknown benchmark selector: " ++ build_options.op);
}

const all_op_benchmarks = .{
    @import("ops/relu.zig").Benchmark,
    @import("ops/relu.zig").Small16,
    @import("ops/relu.zig").Medium64,
    @import("ops/relu.zig").Large256,
    @import("ops/elementwise.zig").AddContiguous,
    @import("ops/elementwise.zig").AddStrided,
    @import("ops/elementwise.zig").AddBroadcast,
    @import("ops/elementwise.zig").ExpContiguous,
    @import("ops/reduction.zig").SumContiguous,
    @import("ops/reduction.zig").SumStrided,
    @import("ops/reduction.zig").SoftmaxContiguous,
    @import("ops/reduction.zig").SoftmaxStrided,
    @import("ops/reduction.zig").Softmax8,
    @import("ops/reduction.zig").Softmax10,
    @import("ops/matmul.zig").Square32,
    @import("ops/matmul.zig").Square64,
    @import("ops/matmul.zig").Square128,
    @import("ops/matmul.zig").Rectangular,
    @import("ops/matmul.zig").LhsStrided,
    @import("ops/matmul.zig").RhsStrided,
    @import("ops/matmul.zig").OutputStrided,
    @import("ops/matmul.zig").BatchContiguous,
    @import("ops/matmul.zig").Reference16x8x24,
    @import("ops/matmul.zig").Reference16x32x64,
};

const model_benchmarks = .{
    @import("models/dense.zig").SmallBatch1,
    @import("models/dense.zig").MediumBatch1,
    @import("models/dense.zig").LargeBatch1,
    @import("models/dense.zig").SmallBatch32,
    @import("models/dense.zig").MediumBatch32,
    @import("models/dense.zig").LargeBatch32,
};

fn runBenchmark(comptime Benchmark: type, init: std.process.Init) void {
    const clock = std.Io.Clock.awake;
    var selected = Benchmark.init();
    if (comptime @hasDecl(Benchmark, "prepare")) selected.prepare();

    const warmup = warmUp(Benchmark, &selected, clock, init);
    const iterations = calibrateIterations(Benchmark, &selected, clock, init);

    var samples_ns: [run_count]f64 = undefined;
    var timed_elapsed_ns: f64 = 0;
    for (&samples_ns) |*sample| {
        const start = clock.now(init.io).nanoseconds;
        selected.run(iterations);
        const end = clock.now(init.io).nanoseconds;
        const elapsed_ns: f64 = @floatFromInt(end - start);
        timed_elapsed_ns += elapsed_ns;
        sample.* = elapsed_ns / @as(f64, @floatFromInt(iterations));
    }

    var minimum_ns = std.math.inf(f64);
    var maximum_ns: f64 = 0;
    var total_ns: f64 = 0;
    for (samples_ns) |sample| {
        minimum_ns = @min(minimum_ns, sample);
        maximum_ns = @max(maximum_ns, sample);
        total_ns += sample;
    }
    const average_ns = total_ns / @as(f64, @floatFromInt(run_count));

    var squared_deviation: f64 = 0;
    for (samples_ns) |sample| {
        const difference = sample - average_ns;
        squared_deviation += difference * difference;
    }
    const standard_deviation_ns = @sqrt(
        squared_deviation / @as(f64, @floatFromInt(run_count)),
    );
    var sorted_samples = samples_ns;
    std.mem.sort(f64, &sorted_samples, {}, std.sort.asc(f64));
    const median_ns = if (run_count % 2 == 0)
        (sorted_samples[run_count / 2 - 1] + sorted_samples[run_count / 2]) / 2
    else
        sorted_samples[run_count / 2];
    const percentile_95_index = @min((95 * run_count + 99) / 100 - 1, run_count - 1);
    const percentile_95_ns = sorted_samples[percentile_95_index];

    const invocations_per_second = @as(f64, std.time.ns_per_s) / average_ns;
    const work_items_per_second = invocations_per_second * Benchmark.work_items_per_invocation;
    const bytes_per_second = invocations_per_second * Benchmark.bytes_per_invocation;
    const coefficient_of_variation = standard_deviation_ns / average_ns * 100;

    std.debug.print(
        \\Benchmark: {s} ({s})
        \\  configuration  {d} warmup iterations / {d:.3} s | {d} samples x {d} iterations
        \\  sample timing  {d} ms target | {d:.3} s total timed
        \\  latency (ns)   min {d:>10.3} | avg {d:>10.3} | max {d:>10.3}
        \\  distribution   median {d:.3} ns | p95 {d:.3} ns
        \\  variability    stddev {d:.3} ns | CV {d:.2}%
        \\  throughput     {d:.3} invocations/s | {d:.3} {s}/s | {d:.3} GiB/s
        \\
    , .{
        Benchmark.name,
        @tagName(builtin.mode),
        warmup.iterations,
        @as(f64, @floatFromInt(warmup.elapsed_ns)) / std.time.ns_per_s,
        run_count,
        iterations,
        if (build_options.iterations == 0) build_options.sample_ms else 0,
        timed_elapsed_ns / std.time.ns_per_s,
        minimum_ns,
        average_ns,
        maximum_ns,
        median_ns,
        percentile_95_ns,
        standard_deviation_ns,
        coefficient_of_variation,
        invocations_per_second,
        work_items_per_second,
        Benchmark.work_unit,
        bytes_per_second / (1024 * 1024 * 1024),
    });

    if (comptime @hasDecl(Benchmark, "latency_divisor")) {
        const divisor = Benchmark.latency_divisor;
        std.debug.print(
            "  normalized     min {d:>10.3} | avg {d:>10.3} | max {d:>10.3} ns/{s}\n",
            .{
                minimum_ns / divisor,
                average_ns / divisor,
                maximum_ns / divisor,
                Benchmark.latency_unit,
            },
        );
    }
    if (comptime @hasDecl(Benchmark, "parameter_count")) {
        std.debug.print(
            "  model          {d} parameters | batch {d}\n",
            .{ Benchmark.parameter_count, Benchmark.batch },
        );
    }
    std.debug.print("\n", .{});
}

fn calibrateIterations(
    comptime Benchmark: type,
    selected: *Benchmark,
    clock: std.Io.Clock,
    init: std.process.Init,
) usize {
    if (build_options.iterations != 0) return build_options.iterations;

    const target_ns: i96 = @intCast(build_options.sample_ms * std.time.ns_per_ms);
    var iterations: usize = Benchmark.default_iterations;
    for (0..4) |_| {
        const start = clock.now(init.io).nanoseconds;
        selected.run(iterations);
        const elapsed_ns = @max(clock.now(init.io).nanoseconds - start, 1);
        if (elapsed_ns >= target_ns) return iterations;

        const scale: usize = @intCast(@divTrunc(target_ns + elapsed_ns - 1, elapsed_ns));
        iterations = std.math.mul(usize, iterations, scale) catch
            @panic("benchmark calibration exceeded the iteration range");
    }
    return iterations;
}

const WarmupResult = struct {
    iterations: usize,
    elapsed_ns: i96,
};

fn warmUp(
    comptime Benchmark: type,
    selected: *Benchmark,
    clock: std.Io.Clock,
    init: std.process.Init,
) WarmupResult {
    const start = clock.now(init.io).nanoseconds;
    if (build_options.warmup_iterations != 0) {
        selected.run(build_options.warmup_iterations);
        return .{
            .iterations = build_options.warmup_iterations,
            .elapsed_ns = clock.now(init.io).nanoseconds - start,
        };
    }

    const target_ns: i96 = @intCast(build_options.warmup_ms * std.time.ns_per_ms);
    const chunk_iterations = Benchmark.default_warmup_iterations;
    var completed_iterations: usize = 0;
    var elapsed_ns: i96 = 0;
    while (elapsed_ns < target_ns) {
        selected.run(chunk_iterations);
        completed_iterations = std.math.add(usize, completed_iterations, chunk_iterations) catch
            @panic("benchmark warmup exceeded the iteration range");
        elapsed_ns = clock.now(init.io).nanoseconds - start;
    }
    return .{ .iterations = completed_iterations, .elapsed_ns = elapsed_ns };
}
