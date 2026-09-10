const std = @import("std");
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // library
    const zgc_mod = b.addModule("zgc", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    _ = b.addModule("zgc_inspect_cli", .{
        .root_source_file = b.path("src/cli/inspect.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zgc", .module = zgc_mod }},
    });
    _ = b.addModule("zgc_model_runner", .{
        .root_source_file = b.path("src/artifact/model_runner.zig"),
        .target = target,
        .optimize = optimize,
    });

    // tests
    const test_embed_params = b.createModule(.{
        .root_source_file = b.path("tests/fixtures/embed_params.zig"),
        .target = target,
        .optimize = optimize,
    });
    const test_root_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const test_mod = b.addModule("zgc_tests", .{
        .root_source_file = b.path("tests/tests.zig"),
        .target = target,
        .optimize = optimize,
    });

    test_mod.addImport("zgc", test_root_mod);
    test_mod.addImport("embed_params", test_embed_params);

    const tests = b.addTest(.{
        .root_module = test_mod,
    });

    const runner_test_model = b.createModule(.{
        .root_source_file = b.path("tests/fixtures/runner_model.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zgc", .module = zgc_mod }},
    });
    const runner_test_module = b.createModule(.{
        .root_source_file = b.path("src/artifact/model_runner.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "model", .module = runner_test_model }},
    });
    const runner_test_exe = b.addExecutable(.{
        .name = "zgc-model-runner-test",
        .root_module = runner_test_module,
    });
    runner_test_exe.forceUndefinedSymbol(if (target.result.os.tag == .macos)
        "_zgc_run_model"
    else
        "zgc_run_model");

    const check_step = b.step("check", "Compile tests without running them");
    check_step.dependOn(&tests.step);
    check_step.dependOn(&runner_test_exe.step);

    const run_mod_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&runner_test_exe.step);

    // benchmarks
    const benchmark_op = b.option([]const u8, "op", "Operation, model, or benchmark tier to run") orelse "all";
    const benchmark_model = b.option([]const u8, "model", "Dense model size used by -Dop=model") orelse "small";
    const benchmark_batch = b.option(usize, "batch", "Compile-time batch size used by -Dop=model") orelse 1;
    const benchmark_iterations = b.option(usize, "iterations", "Benchmark invocations per timed sample (0 calibrates to sample_ms)") orelse 0;
    const benchmark_runs = b.option(usize, "runs", "Number of timed samples") orelse 30;
    const benchmark_warmup = b.option(usize, "warmup_iterations", "Untimed warmup invocations (0 warms for warmup_ms)") orelse 0;
    const benchmark_sample_ms = b.option(usize, "sample_ms", "Minimum duration of each automatically calibrated timed sample") orelse 250;
    const benchmark_warmup_ms = b.option(usize, "warmup_ms", "Duration of the automatic untimed warmup") orelse 2_000;

    const benchmark_options = b.addOptions();
    benchmark_options.addOption([]const u8, "op", benchmark_op);
    benchmark_options.addOption([]const u8, "model", benchmark_model);
    benchmark_options.addOption(usize, "batch", benchmark_batch);
    benchmark_options.addOption(usize, "iterations", benchmark_iterations);
    benchmark_options.addOption(usize, "runs", benchmark_runs);
    benchmark_options.addOption(usize, "warmup_iterations", benchmark_warmup);
    benchmark_options.addOption(usize, "sample_ms", benchmark_sample_ms);
    benchmark_options.addOption(usize, "warmup_ms", benchmark_warmup_ms);

    const benchmark_exe = b.addExecutable(.{
        .name = "zgc-benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zgc", .module = zgc_mod },
                .{ .name = "build_options", .module = benchmark_options.createModule() },
            },
        }),
    });
    const run_benchmark = b.addRunArtifact(benchmark_exe);
    const benchmark_step = b.step("benchmark", "Run the benchmark selected by -Dop");
    benchmark_step.dependOn(&run_benchmark.step);
}
