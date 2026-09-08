const std = @import("std");
const testing = std.testing;

const embedding_codegen = @import("embedding_codegen");

test "embedding generator copies parameters and emits the expected module" {
    const io = testing.io;
    const allocator = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.createDir(io, "input", .default_dir);
    var input_dir = try tmp.dir.openDir(io, "input", .{});
    defer input_dir.close(io);

    const weights = [_]f32{ 2, -1, 0.5, 3 };
    const biases = [_]f32{ 1, -2 };
    try input_dir.writeFile(io, .{
        .sub_path = "w1.bin",
        .data = std.mem.asBytes(&weights),
    });
    try input_dir.writeFile(io, .{
        .sub_path = "b1.bin",
        .data = std.mem.asBytes(&biases),
    });

    const root_path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}",
        .{tmp.sub_path},
    );
    defer allocator.free(root_path);
    const input_path = try std.fs.path.join(allocator, &.{ root_path, "input" });
    defer allocator.free(input_path);
    const output_path = try std.fs.path.join(allocator, &.{ root_path, "output" });
    defer allocator.free(output_path);

    try embedding_codegen.writeEmbeddedParamsBundle(
        io,
        allocator,
        "embeds.zig",
        input_path,
        output_path,
        1,
    );

    var output_dir = try tmp.dir.openDir(io, "output", .{});
    defer output_dir.close(io);

    const generated = try output_dir.readFileAlloc(
        io,
        "embeds.zig",
        allocator,
        .limited(1024),
    );
    defer allocator.free(generated);
    try testing.expectEqualStrings(
        \\pub const weights = [_][]const u8{
        \\    @embedFile("w1.bin"),
        \\};
        \\
        \\pub const biases = [_][]const u8{
        \\    @embedFile("b1.bin"),
        \\};
        \\
    , generated);

    const copied_weights = try output_dir.readFileAlloc(
        io,
        "w1.bin",
        allocator,
        .limited(1024),
    );
    defer allocator.free(copied_weights);
    try testing.expectEqualSlices(u8, std.mem.asBytes(&weights), copied_weights);

    const copied_biases = try output_dir.readFileAlloc(
        io,
        "b1.bin",
        allocator,
        .limited(1024),
    );
    defer allocator.free(copied_biases);
    try testing.expectEqualSlices(u8, std.mem.asBytes(&biases), copied_biases);
}
