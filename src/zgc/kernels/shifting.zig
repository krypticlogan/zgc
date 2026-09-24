const Op = @import("../operations/semantic.zig").Op;

/// Materialize a translated tensor. Geometry and boundary choice are
/// validated before this kernel is instantiated.
pub fn shift(
    inputs: anytype,
    output: anytype,
    comptime attrs: Op.Compute.ShiftAttrs,
) void {
    const input = inputs[0];
    const rank = @TypeOf(input).rank;
    for (0..output.len()) |linear_index| {
        var remaining = linear_index;
        var source_indices: [rank]usize = undefined;
        var output_indices: [rank]usize = undefined;
        var outside = false;

        comptime var axis = rank;
        inline while (axis > 0) {
            axis -= 1;
            const coordinate = remaining % output.shape[axis];
            remaining /= output.shape[axis];
            output_indices[axis] = coordinate;
            if (mapCoordinate(coordinate, input.shape[axis], attrs.offsets[axis], attrs.boundary)) |source| {
                source_indices[axis] = source;
            } else {
                outside = true;
            }
        }

        const value = if (comptime attrs.boundary == .constant)
            if (outside) inputs[1].get(.{}) else input.get(source_indices)
        else
            input.get(source_indices);
        output.set(output_indices, value);
    }
}

/// Positive offset means output[i + offset] receives input[i].
fn mapCoordinate(
    coordinate: usize,
    extent: usize,
    comptime offset: isize,
    comptime boundary: Op.Compute.ShiftAttrs.Boundary,
) ?usize {
    return switch (boundary) {
        .wrap => blk: {
            const displacement = positiveRemainder(offset, extent);
            break :blk if (coordinate >= displacement)
                coordinate - displacement
            else
                extent - (displacement - coordinate);
        },
        .edge => blk: {
            if (offset >= 0) {
                const displacement: usize = @intCast(offset);
                break :blk if (displacement >= extent or coordinate < displacement)
                    0
                else
                    coordinate - displacement;
            }
            const displacement: usize = @intCast(@abs(offset));
            break :blk if (displacement >= extent or coordinate >= extent - displacement)
                extent - 1
            else
                coordinate + displacement;
        },
        .reflect => blk: {
            // Reflect excludes the edge element on each return trip.
            if (extent == 1) break :blk 0;
            const period = @as(u128, extent - 1) * 2;
            const displacement = positiveRemainderWide(offset, period);
            const phase = if (@as(u128, coordinate) >= displacement)
                @as(u128, coordinate) - displacement
            else
                period - (displacement - coordinate);
            break :blk @intCast(if (phase < extent) phase else period - phase);
        },
        .constant => blk: {
            if (offset >= 0) {
                const displacement: usize = @intCast(offset);
                if (displacement > coordinate) break :blk null;
                break :blk coordinate - displacement;
            }
            const displacement: usize = @intCast(@abs(offset));
            if (displacement >= extent or coordinate >= extent - displacement) break :blk null;
            break :blk coordinate + displacement;
        },
    };
}

fn positiveRemainder(comptime offset: isize, extent: usize) usize {
    return @intCast(positiveRemainderWide(offset, extent));
}

fn positiveRemainderWide(comptime offset: isize, period: u128) u128 {
    return @intCast(@mod(@as(i128, offset), @as(i128, @intCast(period))));
}
