const std = @import("std");
const Dtype = @import("../dtype.zig").Dtype;

pub fn AccumulatorScalar(comptime dtype: Dtype) type {
    return switch (dtype) { // dtypes with smaller bit withs accumulate into larger windows
        .f16 => f32,
        .f32 => f32,
        .i8 => i32,
    };
}

pub fn AccumulatorVector(comptime dtype: Dtype, comptime len: usize) type {
    return @Vector(len, AccumulatorScalar(dtype));
}

pub fn widenVector(
    comptime dtype: Dtype,
    comptime len: usize,
    values: dtype.Vector(len),
) AccumulatorVector(dtype, len) {
    const AccVector = AccumulatorVector(dtype, len);

    return switch (comptime dtype.kind()) {
        .float => @as(AccVector, @floatCast(values)),
        .signed_integer => @as(AccVector, @intCast(values)),
    };
}

pub fn widenScalar(
    comptime dtype: Dtype,
    value: dtype.Scalar(),
) AccumulatorScalar(dtype) {
    const AccT = AccumulatorScalar(dtype);

    return switch (comptime dtype.kind()) {
        .float => @as(AccT, @floatCast(value)),
        .signed_integer => @as(AccT, @intCast(value)),
    };
}

pub fn narrowVector(
    comptime dtype: Dtype,
    comptime len: usize,
    values: AccumulatorVector(dtype, len),
) dtype.Vector(len) {
    return switch (comptime dtype.kind()) {
        .float => @floatCast(values),
        .signed_integer => @intCast(values),
    };
}

pub fn narrowScalar(
    comptime dtype: Dtype,
    value: AccumulatorScalar(dtype),
) dtype.Scalar() {
    return switch (comptime dtype.kind()) {
        .float => @floatCast(value),
        .signed_integer => @intCast(value),
    };
}
