const std = @import("std");
const Dtype = @import("../dtype.zig").Dtype;

/// Apply an element-wise operator through a contiguous SIMD fast path or a generic strided
/// traversal when any participating view is non-contiguous.
fn unary(input: anytype, output: anytype, comptime Operator: type) void {
    const Input = @TypeOf(input);
    const Output = @TypeOf(output);
    const dtype = Input.dtype;
    if (comptime hasStaticGeometry(Input) and hasStaticGeometry(Output)) {
        if (comptime std.mem.eql(isize, &Input.static_strides, &Output.static_strides) and
            Input.static_is_dense_positive and Output.static_is_dense_positive)
        {
            applyUnaryDense(input.denseSlice().?, output.denseSlice().?, dtype, Operator);
            return;
        }
    } else {
        if (std.mem.eql(isize, &input.strides, &output.strides)) {
            if (input.denseSlice()) |input_storage| {
                if (output.denseSlice()) |output_storage| {
                    applyUnaryDense(input_storage, output_storage, dtype, Operator);
                    return;
                }
            }
        }
    }

    for (0..input.len()) |linear_index| {
        const input_index = input.elementOffsetFromLinear(linear_index);
        const output_index = output.elementOffsetFromLinear(linear_index);
        output.storage[output_index] = Operator.scalar(
            dtype,
            input.storage[input_index],
        );
    }
}

fn applyUnaryDense(
    input_storage: anytype,
    output_storage: anytype,
    comptime dtype: Dtype,
    comptime Operator: type,
) void {
    const vector_len = std.simd.suggestVectorLength(dtype.Scalar()) orelse 1;
    const Vector = dtype.Vector(vector_len);

    var index: usize = 0;
    while (index + vector_len <= input_storage.len) : (index += vector_len) {
        const values: Vector = input_storage[index..][0..vector_len].*;
        output_storage[index..][0..vector_len].* = Operator.vector(dtype, vector_len, values);
    }
    while (index < input_storage.len) : (index += 1) {
        output_storage[index] = Operator.scalar(dtype, input_storage[index]);
    }
}

/// Apply a element-wise binary operator through a contiguous SIMD fast path or a generic
/// strided traversal.
fn binary(a: anytype, b: anytype, output: anytype, comptime Operator: type) void {
    const Output = @TypeOf(output);

    const output_shape = if (comptime hasStaticGeometry(Output))
        Output.static_shape
    else
        output.shape;
    const a_view = a.broadcastTo(Output.rank, output_shape);
    const b_view = b.broadcastTo(Output.rank, output_shape);
    const AView = @TypeOf(a_view);
    const BView = @TypeOf(b_view);

    const dtype = Output.dtype;
    if (comptime hasStaticGeometry(AView) and
        hasStaticGeometry(BView) and
        hasStaticGeometry(Output))
    {
        if (comptime std.mem.eql(isize, &AView.static_strides, &BView.static_strides) and
            std.mem.eql(isize, &AView.static_strides, &Output.static_strides) and
            AView.static_is_dense_positive and
            BView.static_is_dense_positive and
            Output.static_is_dense_positive)
        {
            applyBinaryDense(
                a_view.denseSlice().?,
                b_view.denseSlice().?,
                output.denseSlice().?,
                dtype,
                Operator,
            );
            return;
        }
    } else {
        if (std.mem.eql(isize, &a_view.strides, &b_view.strides) and
            std.mem.eql(isize, &a_view.strides, &output.strides))
        {
            if (a_view.denseSlice()) |a_storage| {
                if (b_view.denseSlice()) |b_storage| {
                    if (output.denseSlice()) |output_storage| {
                        applyBinaryDense(a_storage, b_storage, output_storage, dtype, Operator);
                        return;
                    }
                }
            }
        }
    }

    if (comptime Output.rank == 2) {
        if (hasSameLayout(a_view, output) and isTrailingVectorBroadcast(b_view) and output.denseSlice() != null) {
            applyFirstAxisBroadcast(a_view, b_view, output, dtype, Operator, true);
            return;
        }
        if (isTrailingVectorBroadcast(a_view) and hasSameLayout(b_view, output) and output.denseSlice() != null) {
            applyFirstAxisBroadcast(b_view, a_view, output, dtype, Operator, false);
            return;
        }
    }

    if (a_view.contiguousSlice()) |a_storage| {
        if (b_view.contiguousSlice()) |b_storage| {
            if (output.contiguousSlice()) |output_storage| {
                const vector_len = std.simd.suggestVectorLength(Output.scalar_type) orelse 1;
                const Vector = dtype.Vector(vector_len);

                var index: usize = 0;
                while (index + vector_len <= a_storage.len) : (index += vector_len) {
                    const a_values: Vector = a_storage[index..][0..vector_len].*;
                    const b_values: Vector = b_storage[index..][0..vector_len].*;
                    output_storage[index..][0..vector_len].* = Operator.vector(dtype, vector_len, a_values, b_values);
                }
                while (index < a_storage.len) : (index += 1) {
                    output_storage[index] = Operator.scalar(dtype, a_storage[index], b_storage[index]);
                }
                return;
            }
        }
    }

    for (0..output.len()) |linear_index| {
        const a_index = a_view.elementOffsetFromLinear(linear_index);
        const b_index = b_view.elementOffsetFromLinear(linear_index);
        const output_index = output.elementOffsetFromLinear(linear_index);
        output.storage[output_index] = Operator.scalar(
            dtype,
            a_view.storage[a_index],
            b_view.storage[b_index],
        );
    }
}

fn hasSameLayout(input: anytype, output: anytype) bool {
    const Input = @TypeOf(input);
    const Output = @TypeOf(output);
    if (comptime hasStaticGeometry(Input) and hasStaticGeometry(Output)) {
        return comptime std.mem.eql(isize, &Input.static_strides, &Output.static_strides) and
            Input.static_is_dense_positive;
    }
    return std.mem.eql(isize, &input.strides, &output.strides) and
        input.denseSlice() != null;
}

fn isTrailingVectorBroadcast(view: anytype) bool {
    const View = @TypeOf(view);
    if (comptime hasStaticGeometry(View)) {
        return comptime View.static_strides[0] == 0 and View.static_strides[1] == 1;
    }
    return view.strides[0] == 0 and view.strides[1] == 1;
}

fn hasStaticGeometry(comptime View: type) bool {
    return @hasDecl(View, "geometry_is_static") and View.geometry_is_static;
}

/// Apply a trailing vector across a dense [batch, width] tensor whose first
/// axis is contiguous. SIMD lanes traverse independent batch rows while the
/// broadcast value is splatted once per output column.
fn applyFirstAxisBroadcast(
    dense: anytype,
    broadcast: anytype,
    output: anytype,
    comptime dtype: Dtype,
    comptime Operator: type,
    comptime dense_is_a: bool,
) void {
    const vector_len = std.simd.suggestVectorLength(dtype.Scalar()) orelse 1;
    const Vector = dtype.Vector(vector_len);
    const batch = output.shape[0];
    const width = output.shape[1];

    for (0..width) |column| {
        const broadcast_vector: Vector = @splat(broadcast.get(.{ 0, column }));
        var row: usize = 0;
        while (row + vector_len <= batch) : (row += vector_len) {
            const dense_offset = dense.elementOffset(.{ row, column });
            const output_offset = output.elementOffset(.{ row, column });
            const dense_values: Vector = dense.storage[dense_offset..][0..vector_len].*;
            output.storage[output_offset..][0..vector_len].* = if (dense_is_a)
                Operator.vector(dtype, vector_len, dense_values, broadcast_vector)
            else
                Operator.vector(dtype, vector_len, broadcast_vector, dense_values);
        }
        while (row < batch) : (row += 1) {
            const dense_value = dense.get(.{ row, column });
            const broadcast_value = broadcast.get(.{ 0, column });
            output.set(
                .{ row, column },
                if (dense_is_a)
                    Operator.scalar(dtype, dense_value, broadcast_value)
                else
                    Operator.scalar(dtype, broadcast_value, dense_value),
            );
        }
    }
}

fn applyBinaryDense(
    a_storage: anytype,
    b_storage: anytype,
    output_storage: anytype,
    comptime dtype: Dtype,
    comptime Operator: type,
) void {
    const vector_len = std.simd.suggestVectorLength(dtype.Scalar()) orelse 1;
    const Vector = dtype.Vector(vector_len);

    var index: usize = 0;
    while (index + vector_len <= a_storage.len) : (index += vector_len) {
        const a_values: Vector = a_storage[index..][0..vector_len].*;
        const b_values: Vector = b_storage[index..][0..vector_len].*;
        output_storage[index..][0..vector_len].* = Operator.vector(
            dtype,
            vector_len,
            a_values,
            b_values,
        );
    }
    while (index < a_storage.len) : (index += 1) {
        output_storage[index] = Operator.scalar(dtype, a_storage[index], b_storage[index]);
    }
}

pub fn relu(input: anytype, output: anytype) void {
    unary(input, output, struct {
        fn scalar(comptime dtype: Dtype, value: dtype.Scalar()) dtype.Scalar() {
            return @max(value, dtype.zero());
        }

        fn vector(
            comptime dtype: Dtype,
            comptime len: usize,
            values: dtype.Vector(len),
        ) dtype.Vector(len) {
            return @max(values, dtype.vectorZero(len));
        }
    });
}

pub fn exp(input: anytype, output: anytype) void {
    unary(input, output, struct {
        fn scalar(comptime dtype: Dtype, value: dtype.Scalar()) dtype.Scalar() {
            return @exp(value);
        }

        fn vector(
            comptime dtype: Dtype,
            comptime len: usize,
            values: dtype.Vector(len),
        ) dtype.Vector(len) {
            return @exp(values);
        }
    });
}

pub fn add(a: anytype, b: anytype, output: anytype) void {
    binary(a, b, output, struct {
        fn scalar(comptime dtype: Dtype, a_value: dtype.Scalar(), b_value: dtype.Scalar()) dtype.Scalar() {
            return a_value + b_value;
        }

        fn vector(comptime dtype: Dtype, comptime len: usize, a_vec: dtype.Vector(len), b_vec: dtype.Vector(len)) dtype.Vector(len) {
            return a_vec + b_vec;
        }
    });
}

pub fn sub(a: anytype, b: anytype, output: anytype) void {
    binary(a, b, output, struct {
        fn scalar(comptime dtype: Dtype, a_value: dtype.Scalar(), b_value: dtype.Scalar()) dtype.Scalar() {
            return a_value - b_value;
        }

        fn vector(comptime dtype: Dtype, comptime len: usize, a_vec: dtype.Vector(len), b_vec: dtype.Vector(len)) dtype.Vector(len) {
            return a_vec - b_vec;
        }
    });
}

pub fn mul(a: anytype, b: anytype, output: anytype) void {
    binary(a, b, output, struct {
        fn scalar(comptime dtype: Dtype, a_value: dtype.Scalar(), b_value: dtype.Scalar()) dtype.Scalar() {
            return a_value * b_value;
        }

        fn vector(comptime dtype: Dtype, comptime len: usize, a_vec: dtype.Vector(len), b_vec: dtype.Vector(len)) dtype.Vector(len) {
            return a_vec * b_vec;
        }
    });
}

pub fn div(a: anytype, b: anytype, output: anytype) void {
    binary(a, b, output, struct {
        fn scalar(comptime dtype: Dtype, a_value: dtype.Scalar(), b_value: dtype.Scalar()) dtype.Scalar() {
            return a_value / b_value;
        }

        fn vector(comptime dtype: Dtype, comptime len: usize, a_vec: dtype.Vector(len), b_vec: dtype.Vector(len)) dtype.Vector(len) {
            return a_vec / b_vec;
        }
    });
}
