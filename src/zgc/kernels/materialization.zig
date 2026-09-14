/// Copy logical tensor values into distinct output storage. Shape, dtype, and
/// layout compatibility are established before this kernel is instantiated.
pub fn copy(input: anytype, output: anytype) void {
    if (input.contiguousSlice()) |input_values| {
        if (output.contiguousSlice()) |output_values| {
            @memcpy(output_values, input_values);
            return;
        }
    }

    for (0..output.len()) |linear_index| {
        const input_index = input.elementOffsetFromLinear(linear_index);
        const output_index = output.elementOffsetFromLinear(linear_index);
        output.storage[output_index] = input.storage[input_index];
    }
}
