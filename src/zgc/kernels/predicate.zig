fn compare(a: anytype, b: anytype, output: anytype, comptime Predicate: type) void {
    const Output = @TypeOf(output);
    const shape = if (comptime @hasDecl(Output, "geometry_is_static") and Output.geometry_is_static)
        Output.static_shape
    else
        output.shape;
    const lhs = a.broadcastTo(Output.rank, shape);
    const rhs = b.broadcastTo(Output.rank, shape);

    for (0..output.len()) |linear_index| {
        const output_index = output.elementOffsetFromLinear(linear_index);
        const lhs_index = lhs.elementOffsetFromLinear(linear_index);
        const rhs_index = rhs.elementOffsetFromLinear(linear_index);
        output.storage[output_index] = Predicate.apply(lhs.storage[lhs_index], rhs.storage[rhs_index]);
    }
}

fn logicalBinary(a: anytype, b: anytype, output: anytype, comptime Operator: type) void {
    compare(a, b, output, Operator);
}

pub fn equal(a: anytype, b: anytype, output: anytype) void {
    compare(a, b, output, struct {
        fn apply(lhs: anytype, rhs: @TypeOf(lhs)) bool {
            return lhs == rhs;
        }
    });
}

pub fn notEqual(a: anytype, b: anytype, output: anytype) void {
    compare(a, b, output, struct {
        fn apply(lhs: anytype, rhs: @TypeOf(lhs)) bool {
            return lhs != rhs;
        }
    });
}

pub fn lessThan(a: anytype, b: anytype, output: anytype) void {
    compare(a, b, output, struct {
        fn apply(lhs: anytype, rhs: @TypeOf(lhs)) bool {
            return lhs < rhs;
        }
    });
}

pub fn lessEqual(a: anytype, b: anytype, output: anytype) void {
    compare(a, b, output, struct {
        fn apply(lhs: anytype, rhs: @TypeOf(lhs)) bool {
            return lhs <= rhs;
        }
    });
}

pub fn greaterThan(a: anytype, b: anytype, output: anytype) void {
    compare(a, b, output, struct {
        fn apply(lhs: anytype, rhs: @TypeOf(lhs)) bool {
            return lhs > rhs;
        }
    });
}

pub fn greaterEqual(a: anytype, b: anytype, output: anytype) void {
    compare(a, b, output, struct {
        fn apply(lhs: anytype, rhs: @TypeOf(lhs)) bool {
            return lhs >= rhs;
        }
    });
}

pub fn logicalNot(input: anytype, output: anytype) void {
    for (0..output.len()) |linear_index| {
        const input_index = input.elementOffsetFromLinear(linear_index);
        const output_index = output.elementOffsetFromLinear(linear_index);
        output.storage[output_index] = !input.storage[input_index];
    }
}

pub fn logicalAnd(a: anytype, b: anytype, output: anytype) void {
    logicalBinary(a, b, output, struct {
        fn apply(lhs: bool, rhs: bool) bool {
            return lhs and rhs;
        }
    });
}

pub fn logicalOr(a: anytype, b: anytype, output: anytype) void {
    logicalBinary(a, b, output, struct {
        fn apply(lhs: bool, rhs: bool) bool {
            return lhs or rhs;
        }
    });
}

pub fn where(condition: anytype, when_true: anytype, when_false: anytype, output: anytype) void {
    const Output = @TypeOf(output);
    const shape = if (comptime @hasDecl(Output, "geometry_is_static") and Output.geometry_is_static)
        Output.static_shape
    else
        output.shape;
    const mask = condition.broadcastTo(Output.rank, shape);
    const true_values = when_true.broadcastTo(Output.rank, shape);
    const false_values = when_false.broadcastTo(Output.rank, shape);

    for (0..output.len()) |linear_index| {
        const output_index = output.elementOffsetFromLinear(linear_index);
        const mask_index = mask.elementOffsetFromLinear(linear_index);
        const true_index = true_values.elementOffsetFromLinear(linear_index);
        const false_index = false_values.elementOffsetFromLinear(linear_index);
        output.storage[output_index] = if (mask.storage[mask_index])
            true_values.storage[true_index]
        else
            false_values.storage[false_index];
    }
}
