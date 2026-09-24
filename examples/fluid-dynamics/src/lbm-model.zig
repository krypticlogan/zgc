const std = @import("std");
const zgc = @import("zgc");

pub const H = 144;
pub const W = 256;

// D2Q9:
// 6 2 5
// 3 0 1
// 7 4 8
//
// x grows rightward.
// y grows downward here purely as a grid-coordinate convention.
const Sources = enum(usize) {
    f,
    force_x,
    force_y,
    omega,
    cx,
    cy,
    weights,
};

const Definition = zgc.DefinitionBuilder(Sources, .{
    .max_rank = 3,

    // Collision and streaming make this graph larger than the default limits.
    .max_nodes = 192,
    .max_tensors = 320,
    .max_input_refs = 512,
    .max_outputs = 4,
});

const Value = Definition.TensorValue;

/// Stream a [H,W,1] population through a domain with halfway bounce-back
/// walls. A population that would leave the domain returns in its opposite
/// D2Q9 direction at the same boundary cell.
fn streamSolid(
    b: *Definition,
    comptime moving: Value,
    comptime opposite: Value,
    comptime dx: i8,
    comptime dy: i8,
) Value {
    if (dx == 0 and dy == 0) return moving;

    if (dy == 0) {
        if (dx > 0) {
            const wall = b.slice(opposite, .{ .axis = 1, .start = 0, .end = 1 });
            const interior = b.slice(moving, .{ .axis = 1, .start = 0, .end = W - 1 });
            return b.concat(&.{ wall, interior }, 1);
        }
        const interior = b.slice(moving, .{ .axis = 1, .start = 1, .end = W });
        const wall = b.slice(opposite, .{ .axis = 1, .start = W - 1, .end = W });
        return b.concat(&.{ interior, wall }, 1);
    }

    if (dx == 0) {
        if (dy > 0) {
            const wall = b.slice(opposite, .{ .axis = 0, .start = 0, .end = 1 });
            const interior = b.slice(moving, .{ .axis = 0, .start = 0, .end = H - 1 });
            return b.concat(&.{ wall, interior }, 0);
        }
        const interior = b.slice(moving, .{ .axis = 0, .start = 1, .end = H });
        const wall = b.slice(opposite, .{ .axis = 0, .start = H - 1, .end = H });
        return b.concat(&.{ interior, wall }, 0);
    }

    const moving_rows = if (dy > 0)
        b.slice(moving, .{ .axis = 0, .start = 0, .end = H - 1 })
    else
        b.slice(moving, .{ .axis = 0, .start = 1, .end = H });
    const opposite_rows = if (dy > 0)
        b.slice(opposite, .{ .axis = 0, .start = 1, .end = H })
    else
        b.slice(opposite, .{ .axis = 0, .start = 0, .end = H - 1 });

    const body = if (dx > 0) blk: {
        const wall = b.slice(opposite_rows, .{ .axis = 1, .start = 0, .end = 1 });
        const interior = b.slice(moving_rows, .{ .axis = 1, .start = 0, .end = W - 1 });
        break :blk b.concat(&.{ wall, interior }, 1);
    } else blk: {
        const interior = b.slice(moving_rows, .{ .axis = 1, .start = 1, .end = W });
        const wall = b.slice(opposite_rows, .{ .axis = 1, .start = W - 1, .end = W });
        break :blk b.concat(&.{ interior, wall }, 1);
    };

    if (dy > 0) {
        const wall = b.slice(opposite, .{ .axis = 0, .start = 0, .end = 1 });
        return b.concat(&.{ wall, body }, 0);
    }
    const wall = b.slice(opposite, .{ .axis = 0, .start = H - 1, .end = H });
    return b.concat(&.{ body, wall }, 0);
}

/// Select one D2Q9 population while retaining the channel dimension:
///
/// [H,W,9] -> [H,W,1]
fn channel(
    b: *Definition,
    comptime x: Value,
    comptime i: usize,
) Value {
    return b.slice(x, .{
        .axis = 2,
        .start = i,
        .end = i + 1,
    });
}

fn define(b: *Definition) void {
    // ------------------------------------------------------------
    // Sources
    // ------------------------------------------------------------

    // Distribution function:
    //
    // f[y,x,i]
    //
    // Each cell contains the nine D2Q9 populations.
    const f = b.input(
        .f,
        .f32,
        &.{ H, W, 9 },
    );

    // Caller-controlled momentum injection. One force vector per lattice cell;
    // the singleton channel axis keeps it broadcast-compatible with rho and u.
    const force_x = b.input(.force_x, .f32, &.{ H, W, 1 });
    const force_y = b.input(.force_y, .f32, &.{ H, W, 1 });

    // One runtime relaxation value, broadcast over every distribution.
    const omega = b.input(.omega, .f32, &.{1});

    // Direction vectors:
    //
    // cx = [ 0, 1, 0,-1, 0, 1,-1,-1, 1 ]
    // cy = [ 0, 0, 1, 0,-1, 1, 1,-1,-1 ]
    //
    // Stored as [9], which naturally broadcasts against [H,W,9].
    const cx = b.constant(.cx, .f32, &.{9});
    const cy = b.constant(.cy, .f32, &.{9});

    // D2Q9 lattice weights:
    //
    // [
    //     4/9,
    //     1/9, 1/9, 1/9, 1/9,
    //     1/36,1/36,1/36,1/36,
    // ]
    const weights = b.constant(.weights, .f32, &.{9});

    // ------------------------------------------------------------
    // Scalar constants
    // ------------------------------------------------------------

    const one = b.scalar(.f32, 1.0);
    const three = b.scalar(.f32, 3.0);
    const four_point_five = b.scalar(.f32, 4.5);
    const one_point_five = b.scalar(.f32, 1.5);

    // BGK relaxation parameter.
    //
    // omega = 1 / tau
    //
    // tau > 0.5 in lattice units.
    //
    // The caller supplies omega each step.

    // ============================================================
    // 1. MACROSCOPIC DENSITY
    // ============================================================

    // rho = Σ_i f_i
    //
    // Keep the channel dimension:
    //
    // [H,W,9] -> [H,W,1]
    //
    // This makes subsequent broadcasting against the nine
    // distribution channels straightforward.
    const rho = b.sum(f, .{
        .axes = &.{2},
        .keep_dims = true,
    });

    // ============================================================
    // 2. MACROSCOPIC VELOCITY
    // ============================================================

    // Momentum:
    //
    // jx = Σ_i f_i cx_i
    // jy = Σ_i f_i cy_i

    const momentum_x = b.sum(
        b.mul(f, cx),
        .{
            .axes = &.{2},
            .keep_dims = true,
        },
    );

    const momentum_y = b.sum(
        b.mul(f, cy),
        .{
            .axes = &.{2},
            .keep_dims = true,
        },
    );

    // u = j / rho
    //
    // ux,uy both have shape [H,W,1].
    //
    // Assumption: rho is non-zero everywhere.
    const ux = b.div(momentum_x, rho);
    const uy = b.div(momentum_y, rho);

    // Use the externally supplied impulse to shift collision equilibrium.
    // Zero-valued force fields recover the unforced BGK step exactly.
    const collision_ux = b.add(ux, b.div(force_x, rho));
    const collision_uy = b.add(uy, b.div(force_y, rho));

    // |u|² = ux² + uy²
    const ux_sq = b.mul(collision_ux, collision_ux);
    const uy_sq = b.mul(collision_uy, collision_uy);
    const velocity_sq = b.add(ux_sq, uy_sq);

    // ============================================================
    // 3. DIRECTIONAL VELOCITY DOT PRODUCTS
    // ============================================================

    // e_i · u = cx_i ux + cy_i uy
    //
    // cx [9]       \
    // ux [H,W,1]    -> broadcasting -> [H,W,9]
    //
    // same for y.

    const ex_ux = b.mul(collision_ux, cx);
    const ey_uy = b.mul(collision_uy, cy);

    const eu = b.add(ex_ux, ey_uy);

    const eu_sq = b.mul(eu, eu);

    // ============================================================
    // 4. EQUILIBRIUM DISTRIBUTION
    // ============================================================

    // Standard D2Q9 equilibrium:
    //
    // f_eq_i =
    //
    // w_i rho [
    //     1
    //     + 3(e_i·u)
    //     + 4.5(e_i·u)^2
    //     - 1.5|u|^2
    // ]

    const linear = b.mul(three, eu);

    const quadratic = b.mul(
        four_point_five,
        eu_sq,
    );

    const velocity_term = b.mul(
        one_point_five,
        velocity_sq,
    );

    const equilibrium_poly = b.sub(
        b.add(
            b.add(one, linear),
            quadratic,
        ),
        velocity_term,
    );

    // rho [H,W,1] broadcasts over direction dimension.
    const weighted_rho = b.mul(rho, weights);

    const f_eq = b.mul(
        weighted_rho,
        equilibrium_poly,
    );

    // ============================================================
    // 5. BGK COLLISION
    // ============================================================

    // f* = f + omega(f_eq - f)

    const toward_equilibrium = b.sub(
        f_eq,
        f,
    );

    const collision_delta = b.mul(
        omega,
        toward_equilibrium,
    );

    const post_collision = b.add(
        f,
        collision_delta,
    );

    // ============================================================
    // 6. STREAMING
    // ============================================================

    // Split the post-collision population vector into nine
    // [H,W,1] tensors.

    const f0 = channel(b, post_collision, 0);
    const f1 = channel(b, post_collision, 1);
    const f2 = channel(b, post_collision, 2);
    const f3 = channel(b, post_collision, 3);
    const f4 = channel(b, post_collision, 4);
    const f5 = channel(b, post_collision, 5);
    const f6 = channel(b, post_collision, 6);
    const f7 = channel(b, post_collision, 7);
    const f8 = channel(b, post_collision, 8);

    // Stream each population along its lattice direction.
    //
    // Populations that encounter an outer wall bounce into their opposite
    // direction at the same cell, producing a closed no-slip domain.
    //
    //       6  2  5
    //        \ | /
    //      3 -0- 1
    //        / | \
    //       7  4  8

    const s0 = f0;

    const s1 = streamSolid(b, f1, f3, 1, 0);
    const s2 = streamSolid(b, f2, f4, 0, 1);
    const s3 = streamSolid(b, f3, f1, -1, 0);
    const s4 = streamSolid(b, f4, f2, 0, -1);

    const s5 = streamSolid(b, f5, f7, 1, 1);
    const s6 = streamSolid(b, f6, f8, -1, 1);
    const s7 = streamSolid(b, f7, f5, -1, -1);
    const s8 = streamSolid(b, f8, f6, 1, -1);

    // Reassemble:
    //
    // nine [H,W,1] tensors -> [H,W,9]

    const next_f = b.concat(
        &.{
            s0,
            s1,
            s2,
            s3,
            s4,
            s5,
            s6,
            s7,
            s8,
        },
        2,
    );

    // ============================================================
    // Outputs
    // ============================================================

    // Main recurrent simulation state.
    b.output(next_f);

    // These aren't required for the next step, but they're useful for
    // rendering/inspection and make the example much nicer.
    b.output(rho);
    b.output(ux);
    b.output(uy);
}

pub const definition = blk: {
    @setEvalBranchQuota(200_000);
    var builder = Definition.init();
    define(&builder);
    break :blk builder.finish();
};

pub const FluidStep = definition.model();
