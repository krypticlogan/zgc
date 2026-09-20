const std = @import("std");
const rl = @import("raylib");

pub fn ParticleView(comptime width: usize, comptime height: usize) type {
    return struct {
        const Self = @This();
        pub const particle_count = 7_200;

        const Particle = struct {
            position: rl.Vector2,
            radius: f32,
            shade: u8,
            alpha: u8,
        };

        particles: [particle_count]Particle,

        /// Seed a finite smoke volume, weighted toward faster regions of the
        /// same velocity field used by the diagnostic renderer.
        pub fn reset(self: *Self, velocity_x: []const f32, velocity_y: []const f32) void {
            assertVelocityShape(velocity_x, velocity_y);
            var random_state: u64 = 0x4d595df4d0f33173;
            var maximum_speed_squared: f32 = 0;
            for (velocity_x, velocity_y) |vx, vy| {
                maximum_speed_squared = @max(maximum_speed_squared, vx * vx + vy * vy);
            }

            for (&self.particles) |*particle| {
                var position: rl.Vector2 = undefined;
                while (true) {
                    position = .init(
                        randomUnit(&random_state) * @as(f32, @floatFromInt(width)),
                        randomUnit(&random_state) * @as(f32, @floatFromInt(height)),
                    );
                    const velocity = sampleVelocity(position, velocity_x, velocity_y);
                    const speed_squared = velocity.x * velocity.x + velocity.y * velocity.y;
                    const relative_speed = if (maximum_speed_squared > 0)
                        @sqrt(speed_squared / maximum_speed_squared)
                    else
                        0;
                    // Preserve some smoke in calm areas while making the
                    // initial distribution visibly follow the speed map.
                    const acceptance = 0.1 + 0.9 * relative_speed;
                    if (randomUnit(&random_state) <= acceptance) break;
                }

                particle.* = .{
                    .position = position,
                    .radius = 0.65 + randomUnit(&random_state) * 1.0,
                    .shade = @intFromFloat(145.0 + randomUnit(&random_state) * 75.0),
                    .alpha = @intFromFloat(80.0 + randomUnit(&random_state) * 110.0),
                };
            }
        }

        /// Advect the renderer-owned particles through the latest velocity view.
        pub fn update(self: *Self, velocity_x: []const f32, velocity_y: []const f32) void {
            assertVelocityShape(velocity_x, velocity_y);

            for (&self.particles) |*particle| {
                const velocity = sampleVelocity(particle.position, velocity_x, velocity_y);
                particle.position.x = contain(
                    particle.position.x + velocity.x,
                    @floatFromInt(width),
                );
                particle.position.y = contain(
                    particle.position.y + velocity.y,
                    @floatFromInt(height),
                );
            }
        }

        pub fn draw(self: *const Self, cell_size: i32, top: i32) void {
            const scale: f32 = @floatFromInt(cell_size);
            const top_f: f32 = @floatFromInt(top);
            for (self.particles) |particle| {
                rl.drawCircleV(
                    .init(
                        (particle.position.x + 0.5) * scale,
                        top_f + (particle.position.y + 0.5) * scale,
                    ),
                    particle.radius,
                    rl.Color.init(particle.shade, particle.shade, particle.shade, particle.alpha),
                );
            }
        }

        fn randomUnit(state: *u64) f32 {
            state.* ^= state.* >> 12;
            state.* ^= state.* << 25;
            state.* ^= state.* >> 27;
            const value: u32 = @truncate((state.* *% 0x2545f4914f6cdd1d) >> 32);
            return @as(f32, @floatFromInt(value)) / @as(f32, @floatFromInt(std.math.maxInt(u32)));
        }

        fn contain(value: f32, extent: f32) f32 {
            return std.math.clamp(value, 0.0, extent - 0.0001);
        }

        fn sampleVelocity(
            position: rl.Vector2,
            velocity_x: []const f32,
            velocity_y: []const f32,
        ) rl.Vector2 {
            const x0: usize = @intFromFloat(@floor(position.x));
            const y0: usize = @intFromFloat(@floor(position.y));
            const x1 = @min(x0 + 1, width - 1);
            const y1 = @min(y0 + 1, height - 1);
            const tx = position.x - @as(f32, @floatFromInt(x0));
            const ty = position.y - @as(f32, @floatFromInt(y0));

            return .init(
                bilinear(velocity_x, x0, y0, x1, y1, tx, ty),
                bilinear(velocity_y, x0, y0, x1, y1, tx, ty),
            );
        }

        fn bilinear(
            field: []const f32,
            x0: usize,
            y0: usize,
            x1: usize,
            y1: usize,
            tx: f32,
            ty: f32,
        ) f32 {
            const top = std.math.lerp(field[y0 * width + x0], field[y0 * width + x1], tx);
            const bottom = std.math.lerp(field[y1 * width + x0], field[y1 * width + x1], tx);
            return std.math.lerp(top, bottom, ty);
        }

        fn assertVelocityShape(velocity_x: []const f32, velocity_y: []const f32) void {
            std.debug.assert(velocity_x.len == width * height);
            std.debug.assert(velocity_y.len == width * height);
        }
    };
}
