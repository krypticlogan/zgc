# Fluid dynamics

This example runs a D2Q9 lattice Boltzmann step as a ZGC graph and renders the
resulting macroscopic fields with Raylib. The application owns the nine
distribution values per cell, supplies them to the model each step, and copies
the next distribution back for the following step. The model also outputs
density and both velocity components for rendering.

Run from this directory:

```sh
zig build run -Doptimize=ReleaseFast
```

The 256 × 144 initial field contains two opposing vortices. Its outer edges are
closed no-slip walls: populations that reach a wall bounce into their opposite
D2Q9 direction instead of crossing or wrapping around the domain.

- `Space`: pause or resume
- `N`: advance one step and pause
- `R`: restore the initial vortices
- `P`: switch between data and smoke-particle views
- `V`: switch between speed and density coloring
- `A`: show or hide velocity arrows sampled across the field
- `[` / `]`: decrease or increase the BGK relaxation value, `omega`
- Left mouse drag: stir fluid in the direction of the drag

Mouse motion supplies a soft, localized force field to the graph for the next
solver step. The force is clipped naturally by the closed domain and creates no
additional solid or barrier cells.

The speed view maps `sqrt(ux² + uy²)` to color. The density view shows deviations
around the equilibrium density of one. Both fields are read from model outputs;
rendering does not alter the solver graph.

The particle view contains 7,200 grey smoke particles initially distributed with
greater density in faster parts of the velocity map. They are renderer-owned and
advected through bilinear samples of the velocity outputs; they are not graph
tensors and are not overlaid on the diagnostic data views. The fixed population
keeps the represented smoke volume and memory use bounded.

`omega` is copied into a model input each step, so changes apply without rebuilding
the graph. The controls keep it between 0.6 and 1.7. Velocity arrows use the same
output fields as the color view; arrow direction shows flow direction and arrow
length shows relative speed.
