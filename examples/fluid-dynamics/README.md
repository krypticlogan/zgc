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

The 256 × 144 initial field contains two opposing vortices. Streaming uses
shape-preserving shifts with wrap boundaries, so populations leaving one edge
re-enter at the opposite edge. The model still splits and reassembles the nine
population channels around these shifts.

- `Space`: pause or resume
- `N`: advance one step and pause
- `R`: restore the initial vortices
- `V`: switch between speed and density coloring
- `A`: show or hide velocity arrows sampled across the field
- `[` / `]`: decrease or increase the BGK relaxation value, `omega`
- Left mouse drag: stir fluid in the direction of the drag

Mouse motion supplies a soft, localized force field to the graph for the next
solver step. The force wraps around the periodic domain near its edges and
creates no solid or barrier cells.

The speed view maps `sqrt(ux² + uy²)` to color. The density view shows deviations
around the equilibrium density of one. Both fields are read from model outputs;
rendering does not alter the solver graph.

`omega` is copied into a model input each step, so changes apply without rebuilding
the graph. The controls keep it between 0.6 and 1.7. Velocity arrows use the same
output fields as the color view; arrow direction shows flow direction and arrow
length shows relative speed.
