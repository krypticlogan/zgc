# Conway's Game of Life

This example compiles one Conway transition into a ZGC model and renders the
world with Raylib. The application owns the current generation and binds it as
the model input before each step.

```sh
zig build run -Doptimize=ReleaseFast
```

Controls:

- `Space`: pause or resume
- `N`: advance one generation
- `R`: restore the initial patterns
- `C`: clear and pause
- Left mouse button: create cells and pause
- Right mouse button: erase cells and pause

The model uses dead boundaries. It pads the boolean world by one cell, exposes
all 3×3 neighborhoods as one overlapping window view, reduces the neighborhood
axes, and evaluates the survival and birth predicates.
