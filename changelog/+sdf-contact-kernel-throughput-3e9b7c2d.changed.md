Speed up mesh-SDF and hydroelastic contact generation without changing the
generated contacts:

  - The mesh-SDF edge search compacts contact candidates through a second
    cooperative stack so the gradient sample and reducer export run on full
    warps, merges the two endpoint checks into one fetch, and skips the Brent
    search for edges whose three initial samples already prove, via the
    1-Lipschitz bound, that no point can reach the contact threshold.
  - Texture SDF samplers select the coarse or subgrid storage before issuing
    their fetches instead of branching around them.
  - The hydroelastic generation and first refinement kernels declare launch
    bounds so the compiler stops allocating every register, raising residency
    from 8 to 12 warps per SM (the generation kernel runs 45% faster at 80
    nut/bolt worlds).
  - The light hydroelastic grid-stride kernels (scatters, reduce, export,
    decode) launch a grid sized from the traversed narrow-band tiles instead
    of the full configured grid, which removes most of their fixed cost in
    small scenes (about 10% of collide time at 5 nut/bolt worlds).
  - Hydroelastic octree refinement scans only the live prefix of its
    worst-case-sized count buffers with a tile-based chunked scan, and the
    scatter kernels publish the next level's count themselves instead of a
    separate single-thread launch per level.
