Speed up mesh-SDF and hydroelastic contact generation without changing the
generated contacts:

  - The mesh-SDF edge search compacts contact candidates through a second
    cooperative stack so the gradient sample and reducer export run on full
    warps, merges the two endpoint checks into one fetch, and skips the Brent
    search for edges whose three initial samples already prove, via the
    1-Lipschitz bound, that no point can reach the contact threshold.
  - Texture SDF samplers select the coarse or subgrid storage before issuing
    their fetches instead of branching around them, and the reducer's
    pre-prune probes resolve both hashtable entries and read every slot
    before comparing.
  - The hydroelastic generation kernel ranks marching-cubes faces by index and
    re-extracts the winners on write instead of carrying every candidate's
    payload through the loop.
