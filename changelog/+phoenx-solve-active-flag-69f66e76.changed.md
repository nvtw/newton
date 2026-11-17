Speed up `SolverPhoenX` exact joint solves by skipping solves that have no work inside the solve kernels instead of with a CUDA graph conditional.
