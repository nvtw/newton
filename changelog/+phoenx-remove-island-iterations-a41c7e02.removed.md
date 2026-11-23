Remove the `island_parallel_iterations` option of `SolverPhoenX`; the single-world sweep already runs every color device-wide, and per-island blocks serialized large packed islands.
