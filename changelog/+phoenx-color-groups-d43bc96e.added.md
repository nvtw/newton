Add the experimental `mass_splitting_color_group_size` option to
`SolverPhoenX` so sequential constraint colors can share mass copies in
small CUDA rigid mechanisms with block PGS joints and point contacts.
Keep existing scheduling when the option is zero.
