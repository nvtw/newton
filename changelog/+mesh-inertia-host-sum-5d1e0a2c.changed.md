Speed up mesh mass property computation by reducing per-triangle sums on the host in one copy instead of a GPU scan and readback per component.
