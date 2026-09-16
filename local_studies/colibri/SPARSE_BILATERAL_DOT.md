# Exact-zero bilateral dot-product experiment

At the saved Colibri pose,1200of2256active joint-wrench components are
exactly zero (53.19%). The local callback skips these coefficients while
retaining FP64 products and the original accumulation order for nonzeros.
Preparation and impulse scatter remain canonical. This assumes finite inputs.

Validation:10,000CPU finite-input dot products, including signed zeros and
subnormals, match bitwise. Three joint-policy GPU test methods plus the
four-case mixed-joint/contact conservation method pass. Public330frames
pass and all ten saved arrays match the baseline byte-for-byte.

Isolated mean12.952167ms/frame (77.21FPS), versus baseline12.411642ms
(80.57FPS). This is slower, so the prototype is rejected and no production
change is made. The extra per-component branches did not provide a benefit.
Other GPU jobs were held throughout measurement.

Artifacts: `/tmp/colibri_sparse_bilateral_dot_cpu.json`,
`/tmp/colibri_sparse_bilateral_dot_tests.log`,
`/tmp/colibri_sparse_bilateral_dot330.json` and `.npz`.

Reproduce with:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.sparse_bilateral_dot local_studies.colibri.check_public_analytic_gradient --frames 330 --substeps 30 --save-history --output /tmp/colibri_sparse_dot_repeat.json
```
