# FP64 reciprocal-diagonal experiment

Local producer/consumer replacement computes inverse LDL pivots during
preparation and multiplies by them in the joint sweep. All factorization,
residual and impulse arithmetic remains FP64; the law and solver schedule
are unchanged. Rounding can differ in principle. Unique Warp function names
avoid the aliasing discovered in the separate prefix profiler.

Validation:

-480CPU factor cases through diagonal spread1e16: maximum normalized
 backward error1.19e-16, relative solution difference2.77e-16.
-Three GPU joint-policy tests and the four-case mixed joint/contact
 conservation test pass (four unittest methods).
-Public330frames pass; all ten saved arrays match baseline byte-for-byte.

Isolated timing:12.372005ms/frame (80.83FPS) versus prior adjacent-work
baseline12.411642ms (80.57FPS). The0.32% difference does not establish a
benefit; no production promotion. Other agent GPU launches were held.

Artifacts: `/tmp/colibri_reciprocal_bilateral_cpu.json`,
`/tmp/colibri_reciprocal_bilateral_tests.log`,
`/tmp/colibri_reciprocal_bilateral330.json` and `.npz`.

Run the local screen with:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.reciprocal_bilateral local_studies.colibri.check_public_analytic_gradient --frames 330 --substeps 30 --save-history --output /tmp/colibri_reciprocal_repeat.json
```
