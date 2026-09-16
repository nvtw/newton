# Bilateral preparation launch-size experiment

The canonical preparation kernel uses the default block size. The local
`bilateral_launch_size.py` changes only its launch block size to32, allowing
small joint batches to span multiple blocks. Equations, operation ordering,
precision, preparation frequency and all physics settings remain identical.

Isolated330-frame public Colibri test,30 warmup and300 measured, at30 substeps
per120Hz contact update: baseline12.411642ms/frame (80.57FPS), candidate
12.616703ms/frame (79.26FPS). Both pass all checks, and all ten saved arrays
match exactly. Peers held GPU work and the compute-process list was empty
before launch. This does not demonstrate a performance benefit; do not promote.

Reports: `/tmp/colibri_fused_begin_baseline_ab330.json` and
`/tmp/colibri_bilateral_block32_330.json`, with matching NPZ trajectory files.
