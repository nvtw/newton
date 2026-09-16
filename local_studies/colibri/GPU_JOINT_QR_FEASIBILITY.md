# GPU dense QR feasibility

The frozen geometry-consistent218x184 matrix was factored with
Torch2.11.0+cu128's CUDA vendor QR on RTX PRO6000 Blackwell. Complete QR
reconstruction error6.14e-16 and orthogonality error1.33e-15 pass. The
all188-row correction equation residual is4.32e-11; physical mobility agrees
with the CPU reference to2.09e-11 relative. No modes were dropped.

Seven batches of50 calls after10warmups measured median1.67035 ms/call in
eager execution and1.62617 ms under CUDA graph replay. This includes only
QR of an already assembled frozen matrix. Geometry, rank certification,
reduced mass assembly, contact solve and state scatter are excluded.

At two final relaxations per60Hz frame, graph QR alone would cost3.25235 ms.
At60substeps/frame it would cost97.5704 ms. These arithmetic cost estimates
are not complete-solver FPS predictions. Dense vendor QR at every substep
is not competitive with the validated11.08 ms/frame solver. A smaller
tree-condensed loop/drive factor is the next implementation direction.

Reproduce: uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python
-m local_studies.colibri.benchmark_joint_qr_gpu. Result:
/tmp/colibri_joint_qr_gpu_benchmark.json. No production dependency or runtime
source changed; Torch is used only by this local feasibility benchmark.
