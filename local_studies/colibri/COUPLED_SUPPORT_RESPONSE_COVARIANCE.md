# Frozen coupled physical response covariance

The CPU audit reconstructs the accepted contact impulse response using a
combined hard-joint/compliant-drive linear system. It agrees with the previous
mass-nullspace response. Twelve random proper rotations and translations,
including six reversed body orders, preserve the response, reaction impulses,
kinetic work and momentum balance against the static support reaction.

This checks the physical response at the saved accepted contact impulses.
It does not re-solve the nonlinear contact problem, exercise the native GPU
implementation, or establish live scheduling correctness.

Run `uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python
-m local_studies.colibri.audit_support_response_covariance` from this worktree.
Report: `/tmp/colibri_support_response_covariance.json`.
