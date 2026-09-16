"""Hash both collision-update boundaries and post-frame states without changing solver code."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri import check_slab_colibri as slab
from local_studies.colibri.check_public_slab_envelope import PublicSlabExample
from local_studies.colibri.conservative_mesh_candidates import install_conservative_mesh_candidates

runner = slab.runner
runner.Example = PublicSlabExample
runner.install_fused = slab.install_fused
_restore_envelope = install_conservative_mesh_candidates()


def take_arg(name, default=None):
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    result = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return result


destination = Path(take_arg("--hash-dir", "/tmp/colibri_hash_legacy"))
reference_arg = take_arg("--hash-reference")
hashes_only = take_arg("--hashes-only", "0") == "1"
reference = Path(reference_arg) if reference_arg else None
destination.mkdir(parents=True, exist_ok=True)
reference_hashes = json.loads((reference / "hashes.json").read_text()) if reference else None
instances = []
original_example = runner.Example
original_capture = wp.ScopedCapture


class HashExample(original_example):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.hash_frame = 0
        self.hashes = []
        self.first_difference = None
        self.first_state_difference = None
        self.state_history = []
        self.phase_buffers = []
        self.hooks_installed = False
        instances.append(self)

    def install_hash_hooks(self):
        if self.hooks_installed:
            return
        self.hooks_installed = True
        world = self.solver.world
        scratch = world._ingest_scratch
        contacts = self.contacts
        before = {}
        for name in (
            "rigid_contact_count",
            "contact_generation",
            "rigid_contact_shape0",
            "rigid_contact_shape1",
            "rigid_contact_point0",
            "rigid_contact_point1",
            "rigid_contact_normal",
            "rigid_contact_margin0",
            "rigid_contact_margin1",
            "rigid_contact_match_index",
        ):
            before[name] = getattr(contacts, name)
        before["body_position"] = world.bodies.position
        before["body_orientation"] = world.bodies.orientation
        before["shape_transform"] = self.model.shape_transform
        after = {
            "num_columns": scratch.num_contact_columns,
            "num_pairs": scratch.num_pairs,
            "header": world._contact_cols.data,
            "owner": world._contact_cols.articulation_owner,
            "cid_cur": world._cid_of_contact_cur,
            "cid_prev": world._cid_of_contact_prev,
            "cc_valid_count": world._cc_valid_count,
            "reuse_indices": world._reuse_contact_indices,
            "last_generation": world._last_contact_generation,
        }
        for name in (
            "pair_source_idx",
            "pair_first",
            "pair_count",
            "pair_columns",
            "pair_col_offset",
            "pair_id",
            "inv_sort_perm",
            "prev_inv_sort_perm",
            "sorted_match_index",
        ):
            value = getattr(scratch, name, None)
            if value is not None:
                after[name] = value
        for name in ("impulses", "prev_impulses", "lambdas", "prev_lambdas", "derived"):
            after["cc_" + name] = getattr(world._contact_container, name)
        for _ in range(2):
            self.phase_buffers.append(
                {
                    "before": {k: wp.empty_like(v) for k, v in before.items()},
                    "after": {k: wp.empty_like(v) for k, v in after.items()},
                }
            )
        original_ingest = world._ingest_and_warmstart_contacts
        counter = [0]

        def ingest(*args, **kwargs):
            slot = counter[0] % 2
            counter[0] += 1
            for k, v in before.items():
                wp.copy(self.phase_buffers[slot]["before"][k], v)
            original_ingest(*args, **kwargs)
            for k, v in after.items():
                wp.copy(self.phase_buffers[slot]["after"][k], v)

        world._ingest_and_warmstart_contacts = ingest

    def step(self):
        super().step()
        self.hash_frame += 1
        arrays = {"post.q": self.state_0.body_q.numpy(), "post.qd": self.state_0.body_qd.numpy()}
        for slot, buffers in enumerate(self.phase_buffers):
            before = {k: v.numpy() for k, v in buffers["before"].items()}
            after = {k: v.numpy() for k, v in buffers["after"].items()}
            count = int(before["rigid_contact_count"][0])
            columns = int(after["num_columns"][0])
            pairs = int(after["num_pairs"][0])
            for k, v in before.items():
                arrays[f"outer{slot}.before.{k}"] = v if v.size == 1 else v[:count]
            for k, source_value in after.items():
                v = source_value
                if k == "header":
                    v = v[:, :columns]
                elif k in ("owner", "pair_source_idx"):
                    v = v[:columns]
                elif k in ("pair_first", "pair_count", "pair_columns", "pair_col_offset"):
                    v = v[:pairs]
                elif k in ("pair_id", "inv_sort_perm", "sorted_match_index"):
                    v = v[:count]
                elif k in ("cc_impulses", "cc_lambdas", "cc_derived"):
                    v = v[:, :count]
                arrays[f"outer{slot}.after.{k}"] = v
        self.state_history.append((arrays["post.q"], arrays["post.qd"]))
        hashes = {
            k: hashlib.sha256(str((v.dtype.str, v.shape)).encode() + v.tobytes()).hexdigest() for k, v in arrays.items()
        }
        self.hashes.append(hashes)
        if reference is None:
            if not hashes_only:
                np.savez_compressed(destination / f"frame{self.hash_frame:04d}.npz", **arrays)
        else:
            expected = reference_hashes[self.hash_frame - 1]
            different = [k for k in sorted(set(hashes) | set(expected)) if hashes.get(k) != expected.get(k)]
            if different and self.first_difference is None:
                self.first_difference = {"frame": self.hash_frame, "fields": different}
                np.savez_compressed(destination / "first_difference.npz", **arrays)
                (destination / "first_difference.json").write_text(json.dumps(self.first_difference, indent=2))
                print("FIRST_DIFFERENCE", json.dumps(self.first_difference), flush=True)
            state_fields = [k for k in ("post.q", "post.qd") if hashes[k] != expected[k]]
            if state_fields and self.first_state_difference is None:
                self.first_state_difference = {
                    "frame": self.hash_frame,
                    "fields": state_fields,
                    "all_different_fields": different,
                }
                np.savez_compressed(
                    destination / "first_state_difference.npz", **arrays, shape_labels=self.model.shape_label
                )
                baseline = np.load(reference / "states.npz")
                np.savez_compressed(
                    destination / "first_state_reference.npz",
                    q=baseline["q"][self.hash_frame - 1],
                    qd=baseline["qd"][self.hash_frame - 1],
                )
                (destination / "first_state_difference.json").write_text(
                    json.dumps(self.first_state_difference, indent=2)
                )
                print("FIRST_STATE_DIFFERENCE", json.dumps(self.first_state_difference), flush=True)

        (destination / "hashes.json").write_text(json.dumps(self.hashes))
        if self.hash_frame % 10 == 0:
            print("HASH_FRAME", self.hash_frame, flush=True)


def capture(*args, **kwargs):
    if instances:
        instances[0].install_hash_hooks()
    return original_capture(*args, **kwargs)


if __name__ == "__main__":
    runner.Example = HashExample
    wp.ScopedCapture = capture
    try:
        runner.main()
    finally:
        _restore_envelope()
        wp.ScopedCapture = original_capture
        if instances:
            np.savez_compressed(
                destination / "states.npz",
                q=np.array([x[0] for x in instances[0].state_history]),
                qd=np.array([x[1] for x in instances[0].state_history]),
            )
            (destination / "summary.json").write_text(
                json.dumps(
                    {
                        "frames": instances[0].hash_frame,
                        "first_difference": instances[0].first_difference,
                        "first_state_difference": instances[0].first_state_difference,
                        "note": "Captured-copy and host-hash diagnostic; no performance claim.",
                    },
                    indent=2,
                )
            )
