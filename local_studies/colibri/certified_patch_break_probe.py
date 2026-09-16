# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Source-only instrumentation of the certified-domain break veto.

Call ``instrument(source)`` before importing the generated material module.
The probe does not change the veto, membership, forces, or history.
"""


def instrument(source):
    marker = "    stats: wp.array[int]\n"
    assert source.count(marker) == 1
    source = source.replace(marker, marker + "    break_probe: wp.array2d[float]\n")
    marker = "    broken = wp.bool(False)\n"
    assert source.count(marker) == 1
    source = source.replace(
        marker,
        marker
        + """    probe_slot = domain.stats[0] % domain.break_probe.shape[1]
    for row in range(7):
        domain.break_probe[row, probe_slot] = 0.0
    domain.break_probe[0, probe_slot] = float(domain.stats[0])
""",
    )
    marker = "    for k in range(domain.previous_count[0]):\n"
    assert source.count(marker) == 1
    source = source.replace(
        marker,
        marker
        + """        if domain.previous_member[k] != 0:
            load = cc.prev_impulses[0, k]
            flag = cc.prev_lambdas[12, k] != 0.0
            row = int(1)
            if flag:
                row = 2
                if load > 0.0:
                    row = 3
                    domain.break_probe[5, probe_slot] += load
                    domain.break_probe[6, probe_slot] += wp.sqrt(cc.prev_impulses[1, k] * cc.prev_impulses[1, k] + cc.prev_impulses[2, k] * cc.prev_impulses[2, k])
            domain.break_probe[row, probe_slot] += 1.0
            domain.break_probe[4, probe_slot] += load
""",
    )
    marker = "    domain.stats = wp.zeros(7, dtype=int, device=cc.lambdas.device)\n"
    assert source.count(marker) == 1
    source = source.replace(
        marker, marker + "    domain.break_probe = wp.zeros((7, 128), dtype=float, device=cc.lambdas.device)\n"
    )
    marker = "    def finish():\n"
    assert source.count(marker) == 1
    source = source.replace(
        marker,
        marker
        + """        np.savez(Path(output).with_suffix('.patch_break_probe.npz'), records=domain.break_probe.numpy(), fields=np.asarray(['ingest', 'unbroken_members', 'broken_zero_load', 'broken_positive_load', 'total_normal_impulse', 'broken_normal_impulse', 'broken_tangent_impulse']))
""",
    )
    return source
