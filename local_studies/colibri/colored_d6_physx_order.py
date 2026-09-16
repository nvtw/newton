# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Combine PhysX static-last scheduling and drive/angular/linear D6 order.

Only reorder complete existing rows; all coefficients, impulses and physical
responses retain their original indices. Restrict this control to one hinge.
"""

from . import colored_d6_scalar, colored_d6_static_last


def ordered_iteration(source):
    """Visit drive rows before angular hard rows before linear hard rows."""
    start = source.index("        for i in range(data.row_count[cid]):")
    end = source.index("        _ms_store_body_pair", start)
    body = source[start:end]
    marker = "            j0 = data.wrench0[structural, local]"
    assert body.count(marker) == 1
    body = body.replace(
        marker,
        """            family = wp.int32(2)
            if data.row_dynamic[row]:
                family = wp.int32(0)
            elif local >= wp.int32(3):
                family = wp.int32(1)
            if family != group:
                continue
"""
        + marker,
    )
    body = "        for group in range(3):\n" + "".join(
        "    " + line if line.strip() else line for line in body.splitlines(True)
    )
    return source[:start] + body + source[end:]


def main():
    """Install the bounded row-order control before importing Newton."""
    colored_d6_scalar.ITERATE = ordered_iteration(colored_d6_scalar.ITERATE)
    colored_d6_static_last.main()


if __name__ == "__main__":
    main()
