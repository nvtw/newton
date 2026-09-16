"""CPU first-fit oracle; CUDA checks require an explicitly reserved window."""

import argparse
import json

import numpy as np


def oracle(data, endpoints, active, width):
    """Use independent per-body Python sets, preserving inactive output bytes."""
    data["masks"].fill(0)
    data["counts"].fill(0)
    used = [set() for _ in range(len(data["masks"]))]
    colors = 0
    for row in range(active):
        bodies = [int(body) for body in endpoints[row] if body >= 0]
        occupied = set().union(*(used[body] for body in bodies))
        chosen = 0
        while chosen in occupied:
            chosen += 1
        for body in bodies:
            used[body].add(chosen)
            data["masks"][body, chosen // 32] |= np.uint32(1 << (chosen % 32))
        data["row_color"][row] = chosen
        data["row_partition"][row] = chosen // width
        data["counts"][chosen] += 1
        colors = max(colors, chosen + 1)
    data["num_colors"][0] = colors
    data["starts"][0] = 0
    for color in range(colors):
        data["starts"][color + 1] = data["starts"][color] + data["counts"][color]
        data["cursors"][color] = data["starts"][color]
    for row in range(active):
        color = data["row_color"][row]
        index = data["cursors"][color]
        data["ids"][index] = row
        data["cursors"][color] += 1


def cases():
    """Cover cache boundaries, body zero, repeated endpoints, and fallback graphs."""
    rng = np.random.default_rng(739)
    for bodies, capacity, rigid in (
        (1, 320, True),
        (38, 512, True),
        (128, 320, True),
        (129, 320, True),
        (19, 320, False),
        (2, 1, True),
    ):
        endpoints = np.full((capacity, 8), -1, np.int32)
        count = 2 if rigid else 8
        endpoints[:, :count] = rng.integers(-1, bodies, (capacity, count))
        # More than 256 colors must fall back to global words without truncation.
        endpoints[: min(300, capacity), :count] = 0
        if capacity > 300:
            endpoints[300] = -1
        yield bodies, endpoints, rigid


def allocate_host(capacity, bodies):
    """Nonzero sentinels expose accidental clearing of inactive fields."""
    return {
        "masks": np.full((bodies, (capacity + 31) // 32), 17, np.uint32),
        **{
            name: np.full(capacity + 1, 17, np.int32)
            for name in ("row_color", "row_partition", "counts", "starts", "cursors", "ids")
        },
        "num_colors": np.full(1, 17, np.int32),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Requires explicit GPU reservation")
    args = parser.parse_args()
    if args.cuda:
        import warp as wp

        from local_studies.colibri.shared_color_rows import build
        from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
            ElementInteractionData,
        )
        from newton._src.solvers.phoenx.mass_splitting import color_groups

        wp.init()
    checks = 0
    for bodies, endpoints, rigid in cases():
        capacity = len(endpoints)
        expected = allocate_host(capacity, bodies)
        if args.cuda:
            elements = wp.zeros(capacity, dtype=ElementInteractionData, device="cuda:0")
            host = elements.numpy()
            host["bodies"] = endpoints
            elements.assign(host)
            active = wp.zeros(1, dtype=wp.int32, device="cuda:0")
            states = [color_groups.allocate(capacity, bodies, "cuda:0") for _ in range(2)]
            for state in states:
                for name in state:
                    state[name].assign(expected[name])
        for width in (1, 4, 8, 33):
            for count in (capacity, min(256, capacity), 0, 1, min(257, capacity), capacity):
                oracle(expected, endpoints, count, width)
                assert int(expected["counts"].sum()) == count
                assert sorted(expected["ids"][:count].tolist()) == list(range(count))
                for row in range(count):
                    for previous in range(row):
                        if expected["row_color"][row] == expected["row_color"][previous]:
                            assert not set(endpoints[row][endpoints[row] >= 0]) & set(
                                endpoints[previous][endpoints[previous] >= 0]
                            )
                if args.cuda:
                    active.assign(np.array([count], np.int32))
                    for state, builder in zip(states, (color_groups.build, build), strict=True):
                        # First eager launch compiles; captured repeat exercises the same buffers.
                        builder(state, elements, active, width, "cuda:0", rigid_only=rigid)
                        with wp.ScopedCapture(device="cuda:0") as capture:
                            builder(state, elements, active, width, "cuda:0", rigid_only=rigid)
                        wp.capture_launch(capture.graph)
                        for name, reference in expected.items():
                            actual = state[name].numpy()
                            assert actual.tobytes() == reference.tobytes(), (
                                bodies,
                                capacity,
                                rigid,
                                width,
                                count,
                                name,
                            )
                checks += 1
    print(json.dumps({"cases": checks, "cuda": args.cuda, "status": "passed"}))


if __name__ == "__main__":
    main()
