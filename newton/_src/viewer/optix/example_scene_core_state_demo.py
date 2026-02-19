"""State-machine style demo inspired by SceneCore.cs.

This sample demonstrates handle-based scene editing operations without
requiring OptiX runtime objects.
"""

from __future__ import annotations

import argparse

import numpy as np

from newton._src.viewer.optix.scene_core import SceneCore


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--instances", type=int, default=4)
    args = parser.parse_args()

    scene = SceneCore()
    cube = scene.add_cube(1.0)
    plane = scene.add_plane(10.0, 10.0)
    print(f"Added meshes: cube={cube.value}, plane={plane.value}, state={int(scene.state)}")

    for i in range(max(args.instances, 1)):
        t = np.eye(4, dtype=np.float32)
        t[0, 3] = float(i) * 1.5
        scene.add_instance(t, cube, hit_kernel_sbt_offset=0)
    print(f"Instances: {scene.get_instance_count()}, state={int(scene.state)}")

    if scene.get_instance_count() > 0:
        first = next(iter(scene.instances.items()))[0]
        t2 = np.eye(4, dtype=np.float32)
        t2[1, 3] = 2.0
        scene.set_instance_transform(first, t2)
    print(f"After transform edit, state={int(scene.state)}")


if __name__ == "__main__":
    main()
