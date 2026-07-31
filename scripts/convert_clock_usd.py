# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert the Analog Digital Clock asset from PhysX to Newton USD data."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Sdf, Usd


def convert(source: Path, output: Path, max_sdf_resolution: int) -> None:
    """Strip PhysX caches and author Newton SDF collision settings.

    Args:
        source: Source clock USD path.
        output: Destination path for the converted USD.
        max_sdf_resolution: Maximum generated SDF resolution.
    """
    if source.resolve() == output.resolve():
        raise ValueError("Source and output paths must differ")
    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None:
        raise ValueError(f"Unable to open USD stage: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if not source_stage.GetRootLayer().Export(str(output)):
        raise RuntimeError(f"Unable to export USD stage: {output}")

    stage = Usd.Stage.Open(str(output))
    root_layer = stage.GetRootLayer()
    converted_sdfs = 0
    removed_bytes = 0

    for prim in stage.Traverse():
        property_names = {prop.GetName() for prop in prim.GetProperties()}
        if any(name.startswith("material:binding") for name in property_names):
            prim.AddAppliedSchema("MaterialBindingAPI")
        if any(name.startswith("drive:angular:") for name in property_names):
            prim.AddAppliedSchema("PhysicsDriveAPI:angular")

        for prop in list(prim.GetProperties()):
            name = prop.GetName()
            if name.startswith("physxCookedData:"):
                value = prop.Get() if hasattr(prop, "Get") else None
                removed_bytes += len(value) if value is not None else 0
                prim.RemoveProperty(name)
            elif name.startswith("omni:") or name == "proxyPrim":
                prim.RemoveProperty(name)

        approximation = prim.GetAttribute("physics:approximation")
        if not approximation or approximation.Get() != "sdf":
            continue

        resolution_attr = prim.GetAttribute("physxSDFMeshCollision:sdfResolution")
        resolution = resolution_attr.Get() if resolution_attr else 64
        # Newton SDF textures are tiled in 8x8x8 blocks.
        resolution = min(max_sdf_resolution, max(8, round(int(resolution) / 8) * 8))
        prim.AddAppliedSchema("NewtonSDFCollisionAPI")
        prim.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(resolution)
        approximation.Set("none")
        prim.RemoveProperty("physxSDFMeshCollision:sdfResolution")
        converted_sdfs += 1

    for path in ("/Render", "/OmniverseKit_Persp"):
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)

    compacted_output = output.with_name(f"{output.stem}.compacted{output.suffix}")
    if not root_layer.Export(str(compacted_output)):
        raise RuntimeError(f"Unable to compact USD stage: {output}")
    compacted_output.replace(output)
    print(f"Converted {converted_sdfs} SDF colliders")
    print(f"Removed {removed_bytes / (1024 * 1024):.1f} MiB of PhysX cooked data")
    print(f"Wrote {output} ({output.stat().st_size / (1024 * 1024):.1f} MiB)")


def main() -> None:
    """Parse arguments and convert the clock scene."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source AnalogDigitalClock_SI_Units.usd file.")
    parser.add_argument("output", type=Path, help="Destination Newton USD file.")
    parser.add_argument(
        "--max-sdf-resolution",
        type=int,
        default=128,
        help="Maximum Newton SDF resolution; must be divisible by 8 (default: 128).",
    )
    args = parser.parse_args()
    if args.max_sdf_resolution <= 0 or args.max_sdf_resolution % 8:
        parser.error("--max-sdf-resolution must be positive and divisible by 8")
    convert(args.source, args.output, args.max_sdf_resolution)


if __name__ == "__main__":
    main()
