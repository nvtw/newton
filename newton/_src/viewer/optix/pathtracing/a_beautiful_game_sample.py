"""A Beautiful Game sample (Python equivalent of C# sample).

Load order matches C#:
1) glTF
2) OBJ fallback
3) Cornell Box fallback
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from .camera import Camera
    from .pathtracing_viewer import PathTracingViewer
    from .scene import Scene
except ImportError:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    stale = [name for name in sys.modules if name == "newton" or name.startswith("newton.")]
    for name in stale:
        del sys.modules[name]
    from newton._src.viewer.optix.pathtracing.camera import Camera
    from newton._src.viewer.optix.pathtracing.pathtracing_viewer import PathTracingViewer
    from newton._src.viewer.optix.pathtracing.scene import Scene


class ABeautifulGameSample:
    """Python sample matching MinimalDlssRR ABeautifulGameSample."""

    name = "A Beautiful Game"
    description = "Chess set scene with HDR environment lighting"

    def __init__(self):
        self.hdr_path: str | None = None

    @staticmethod
    def get_initial_camera(aspect_ratio: float) -> Camera:
        return Camera(
            position=(0.0, 0.0, 6.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov=45.0,
            aspect_ratio=aspect_ratio,
        )

    @staticmethod
    def _candidate_scene_paths() -> tuple[str, str]:
        gltf_path = r"C:\git\downloaded_resources\ABeautifulGame\glTF\ABeautifulGame.gltf"
        obj_path = r"C:\git\single-file-vulkan-pathtracing\assets\obj\ABeautifulGame.obj"
        return gltf_path, obj_path

    @staticmethod
    def find_hdr_file(base_directory: str | None = None) -> str | None:
        if base_directory is None:
            base_directory = str(Path(__file__).resolve().parent)
        search_paths = [
            str(Path(base_directory) / ".." / ".." / ".." / ".." / "media"),
            str(Path(base_directory) / "media"),
            r"C:\git\downloaded_resources",
        ]
        for search_path in search_paths:
            p = Path(search_path).resolve()
            if not p.exists():
                continue
            preferred = p / "environment.hdr"
            if preferred.exists():
                return str(preferred)
            hdr_files = sorted(p.glob("*.hdr"))
            if hdr_files:
                return str(hdr_files[0])
        return None

    def load(self, scene: Scene, base_directory: str | None = None) -> bool:
        gltf_path, obj_path = self._candidate_scene_paths()
        loaded = False

        if Path(gltf_path).exists():
            print(f"[ABeautifulGame] Loading glTF: {gltf_path}")
            loaded = scene.load_from_gltf(gltf_path)

        if not loaded and Path(obj_path).exists():
            print(f"[ABeautifulGame] Loading OBJ fallback: {obj_path}")
            loaded = scene.load_from_obj(obj_path)

        if not loaded:
            print("[ABeautifulGame] Using Cornell Box fallback")
            scene.create_cornell_box()
            loaded = True

        self.hdr_path = self.find_hdr_file(base_directory)
        if self.hdr_path:
            print(f"[ABeautifulGame] Found HDR environment: {self.hdr_path}")
        else:
            print("[ABeautifulGame] No HDR found, procedural sky settings will be used")
        return loaded

    def apply_environment_to_viewer(self, viewer: PathTracingViewer):
        """Apply environment setup to viewer.

        Note:
            Current Python OptiX sample uses procedural sky runtime parameters.
            This function keeps parity with C# fallback behavior and configures
            a sky setup. HDR texture binding hooks can be added in viewer runtime.
        """
        if self.hdr_path:
            viewer.set_environment_hdr(self.hdr_path)
            # Keep a neutral, low-noise sky baseline while HDR texture path
            # integration is added to the Python pipeline.
            viewer.sky_rgb_unit_conversion = (1.0 / 20000.0, 1.0 / 20000.0, 1.0 / 20000.0)
            viewer.sky_multiplier = 0.1
            viewer.sky_haze = 0.2
            viewer.sky_redblueshift = 0.05
            viewer.sky_saturation = 1.0
            viewer.sky_ground_color = (0.2, 0.2, 0.22)
            viewer.sky_night_color = (0.03, 0.04, 0.06)
            viewer.sky_sun_direction = (0.35, 0.9, 0.2)
            viewer.sky_sun_disk_intensity = 3.0
            viewer.sky_sun_glow_intensity = 1.5
        else:
            # Procedural fallback equivalent to C# sample fallback branch.
            viewer.sky_rgb_unit_conversion = (1.0, 1.0, 1.0)
            viewer.sky_multiplier = 1.0
            viewer.sky_haze = 0.0
            viewer.sky_redblueshift = 0.0
            viewer.sky_saturation = 1.0
            viewer.sky_ground_color = (0.2, 0.2, 0.3)
            viewer.sky_night_color = (0.5, 0.7, 1.0)
            viewer.sky_sun_direction = (0.0, 1.0, 0.0)
            viewer.sky_sun_disk_intensity = 0.0
            viewer.sky_sun_glow_intensity = 0.0
        viewer.sky_y_is_up = 1


def build_a_beautiful_game(scene: Scene) -> None:
    sample = ABeautifulGameSample()
    sample.load(scene)
    # Keep sample around on scene for viewer setup phase.
    scene._abg_sample = sample


def create_a_beautiful_game_viewer(width: int = 1280, height: int = 720) -> PathTracingViewer:
    sample = ABeautifulGameSample()
    # Match C# startup behavior: environment choice is known before rendering starts.
    sample.hdr_path = sample.find_hdr_file()
    camera = sample.get_initial_camera(width / height)
    viewer = PathTracingViewer(
        width=width,
        height=height,
        scene_setup=lambda s: sample.load(s),
        camera=camera,
        accumulate_samples=False,
        samples_per_frame=1,
        max_bounces=4,
        direct_light_samples=1,
        use_halton_jitter=True,
    )
    sample.apply_environment_to_viewer(viewer)
    return viewer


def main() -> int:
    print("=" * 72)
    print("A Beautiful Game (Python/OptiX)")
    print("=" * 72)
    viewer = create_a_beautiful_game_viewer()
    if not viewer.build():
        print("Failed to build A Beautiful Game sample")
        return 1
    for _ in range(8):
        viewer.render()
    out = viewer.get_output()
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{float(np.min(out)):.4f}, {float(np.max(out)):.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
