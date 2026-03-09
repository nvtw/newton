# Changelog

## [Unreleased]

### Added

- Add CUDA 3D texture-based SDF for faster mesh-mesh collision sampling, replacing NanoVDB volume lookups in the mesh-mesh collision pipeline with analytical trilinear gradient from 8 corner texel reads.
- Interactive example browser in the GL viewer with tree-view navigation and switch/reset support

### Changed

### Deprecated

### Removed

### Fixed

- Fix viewer crash with `imgui_bundle>=1.92.6` when editing colors by normalizing `color_edit3` input/output in `_edit_color3`
- Fix body `gravcomp` not being written to the MuJoCo spec, causing it to be absent from XML saved via `save_to_mjcf`

## [1.0.0] - YYYY-MM-DD

Initial public release.
