# PhoenX fluid

- [x] Port the supplementary 3D moments with four conserved fields and five FP16 traceless-stress fields.
- [x] Fuse the 3D predictor and corrector around K² threads per element and shared-memory transposes.
- [x] Rasterize every Newton shape type to static solution-point volume fractions.
- [x] Export a compact sparse brick stream and provide optional NanoVDB conversion.
- [x] Add OptiX volume transport in `otk-pyoptix`; use `vk_denoise_dlssrr` only for DLSS-RR integration hints.
- [ ] Match the KPM-FR teaser density, coherent filament detail, and blue/orange transfer quality at Full HD in real time.
