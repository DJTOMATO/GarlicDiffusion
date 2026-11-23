---
name: Hardware Benchmark Report
about: Submit verified generation timings from your Base PS4, PS4 Slim, or PS4 Pro
title: "[Benchmark] "
labels: benchmark
---

### Hardware Details
- **Console Model:** (PS4 Fat CUH-1xxx / PS4 Slim CUH-2xxx / PS4 Pro CUH-7xxx)
- **Firmware:** (e.g. 9.00 / 11.00)
- **Storage:** (Internal HDD / Internal SSD / USB SSD)
- **GarlicDiffusion Version:** (e.g. v1.10)

### Measured Results
Copy the `stats` and `upscale` lines directly from `/data/garlic_diffusion_log.txt` (and `/status` `bench_mode_tf` if available):

```text
[gdiff ...] console: ...
[gdiff ...] stats ...
[gdiff ...] upscale RealESRGAN Anime v3: ...
```

### Notes (Fan Noise / Stability)
How did the fan noise and thermals feel in Silent / Quiet / Balanced mode?
