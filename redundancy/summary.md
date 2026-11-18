# Latent redundancy: 2,000 clips from `gate1_data`

Consecutive-frame MSD per clip: median 0.1231, 90th percentile 0.2201; mean frame energy 0.8528. Most-moving third: clips with MSD >= 0.1501.

| C | top-eigenvalue share (all / most-moving) | linear SNR dB | average + linear decoder SNR dB | static SNR dB | Gate 0 best coded dB | Gate 0 learned compressor dB |
|---|---|---|---|---|---|---|
| 2 | 96.1% / 93.7% | 14.11 / 12.02 | 14.11 / 12.02 | 14.11 / 12.02 | nan | nan |
| 3 | 94.0% / 90.3% | 12.24 / 10.15 | 12.24 / 10.15 | 12.23 / 10.14 | nan | nan |
| 4 | 92.5% / 87.9% | 11.28 / 9.18 | 11.28 / 9.18 | 11.26 / 9.17 | 10.72 | 9.58 |
| 5 | 91.4% / 86.1% | 10.66 / 8.57 | 10.66 / 8.57 | 10.63 / 8.55 | nan | nan |
| 6 | 90.4% / 84.5% | 10.19 / 8.10 | 10.19 / 8.10 | 10.16 / 8.08 | nan | nan |
| 8 | 88.8% / 82.0% | 9.52 / 7.46 | 9.52 / 7.45 | 9.48 / 7.42 | 8.83 | 8.03 |
| 10 | 87.9% / 80.6% | 9.18 / 7.12 | 9.18 / 7.12 | 9.14 / 7.10 | nan | nan |
| 16 | 85.5% / 76.9% | 8.38 / 6.36 | 8.38 / 6.35 | 8.31 / 6.31 | nan | nan |

SNR columns: all clips / most-moving third. Linear SNR = best single linear code shared by all positions with the best linear decoder: what a fixed code gets with no learned prior.
