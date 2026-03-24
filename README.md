# GrainSize_Thesis

Analyzes radial variations in XRD intensity as a function of azimuthal angle
(γ) from synchrotron/lab 2D diffraction data. Outputs azimuthal ring metrics
— texture index, CWT peak count, Fourier ODF coefficients, Warren grain proxy,
and more — for each scan point in a combinatorial library.

## Setup

```bash
pip install -r requirements.txt
```

Edit `Inputs.py` to point to your data, calibration file, and config. All
other scripts read paths from there; no other files need to change when
moving to a new machine.

## Input file formats

- Diffraction images: `.tif` / `.tiff`
- Calibration: `.poni` (generated with `pyFAI-calib2`)
- XRF config: `.cfg` (generated with PyMCA)

## Execution order

```
cwt_statistics.py          # integrate images, compute ring metrics, write CSVs + PNGs
metricplot.py              # per-subfolder metric-vs-scan-point figures
combinatorial_runner.py    # multi-row stacked position figure across R1–R6
```

## File overview

| File | Role |
|---|---|
| `Inputs.py` | All paths and 2θ ROI windows — **edit this file only** for a new machine |
| `ring_metrics_cwt.py` | Core metric functions (`compute_ring_statistics`, `plot_ring_metrics`) |
| `cwt_statistics.py` | Batch runner: walks `root_dir`, integrates TIFFs, writes CSVs + diagnostic PNGs |
| `metricplot.py` | Reads existing `ring_metrics_summary.csv` files and plots metrics vs scan point |
| `combinatorial_runner.py` | Concatenates metrics from multiple scan rows into a single position-stacked figure |

## 2θ ROI configuration

Edit `tth_ranges` in `Inputs.py`. All downstream scripts derive their ring
list from this variable automatically.

```python
tth_ranges = [(12, 14), (15, 16), (20, 21)]
```
