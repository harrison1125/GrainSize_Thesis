"""
Input paths for XRD and XRF analysis workflows

This module centralizes all filepaths and runtime parameters used across the
analysis pipeline. To move to a new machine or dataset, change PROJECT_ROOT
only. All other paths are resolved automatically.

Separation of concerns
-----------------------
Manual workflows (cwt_statistics.py, metricplot.py, combinatorial_runner.py)
import paths from this module. Automated workflows that stream new results do
not require explicitly defined filepaths and do not use this module.

Layout convention (enforced by auto-discovery)
-----------------------------------------------
PROJECT_ROOT/
    Config/
        <name>.poni          # exactly one pyFAI calibration file
        <name>.cfg           # exactly one PYMCA configuration file
        SampleData*/         # one subdirectory per scan row

Attributes
----------
PROJECT_ROOT : Path
    Root of the project directory. Only value that needs editing per machine.
root_dir : Path
    Directory walked recursively by cwt_statistics.py for TIFF images.
    Currently set to PROJECT_ROOT.
poni_file : Path
    The single .poni file found in Config/. Raises FileNotFoundError if
    zero or more than one are present.
config_path : Path
    The single .cfg file found in Config/. Same error behaviour as poni_file.
combinatorial_root : Path
    Parent directory containing the per-row scan subdirectories.
combinatorial_dirs : list of Path
    All SampleData* subdirectories under Config/, sorted lexicographically
    (matches R-number order for standard JHAMAC naming).
combinatorial_output : Path
    Destination path for the combined_ring_metrics.png output figure.
tth_ranges : list of tuple of float
    2theta ROI windows (degrees) passed to cwt_statistics().
    Each entry (tth_lo, tth_hi) defines one ring to analyze. The order here
    determines the ring{i}_ column prefix in all output CSVs and the column
    order in per-image PNGs. Add or remove entries freely; downstream scripts
    derive their RINGS list from this variable automatically.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# User configuration — only this line changes between machines
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/Users/hpark108/Projects/GrainSize_Qualitative")


# ---------------------------------------------------------------------------
# Derived paths — do not edit; resolved automatically from PROJECT_ROOT
# ---------------------------------------------------------------------------

_CONFIG_DIR = PROJECT_ROOT / "Config"

root_dir = PROJECT_ROOT

# pyFAI calibration file: the single .poni file found in Config/.
# Raises FileNotFoundError with a clear message if none or multiple are found.
_poni_candidates = list(_CONFIG_DIR.glob("*.poni"))
if len(_poni_candidates) == 0:
    raise FileNotFoundError(f"No .poni file found in {_CONFIG_DIR}")
if len(_poni_candidates) > 1:
    raise FileNotFoundError(
        f"Multiple .poni files found in {_CONFIG_DIR}: "
        f"{[p.name for p in _poni_candidates]}. "
        "Remove duplicates or set poni_file manually."
    )
poni_file = str(_poni_candidates[0])

# PYMCA configuration file: the single .cfg file found in Config/.
_cfg_candidates = list(_CONFIG_DIR.glob("*.cfg"))
if len(_cfg_candidates) == 0:
    raise FileNotFoundError(f"No .cfg file found in {_CONFIG_DIR}")
if len(_cfg_candidates) > 1:
    raise FileNotFoundError(
        f"Multiple .cfg files found in {_CONFIG_DIR}: "
        f"{[p.name for p in _cfg_candidates]}. "
        "Remove duplicates or set config_path manually."
    )
config_path = str(_cfg_candidates[0])

# Combinatorial scan subdirectories: all SampleData* folders under Config/,
# sorted lexicographically (which matches R-number order for JHAMAC naming).
combinatorial_root = _CONFIG_DIR
combinatorial_dirs = sorted(_CONFIG_DIR.glob("SampleData*"))
combinatorial_output = _CONFIG_DIR / "combined_ring_metrics.png"


# ---------------------------------------------------------------------------
# 2theta ROI windows
# ---------------------------------------------------------------------------

tth_ranges = [(12, 14), (15, 16), (20, 21)]

#tth_ranges = [
#    (13.5, 14.5),
#    (15.5, 17.0),
#    (22.5, 24.5),
#    # Add additional (tth_lo, tth_hi) pairs here as needed.
#]
