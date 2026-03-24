"""
Input paths for XRD and XRF analysis workflows

This module centralizes the filepaths used in automated XRD analysis workflows
for further use. This discretizes input values from further analysis scripts,
which increases cleanliness down the line as we separate manual vs automated
workflows (manual workflows use this input folder, automated workflows naturally
stream new results without requiring explicitly defining filepaths).

Attributes
----------
root_dir : str
    Path to directory containing all files to analyze.
poni_file : str
    Path to calibration file for XRD. Generated through PyFAI.
config_path : str
    Path to configuration file for XRF. Generated through PYMCA.
tth_ranges : list of tuple of float
    2theta ROI windows (degrees) passed to azimuthal_ring_statistics().
    Each entry (tth_lo, tth_hi) defines one ring to analyze. The order
    here determines the ring{i}_ column prefix in all output CSVs and the
    column order in per-image PNGs. Add or remove entries freely; downstream
    scripts (metricplot.py, temp_runner.py) read the RINGS config list, which
    should be updated to match.
"""
root_dir    = "/Users/hpark108/Projects/GrainSize_Qualitative"
poni_file   = '/Users/hpark108/Desktop/Immediate/updated xrf.poni'
config_path = '/Users/hpark108/Desktop/Immediate/aimdpaper.cfg'

tth_ranges = [(12,14),(15,16),(20,21)]

#
#tth_ranges = [
#    (13.5,14.5),
#    (15.5, 17.0),
#    (22.5, 24.5),
    # Add additional (tth_lo, tth_hi) pairs here as needed.
    # Remember to update the RINGS list in metricplot.py and temp_runner.py
    # to match (same count, same index order).
#]
