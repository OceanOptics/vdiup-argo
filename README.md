PVST-VDIUP project 
==================

Run `uv sync` to install the `vdiup_argo` package from `src/` before running the scripts.

The main processing file is `Updated_KdCompute_Code.py`. It reads calibrated downwelling irradiance Ed profiles directly from the Coriolis auxiliary NetCDF files on the Argo GDAC, applies a wavelength-resolved quality control, and computes hyperspectral Kd(λ) with bootstrapped uncertainties. Outputs include Ed profiles and SeaBASS-compliant Kd files.

The `scripts/download.py` script can be used to efficiently download profiles from argo floats of interest.

The `scripts/matchups.py` script will download Kd from PACE-OCI L2 and match it with quality-controlled Kd profiles following the Bailey & Werdell (2006) protocol.

Other scripts are available in the `scripts/` folder, to prepare and check files for SeaBASS submission (`prepare_seabass.py`, `check_seabass.py`), to plot Kds (`scripts/report.py`), and to export the Kds (`scripts/export.py`).

A hyperspectral-specific QC was developed for the RAMSES-equipped BGC-Argo floats (`src/vdiup_argo/process/qc.py`). Implementation details are available in the SeaBASS file documentation and in Begouen Demeaux & Andres et al. (2026).

Previous versions of the code are available in the `obsolete/` folder, including `Test_explore.py`, which downloaded raw data from the GDAC and applied manufacturer calibration files obtained individually from Argo PIs.
