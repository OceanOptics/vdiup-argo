PVST-VDIUP project. PI: Nils Haentjens (nils.haentjens@maine.edu).

The main processing file is Updated_KdCompute_Code.py. It reads calibrated downwelling irradiance Ed profiles directly from the Coriolis auxiliary NetCDF files on the Argo GDAC, applies a wavelength-resolved quality control, and computes hyperspectral Kd(λ) with bootstrapped uncertainties. Outputs include Ed profiles and SeaBASS-compliant Kd files.

The previous version (Test_explore.py) downloaded raw data from the GDAC and applied manufacturer calibration files obtained individually from Argo PIs. Now obsolete.

Matchup_PACE_data.py provides matchup code between the quality-controlled Kd profiles and PACE OCI L2 Kd granules, following the Bailey & Werdell (2006) protocol.

A hyperspectral-specific QC was developed for the RAMSES-equipped BGC-Argo floats (Organelli_QC_Shapiro.py). Implementation details are available in the SeaBASS file documentation and in Begouen Demeaux & Andres et al. (2026).
