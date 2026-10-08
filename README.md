PVST-VDIUP project 
==================

PI: Nils Haentjens

The main processing file is Updated_KdCompute_Code.py. It reads calibrated downwelling irradiance Ed profiles directly from the Coriolis auxiliary NetCDF files on the Argo GDAC, applies a wavelength-resolved quality control, and computes hyperspectral Kd(λ) with bootstrapped uncertainties. Outputs include Ed profiles and SeaBASS-compliant Kd files.

The `download.py` script can be used to efficiently download profiles from argo floats of interest.

Matchup_PACE_data.py provides matchup code between the quality-controlled Kd profiles and PACE OCI L2 Kd granules, following the Bailey & Werdell (2006) protocol.

A hyperspectral-specific QC was developed for the RAMSES-equipped BGC-Argo floats (Organelli_QC_Shapiro.py). Implementation details are available in the SeaBASS file documentation and in Begouen Demeaux & Andres et al. (2026).

Previous versions of the code are available in the obsolete/ folder, including Test_explore.py, which downloaded raw data from the GDAC and applied manufacturer calibration files obtained individually from Argo PIs. Now obsolete.
