Vdiup project. 
PI: Nils Haentjens (nils.haentjens@maine.edu).
The main file is Updated_Kd_Compute_Code.py. It downloads Argo float Ed profiles from the Ifremer GDAC directly,quality controls them and computes hyperspectral Kds. It outputs Ed files as well as Seabass-compliant sb files. 
Previous version (Test_explore.py) downloads raw data from the GDAC and use calibration files that must be obtained by Argo PIs individually. Now obsolete.
Matchup_PACE_data.py provides matchup code with PACE OCI L2 Kd granule following standard matchups requirements. 
A specific QC was developped for the hyperspectral floats (Organelli_QC_Shapiro.py). Details on the implementation can be found in the Seabass file documentation as well as in Andres& Begouen Demeaux et al.,2026.
