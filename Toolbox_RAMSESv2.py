# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:13:20 2023

@author: louan
"""

# Import packages
import numpy as np
import pandas as pd
import xarray as xr
import time
import os
import glob

# Plotting librairies
# import cmocean
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt


#%% FUNCTIONS TO CONVERT RAW RAMSES DATA IN PHYSICAL UNITS (uW/cm2/nm)

def ra_single(x,B0,B1,S,B0_Dark,B1_Dark) :
    """
    Function to apply calibration equation and coefficient from the coefficient table which is dependant from wavelength
    to a vector of ramses data at one depth. (Associated with format_ramses)
    """
    
    t = x.Int_Time.iloc[0] # integration time
    offset = x.Dark_count.iloc[0] # dark
    I = x.iloc[0,4:].to_numpy() # radiance/irradiance
    
    # Etape 1 : Normalisation
    M = I / 65535
    
    # Etape 2 : Background Substraction
    B = B0 + t * B1 / 8192
    C = M - B
    
    offset = offset / 65535
    offset = offset - B0_Dark - t * B1_Dark / 8192
    
    D = C - offset
    
    # Etape 3 : Normalisation du temps d'intégration
    E = D * 8192 / t
    
    # Conversion en uW/cm2/nm
    E = E / 10
    
    # Etape 4 : Sensibilité
    return E / S

def ra_single_buoy_mode(x,B0,B1,S,B0_Dark,B1_Dark) :
    """
    Function to apply calibration equation and coefficient from the coefficient table which is dependant from wavelength
    to a vector of ramses data at one depth. (Associated with format_ramses_buoy_mode)
    """
    
    t = x.Int_Time.iloc[0] # integration time
    offset = x.Dark_count.iloc[0] # dark
    I = x.iloc[0,5:].to_numpy()

    # Etape 1 : Normalisation
    M = I / 65535

    # Etape 2 : Background Substraction
    B = B0 + t * B1 / 8192
    C = M - B
    
    offset = offset / 65535
    offset = offset - B0_Dark - t * B1_Dark / 8192
    
    D = C - offset
    
    # Etape 3 : Normalisation du temps d'intégration
    E = D * 8192 / t
    
    # Conversion en uW/cm2/nm
    E = E / 10
    
    # Etape 4 : Sensibilité
    return E / S

def ra_single_RData(x,B0,B1,S,B0_Dark,B1_Dark) :
    """
    Function to apply calibration equation and coefficient from the coefficient table which is dependant from wavelength
    to a vector of ramses data at one depth. (Associated with format_ramses_RData)
    """

    t = x.ramses_int_time.iloc[0] # integration time
    offset = x.ramses_dark_count.iloc[0] # dark
    I = x.iloc[0,11:81].to_numpy() # radiance/irradiance

    # Etape 1 : Normalisation
    M = I / 65535

    # Etape 2 : Background Substraction
    B = B0 + t * B1 / 8192
    C = M - B

    offset = offset / 65535
    offset = offset - B0_Dark - t * B1_Dark / 8192

    D = C - offset

    # Etape 3 : Normalisation du temps d'intégration
    E = D * 8192 / t

    # Conversion en uW/cm2/nm
    E = E / 10

    # Etape 4 : Sensibilité
    return E / S

def ra_single_RamsesData(x,B0,B1,S,B0_Dark,B1_Dark) :
    """
    Function to apply calibration equation and coefficient from the coefficient table which is dependant from wavelength
    to a vector of ramses data at one depth. (Associated with format_ramses_RamsesData)
    """
    
    t = x.Int_Time.iloc[0] # integration time
    offset = x.Dark_count.iloc[0] # dark
    I = x.iloc[0,3:].to_numpy() # radiance/irradiance
    
    # Etape 1 : Normalisation
    M = I / 65535
    
    # Etape 2 : Background Substraction
    B = B0 + t * B1 / 8192
    C = M - B
    
    offset = offset / 65535
    offset = offset - B0_Dark - t * B1_Dark / 8192
    
    D = C - offset
    
    # Etape 3 : Normalisation du temps d'intégration
    E = D * 8192 / t
    
    # Conversion en uW/cm2/nm
    E = E / 10
    
    # Etape 4 : Sensibilité
    return E / S

def format_ramses_argopy(ds,metaname,calEd_name,calLu_name,Ed_n_prof,Lu_n_prof,PixelBinning='auto', PixelStop='auto'):
    """
    Function to obtain 2 Table of Ed and Lu in physics units for ONE profile of a BGC-Argo float netcdf Sfile (from GDAC/aux/).
    NB : - for WMO = 4903660 : PixelBinning=1 from cycle 13.
         - for WMO = 1902578 : PixelStop=140 until cycle 4 and no Ed/Lu before cycle 5.
         - for WMO = 2903787 : No Ed/Lu before cycle 1.
         - for WMO = 1902685 : No Lu cycle 19 & No Ed cycle 23 & issue on Post_pressure_Ramses=-100dbar.
         - for WMO = 6990503 : No Ed/Lu before cyle 7.

    Parameters
    ----------
    ds : dataframe pandas argopy
        dataframe argopy of the float's profile in counts (gdac/aux/coriolis/)
    metaname : str
        path of the netcdf fileof the float's metadata (gdac/aux/coriolis/)
    calEd_name : str
        path of the txt file with the calibration coefficient of Ed sensor for this float.
        (supposed to be in the metadata, for now it is in Edouard computer)
    calLu_name : str
        path of the txt file with the calibration coefficient of Lu sensor for this float.
        (supposed to be in the metadata, for now it is in Edouard computer)
    PixelBinning : int
        security to allow user to fixe manually PixelBinning in case he knows that it is the wrong one in the config meta file
        (ex : wmo=4903660_013). Default='auto' --> means that we keep the PixelBinning given by the meta file.

    Returns
    -------
    Ed_physic_profile : pandas.DataFrame
        Table of Ed values in W.m-2.nm-1 with dimensions : depth x wavelength
    Lu_physic_profile : pandas.DataFrame
        Table of Lu values in W.m-2.nm-1.sr-1 with dimensions : depth x wavelength

    """
    
    """ RAWDATA : Find data in counts(depthxwavelength), integration time(depth), dark in counts(depth) and depth """
    # open raw data in counts for one profile
    #file = xr.open_dataset(filename)
    
    ## Extract Ed data into a table with : depth, int_time, dark_count, raw_count_lambda1, ..., raw_count_lambda2
    Ed_raw_profile = pd.DataFrame({ 'Pres_float':ds.PRES.sel(N_PROF=Ed_n_prof).values,
                                   'Post_Pres':ds.RADIOMETER_DOWN_IRR_POST_PRES.sel(N_PROF=Ed_n_prof).values,
                                        'Int_Time':ds.RADIOMETER_DOWN_IRR_INTEGRATION_TIME.sel(N_PROF=Ed_n_prof).values,
                                        'Dark_count':ds.RADIOMETER_DOWN_IRR_DARK_AVERAGE.sel(N_PROF=Ed_n_prof).values })
    Ed_raw_profile = pd.concat([Ed_raw_profile, pd.DataFrame(ds.RAW_DOWNWELLING_IRRADIANCE.sel(N_PROF=Ed_n_prof).values)], axis=1)
       
    
    ## Extract Lu data into a table with : depth, int_time, dark_count, raw_count_lambda1, ..., raw_count_lambda2
    Lu_raw_profile = pd.DataFrame({ 'Pres_float':ds.PRES.sel(N_PROF=Lu_n_prof).values,
                                   'Post_Pres':ds.RADIOMETER_UP_RAD_POST_PRES.sel(N_PROF=Lu_n_prof).values,
                                        'Int_Time':ds.RADIOMETER_UP_RAD_INTEGRATION_TIME.sel(N_PROF=Lu_n_prof).values,
                                        'Dark_count':ds.RADIOMETER_UP_RAD_DARK_AVERAGE.sel(N_PROF=Lu_n_prof).values })
    Lu_raw_profile = pd.concat([Lu_raw_profile, pd.DataFrame(ds.RAW_UPWELLING_RADIANCE.sel(N_PROF=Lu_n_prof).values)], axis=1)
    
    
    
    """ METADONNEES : to find Pixels configuration (PixelStart, stop and Binning) """
    # open meta data 
    meta = xr.open_dataset(metaname)
    
    # Find Config parameters index of RAMSES 1 and 2
    index_Arc = np.where(meta.LAUNCH_CONFIG_PARAMETER_NAME.values==b'CONFIG_RamsesArcOutputPixelBegin_NUMBER                                                                                         ')[0][0]
    index_Acc = np.where(meta.LAUNCH_CONFIG_PARAMETER_NAME.values==b'CONFIG_RamsesAccOutputPixelBegin_NUMBER                                                                                         ')[0][0]
    
    # Find Config parameters thanks to the index
    [PixelStart_Acc, PixelStop_Acc, PixelBinning_Acc] = meta.LAUNCH_CONFIG_PARAMETER_VALUE.values[index_Acc:index_Acc+3]
    [PixelStart_Arc, PixelStop_Arc, PixelBinning_Arc] = meta.LAUNCH_CONFIG_PARAMETER_VALUE.values[index_Arc:index_Arc+3]
    
    if PixelBinning != 'auto' :
        PixelBinning_Acc, PixelBinning_Arc = PixelBinning, PixelBinning
        
    if PixelStop != 'auto' :
        PixelStop_Acc, PixelStop_Arc = PixelStop, PixelStop
        

    """ CALIBRATION FILES : to find equation and coefficients to translate counts into physics units. """
    # open calibration files
    cal_Ed = pd.read_table(calEd_name, sep='\t')
    cal_Lu = pd.read_table(calLu_name, sep='\t')
    
    # correct the "+NAN" into NaN numpy
    cal_Ed[ cal_Ed=="+NAN" ] = np.nan
    cal_Lu[ cal_Lu=="+NAN" ] = np.nan
    
    # convert everything into numerical type values
    cal_Ed.S =pd.to_numeric(cal_Ed.S)
    cal_Lu.S =pd.to_numeric(cal_Lu.S)
    
    # Rearange Ed Calibration parameter depending on profile configuration (found in metadata but for now in Edouard files)
    Ed_InWater=True
    
    # Averaging Ed calibrations factor PixelBinningxPixelBinning
    Ed_sq = np.arange(PixelStart_Acc,PixelStop_Acc,PixelBinning_Acc)
    
    # ajout d'une condition pour éviter les problèmes de shape dans le cas PixelBinning=1
    if PixelBinning==1 :
        Ed_sq = np.arange(PixelStart_Acc,PixelStop_Acc+1,PixelBinning_Acc)
        
    Ed_wave = pd.array([ np.mean(cal_Ed.Wave[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])
    Ed_B0   = pd.array([ np.mean(cal_Ed.B0[   (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])
    Ed_B1   = pd.array([ np.mean(cal_Ed.B1[   (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])
    
    if Ed_InWater :
        Ed_S = pd.array([ np.mean(cal_Ed.S[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])
    else :
        Ed_S = pd.array([ np.mean(cal_Ed.Sair[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])

    # Averaging Ed dark coefficient
    Ed_B0_Dark = cal_Ed.B0[ cal_Ed.Wave==-1 ].mean()
    Ed_B1_Dark = cal_Ed.B1[ cal_Ed.Wave==-1 ].mean()
    Ed_B1_Dark
    
    # Rearange Lu Calibration parameter depending on profile configuration (found in metadata but for now in Edouard files)
    Lu_InWater=True
    
    # Averaging Lu calibrations factor 2by2
    Lu_sq = np.arange(PixelStart_Arc,PixelStop_Arc,PixelBinning_Arc)
    
    # ajout d'une condition pour éviter les problèmes de shape dans le cas PixelBinning=1
    if PixelBinning==1 :
        Lu_sq = np.arange(PixelStart_Acc,PixelStop_Acc+1,PixelBinning_Acc)
    
    Lu_wave = pd.array([ np.mean(cal_Lu.Wave[ (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])
    Lu_B0   = pd.array([ np.mean(cal_Lu.B0[   (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])
    Lu_B1   = pd.array([ np.mean(cal_Lu.B1[   (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])
    
    if Lu_InWater :
        Lu_S = pd.array([ np.mean(cal_Lu.S[ (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])
    else :
        Lu_S = pd.array([ np.mean(cal_Lu.Sair[ (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])

    # Averaging Lu dark coefficient
    Lu_B0_Dark = cal_Lu.B0[ cal_Lu.Wave==-1 ].mean()
    Lu_B1_Dark = cal_Lu.B1[ cal_Lu.Wave==-1 ].mean()
    
    
    """ APPLY CALIBRATION """
    # Create the global table to save radiometric data of the profile into physics units
    Ed_physic_profile = pd.DataFrame(columns=np.round(Ed_wave))
    Ed_physic_profile.insert(0,'Post_Pres', Ed_raw_profile.Post_Pres[~Ed_raw_profile.Post_Pres.isna()])
    Ed_physic_profile.insert(0,'Pres_float', Ed_raw_profile.Pres_float[~Ed_raw_profile.Pres_float.isna()])
    
    # Fill the table with counts data converted
    for z in Ed_raw_profile.Pres_float[~Ed_raw_profile.Pres_float.isna()] :
    
        # Récupération du spectre à la profondeur z
        Ed_z_counts = Ed_raw_profile[ Ed_raw_profile.Pres_float==z ] 
        
        # Apply ra_single function to translate into physic units   
        Ed_z_physic = ra_single(Ed_z_counts,Ed_B0,Ed_B1,Ed_S,Ed_B0_Dark,Ed_B1_Dark)
        
        # Add into the global table
        Ed_physic_profile.loc[Ed_physic_profile.Pres_float==z, Ed_physic_profile.columns[2:]] = Ed_z_physic.reshape((1,-1))
        

    # Create the global table to save radiometric data into physics units
    Lu_physic_profile = pd.DataFrame(columns=np.round(Lu_wave))
    Lu_physic_profile.insert(0,'Post_Pres', Lu_raw_profile.Post_Pres[~Lu_raw_profile.Post_Pres.isna()])
    Lu_physic_profile.insert(0,'Pres_float', Lu_raw_profile.Pres_float[~Lu_raw_profile.Pres_float.isna()])
    
    # Fill the table with counts data converted
    for z in Lu_raw_profile.Pres_float[~Lu_raw_profile.Pres_float.isna()] :
    
        # Récupération du spectre à la profondeur z
        Lu_z_counts = Lu_raw_profile[ Lu_raw_profile.Pres_float==z ] 
        
        # Apply ra_single function to translate into physic units
        Lu_z_physic = ra_single(Lu_z_counts,Lu_B0,Lu_B1,Lu_S,Lu_B0_Dark,Lu_B1_Dark)
        
        # Add into the global table
        Lu_physic_profile.loc[Lu_physic_profile.Pres_float==z, Lu_physic_profile.columns[2:]] = Lu_z_physic.reshape((1,-1))
    
    
    # remove pres_float which will be added in a second step outside this functiun (here we have only ramses data)
    Ed_physic_profile = Ed_physic_profile.drop(columns='Pres_float')
    Lu_physic_profile = Lu_physic_profile.drop(columns='Pres_float')
    
    return Ed_physic_profile, Lu_physic_profile

def format_ramses(filename,metaname,calEd_name,calLu_name,Ed_n_prof,Lu_n_prof,PixelBinning='auto', PixelStop='auto'):
    """
    Function to obtain 2 Table of Ed and Lu in physics units for ONE profile of a BGC-Argo float netcdf Sfile (from GDAC/aux/).
    NB : - for WMO = 4903660 : PixelBinning=1 from cycle 13.
         - for WMO = 1902578 : PixelStop=140 until cycle 4 and no Ed/Lu before cycle 5.
         - for WMO = 2903787 : No Ed/Lu before cycle 1.
         - for WMO = 1902685 : No Lu cycle 19 & No Ed cycle 23 & issue on Post_pressure_Ramses=-100dbar.
         - for WMO = 6990503 : No Ed/Lu before cyle 7.

    Parameters
    ----------
    filename : str
        path of the netcdf file of the float's profile in counts (gdac/aux/coriolis/)
    metaname : str
        path of the netcdf fileof the float's metadata (gdac/aux/coriolis/)
    calEd_name : str
        path of the txt file with the calibration coefficient of Ed sensor for this float.
        (supposed to be in the metadata, for now it is in Edouard computer)
    calLu_name : str
        path of the txt file with the calibration coefficient of Lu sensor for this float.
        (supposed to be in the metadata, for now it is in Edouard computer)
    PixelBinning : int
        security to allow user to fixe manually PixelBinning in case he knows that it is the wrong one in the config meta file
        (ex : wmo=4903660_013). Default='auto' --> means that we keep the PixelBinning given by the meta file.

    Returns
    -------
    Ed_physic_profile : pandas.DataFrame
        Table of Ed values in W.m-2.nm-1 with dimensions : depth x wavelength
    Lu_physic_profile : pandas.DataFrame
        Table of Lu values in W.m-2.nm-1.sr-1 with dimensions : depth x wavelength

    """
    
    """ RAWDATA : Find data in counts(depthxwavelength), integration time(depth), dark in counts(depth) and depth """
    # open raw data in counts for one profile
    file = xr.open_dataset(filename)
    
    ## Extract Ed data into a table with : depth, int_time, dark_count, raw_count_lambda1, ..., raw_count_lambda2
    Ed_raw_profile = pd.DataFrame({ 'Pres_float':file.PRES.sel(N_PROF=Ed_n_prof).values,
                                   'Post_Pres':file.RADIOMETER_DOWN_IRR_POST_PRES.sel(N_PROF=Ed_n_prof).values,
                                        'Int_Time':file.RADIOMETER_DOWN_IRR_INTEGRATION_TIME.sel(N_PROF=Ed_n_prof).values,
                                        'Dark_count':file.RADIOMETER_DOWN_IRR_DARK_AVERAGE.sel(N_PROF=Ed_n_prof).values })
    Ed_raw_profile = pd.concat([Ed_raw_profile, pd.DataFrame(file.RAW_DOWNWELLING_IRRADIANCE.sel(N_PROF=Ed_n_prof).values)], axis=1)
       
    
    ## Extract Lu data into a table with : depth, int_time, dark_count, raw_count_lambda1, ..., raw_count_lambda2
    Lu_raw_profile = pd.DataFrame({ 'Pres_float':file.PRES.sel(N_PROF=Lu_n_prof).values,
                                   'Post_Pres':file.RADIOMETER_UP_RAD_POST_PRES.sel(N_PROF=Lu_n_prof).values,
                                        'Int_Time':file.RADIOMETER_UP_RAD_INTEGRATION_TIME.sel(N_PROF=Lu_n_prof).values,
                                        'Dark_count':file.RADIOMETER_UP_RAD_DARK_AVERAGE.sel(N_PROF=Lu_n_prof).values })
    Lu_raw_profile = pd.concat([Lu_raw_profile, pd.DataFrame(file.RAW_UPWELLING_RADIANCE.sel(N_PROF=Lu_n_prof).values)], axis=1)
    
    
    
    """ METADONNEES : to find Pixels configuration (PixelStart, stop and Binning) """
    # open meta data 
    meta = xr.open_dataset(metaname)
    
    # Find Config parameters index of RAMSES 1 and 2
    index_Arc = np.where(meta.LAUNCH_CONFIG_PARAMETER_NAME.values==b'CONFIG_RamsesArcOutputPixelBegin_NUMBER                                                                                         ')[0][0]
    index_Acc = np.where(meta.LAUNCH_CONFIG_PARAMETER_NAME.values==b'CONFIG_RamsesAccOutputPixelBegin_NUMBER                                                                                         ')[0][0]
    
    # Find Config parameters thanks to the index
    [PixelStart_Acc, PixelStop_Acc, PixelBinning_Acc] = meta.LAUNCH_CONFIG_PARAMETER_VALUE.values[index_Acc:index_Acc+3]
    [PixelStart_Arc, PixelStop_Arc, PixelBinning_Arc] = meta.LAUNCH_CONFIG_PARAMETER_VALUE.values[index_Arc:index_Arc+3]
    
    if PixelBinning != 'auto' :
        PixelBinning_Acc, PixelBinning_Arc = PixelBinning, PixelBinning
        
    if PixelStop != 'auto' :
        PixelStop_Acc, PixelStop_Arc = PixelStop, PixelStop
        

    """ CALIBRATION FILES : to find equation and coefficients to translate counts into physics units. """
    # open calibration files
    cal_Ed = pd.read_table(calEd_name, sep='\t')
    cal_Lu = pd.read_table(calLu_name, sep='\t')
    
    # correct the "+NAN" into NaN numpy
    cal_Ed[ cal_Ed=="+NAN" ] = np.nan
    cal_Lu[ cal_Lu=="+NAN" ] = np.nan
    
    # convert everything into numerical type values
    cal_Ed.S =pd.to_numeric(cal_Ed.S)
    cal_Lu.S =pd.to_numeric(cal_Lu.S)
    
    # Rearange Ed Calibration parameter depending on profile configuration (found in metadata but for now in Edouard files)
    Ed_InWater=True
    
    # Averaging Ed calibrations factor PixelBinningxPixelBinning
    Ed_sq = np.arange(PixelStart_Acc,PixelStop_Acc,PixelBinning_Acc)
    
    # ajout d'une condition pour éviter les problèmes de shape dans le cas PixelBinning=1
    if PixelBinning==1 :
        Ed_sq = np.arange(PixelStart_Acc,PixelStop_Acc+1,PixelBinning_Acc)
        
    Ed_wave = pd.array([ np.mean(cal_Ed.Wave[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])
    Ed_B0   = pd.array([ np.mean(cal_Ed.B0[   (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])
    Ed_B1   = pd.array([ np.mean(cal_Ed.B1[   (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])
    
    if Ed_InWater :
        Ed_S = pd.array([ np.mean(cal_Ed.S[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])
    else :
        Ed_S = pd.array([ np.mean(cal_Ed.Sair[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])

    # Averaging Ed dark coefficient
    Ed_B0_Dark = cal_Ed.B0[ cal_Ed.Wave==-1 ].mean()
    Ed_B1_Dark = cal_Ed.B1[ cal_Ed.Wave==-1 ].mean()
    Ed_B1_Dark
    
    # Rearange Lu Calibration parameter depending on profile configuration (found in metadata but for now in Edouard files)
    Lu_InWater=True
    
    # Averaging Lu calibrations factor 2by2
    Lu_sq = np.arange(PixelStart_Arc,PixelStop_Arc,PixelBinning_Arc)
    
    # ajout d'une condition pour éviter les problèmes de shape dans le cas PixelBinning=1
    if PixelBinning==1 :
        Lu_sq = np.arange(PixelStart_Acc,PixelStop_Acc+1,PixelBinning_Acc)
    
    Lu_wave = pd.array([ np.mean(cal_Lu.Wave[ (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])
    Lu_B0   = pd.array([ np.mean(cal_Lu.B0[   (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])
    Lu_B1   = pd.array([ np.mean(cal_Lu.B1[   (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])
    
    if Lu_InWater :
        Lu_S = pd.array([ np.mean(cal_Lu.S[ (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])
    else :
        Lu_S = pd.array([ np.mean(cal_Lu.Sair[ (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])

    # Averaging Lu dark coefficient
    Lu_B0_Dark = cal_Lu.B0[ cal_Lu.Wave==-1 ].mean()
    Lu_B1_Dark = cal_Lu.B1[ cal_Lu.Wave==-1 ].mean()

    Post_pres = Ed_raw_profile.Post_Pres[~Ed_raw_profile.Post_Pres.isna()]
    
    """ APPLY CALIBRATION """
    # Create the global table to save radiometric data of the profile into physics units
    Ed_physic_profile = pd.DataFrame(columns=np.round(Ed_wave))

    Ed_physic_profile.insert(0, 'PRE_TILT', Ed_raw_profile.Pre_Tilt[~Ed_raw_profile.Post_Pres.isna()])
    Ed_physic_profile.insert(1, 'POST_TILT', Ed_raw_profile.Post_Tilt[~Ed_raw_profile.Post_Pres.isna()])
    Ed_physic_profile.insert(0,'Post_Pres', Ed_raw_profile.Post_Pres[~Ed_raw_profile.Post_Pres.isna()])
    Ed_physic_profile.insert(0,'Pres_float', Ed_raw_profile.Pres_float[~Ed_raw_profile.Pres_float.isna()])
    
    # Fill the table with counts data converted
    for z in Post_pres :
    
        # Récupération du spectre à la profondeur z
        Ed_z_counts = Ed_raw_profile[ Ed_raw_profile.Pres_float==z ] 
        
        # Apply ra_single function to translate into physic units   
        Ed_z_physic = ra_single(Ed_z_counts,Ed_B0,Ed_B1,Ed_S,Ed_B0_Dark,Ed_B1_Dark)
        
        # Add into the global table
        Ed_physic_profile.loc[Ed_physic_profile.Pres_float==z, Ed_physic_profile.columns[2:]] = Ed_z_physic.reshape((1,-1))
        

    # Create the global table to save radiometric data into physics units
    Post_pres = Lu_raw_profile.Post_Pres[~Lu_raw_profile.Post_Pres.isna()]
    Lu_physic_profile = pd.DataFrame(columns=np.round(Lu_wave))
    Lu_physic_profile.insert(0,'Post_Pres', Lu_raw_profile.Post_Pres[~Lu_raw_profile.Post_Pres.isna()])
    Lu_physic_profile.insert(0,'Pres_float', Lu_raw_profile.Pres_float[~Lu_raw_profile.Pres_float.isna()])
    Lu_physic_profile.insert(0, 'PRE_TILT', Lu_raw_profile.Pre_Tilt[~Lu_raw_profile.Post_Pres.isna()])
    Lu_physic_profile.insert(1, 'POST_TILT', Lu_raw_profile.Post_Tilt[~Lu_raw_profile.Post_Pres.isna()])

    # Fill the table with counts data converted
    for z in Post_pres:
    
        # Récupération du spectre à la profondeur z
        Lu_z_counts = Lu_raw_profile[ Lu_raw_profile.Pres_float==z ] 
        
        # Apply ra_single function to translate into physic units
        Lu_z_physic = ra_single(Lu_z_counts,Lu_B0,Lu_B1,Lu_S,Lu_B0_Dark,Lu_B1_Dark)
        
        # Add into the global table
        Lu_physic_profile.loc[Lu_physic_profile.Pres_float==z, Lu_physic_profile.columns[2:]] = Lu_z_physic.reshape((1,-1))
    
    
    # remove pres_float which will be added in a second step outside this functiun (here we have only ramses data)
    Ed_physic_profile = Ed_physic_profile.drop(columns='Pres_float')
    Lu_physic_profile = Lu_physic_profile.drop(columns='Pres_float')
    
    return Ed_physic_profile, Lu_physic_profile

def format_ramses_ed_only(filename, metaname, calEd_name, Ed_n_prof, PixelBinning='auto', PixelStop='auto'):
    """
    Simplified function to obtain a Table of Ed in physics units for ONE profile of a BGC-Argo float.

    Parameters
    ----------
    filename : str
        Path of the netcdf file of the float's profile in counts (gdac/aux/coriolis/)
    metaname : str
        Path of the netcdf file of the float's metadata (gdac/aux/coriolis/)
    calEd_name : str
        Path of the txt file with the calibration coefficient of Ed sensor for this float.
    Ed_n_prof : int
        Profile number for Ed data extraction.
    PixelBinning : int or str
        Manual override for PixelBinning if necessary. Default='auto'.
    PixelStop : int or str
        Manual override for PixelStop if necessary. Default='auto'.

    Returns
    -------
    Ed_physic_profile : pandas.DataFrame
        Table of Ed values in W.m-2.nm-1 with dimensions: depth x wavelength
    """

    # RAWDATA: Extract data in counts (depth x wavelength), integration time (depth), and dark counts (depth)
    file = xr.open_dataset(filename)
    Ed_raw_profile = pd.DataFrame({
        'Post_Pres': file.RADIOMETER_DOWN_IRR_POST_PRES.sel(N_PROF=Ed_n_prof).values,
        'Pres_float': file.PRES.sel(N_PROF=Ed_n_prof).values,
        'Int_Time': file.RADIOMETER_DOWN_IRR_INTEGRATION_TIME.sel(N_PROF=Ed_n_prof).values,
        'tilt_1id': file.RADIOMETER_DOWN_IRR_PRE_INCLINATION.sel(N_PROF=Ed_n_prof).values,
        'tilt': file.RADIOMETER_DOWN_IRR_POST_INCLINATION.sel(N_PROF=Ed_n_prof).values,
        'Dark_count': file.RADIOMETER_DOWN_IRR_DARK_AVERAGE.sel(N_PROF=Ed_n_prof).values
    })
    Ed_raw_profile = pd.concat([Ed_raw_profile, pd.DataFrame(file.RAW_DOWNWELLING_IRRADIANCE.sel(N_PROF=Ed_n_prof).values)], axis=1)

    # METADATA: Extract Pixel configuration (PixelStart, PixelStop, and Binning)
    meta = xr.open_dataset(metaname)

    # Find Config parameters index of RAMSES 1 and 2
    index_Arc = np.where(
        meta.LAUNCH_CONFIG_PARAMETER_NAME.values == b'CONFIG_RamsesArcOutputPixelBegin_NUMBER                                                                                         ')[
        0][0]
    index_Acc = np.where(
        meta.LAUNCH_CONFIG_PARAMETER_NAME.values == b'CONFIG_RamsesAccOutputPixelBegin_NUMBER                                                                                         ')[
        0][0]

    # Find Config parameters thanks to the index
    [PixelStart_Acc, PixelStop_Acc, PixelBinning_Acc] = meta.LAUNCH_CONFIG_PARAMETER_VALUE.values[
                                                        index_Acc:index_Acc + 3]

    if PixelBinning != 'auto':
        PixelBinning_Acc, PixelBinning_Arc = PixelBinning, PixelBinning

    if PixelStop != 'auto':
        PixelStop_Acc, PixelStop_Arc = PixelStop, PixelStop

    # CALIBRATION FILES: Process calibration coefficients
    cal_Ed = pd.read_table(calEd_name, sep='\t')
    cal_Ed[ cal_Ed=="+NAN" ] = np.nan
    cal_Ed = cal_Ed.apply(pd.to_numeric, errors='coerce')

    # Rearange Ed Calibration parameter depending on profile configuration (found in metadata but for now in Edouard files)
    Ed_InWater = True
    # Ensure PixelStart_Acc, PixelStop_Acc, and PixelBinning_Acc are integers
    PixelStart_Acc_int = int(PixelStart_Acc)
    PixelStop_Acc_int = int(PixelStop_Acc)
    PixelBinning_Acc_int = int(PixelBinning_Acc)

    # Use the integer values with numpy.arange
    Ed_sq = np.arange(PixelStart_Acc_int, PixelStop_Acc_int, PixelBinning_Acc_int)
    # Averaging Ed calibrations factor PixelBinningxPixelBinning

    # ajout d'une condition pour éviter les problèmes de shape dans le cas PixelBinning=1
    if PixelBinning == 1:
        Ed_sq = np.arange(PixelStart_Acc_int, PixelStop_Acc_int + 1, PixelBinning_Acc_int)

    Ed_wave = pd.array(
        [np.mean(cal_Ed.Wave[(cal_Ed.N >= Ed_sq[i]) & (cal_Ed.N <= Ed_sq[i] + PixelBinning_Acc_int - 1)]) for i in
         range(len(Ed_sq))])
    Ed_B0 = pd.array(
        [np.mean(cal_Ed.B0[(cal_Ed.N >= Ed_sq[i]) & (cal_Ed.N <= Ed_sq[i] + PixelBinning_Acc_int - 1)]) for i in
         range(len(Ed_sq))])
    Ed_B1 = pd.array(
        [np.mean(cal_Ed.B1[(cal_Ed.N >= Ed_sq[i]) & (cal_Ed.N <= Ed_sq[i] + PixelBinning_Acc_int - 1)]) for i in
         range(len(Ed_sq))])

    if Ed_InWater:
        Ed_S = pd.array(
            [np.mean(cal_Ed.S[(cal_Ed.N >= Ed_sq[i]) & (cal_Ed.N <= Ed_sq[i] + PixelBinning_Acc_int - 1)]) for i in
             range(len(Ed_sq))])
    else:
        Ed_S = pd.array(
            [np.mean(cal_Ed.Sair[(cal_Ed.N >= Ed_sq[i]) & (cal_Ed.N <= Ed_sq[i] + PixelBinning_Acc_int - 1)]) for i in
             range(len(Ed_sq))])

    # Averaging Ed dark coefficient
    Ed_B0_Dark = cal_Ed.B0[cal_Ed.Wave == -1].mean()
    Ed_B1_Dark = cal_Ed.B1[cal_Ed.Wave == -1].mean()

    # APPLY CALIBRATION: Convert counts to physical units
    Ed_physic_profile = pd.DataFrame(columns=np.round(Ed_wave))
    Ed_physic_profile.insert(0, 'Pres_float', Ed_raw_profile.Pres_float[~Ed_raw_profile.Pres_float.isna()])
    Ed_physic_profile.insert(0, 'tilt', Ed_raw_profile.tilt[~Ed_raw_profile.tilt.isna()])
    Ed_physic_profile.insert(0, 'tilt_1id', Ed_raw_profile.tilt_1id[~Ed_raw_profile.tilt_id.isna()])

    # Fill the table with counts data converted
    for z in Ed_raw_profile.Pres_float[~Ed_raw_profile.Pres_float.isna()]:
        # Retrieve at depth z ommiting first three columns

        Ed_z_counts = Ed_raw_profile[Ed_raw_profile.Pres_float == z].iloc[:, 2:]

        # Apply ra_single function to translate into physic units
        Ed_z_physic = ra_single(Ed_z_counts, Ed_B0, Ed_B1, Ed_S, Ed_B0_Dark, Ed_B1_Dark)

        # Add into the global table
        Ed_physic_profile.loc[Ed_physic_profile.Pres_float == z, Ed_physic_profile.columns[3:]] = Ed_z_physic.reshape(
            (1, -1))
    Ed_physic_profile = Ed_physic_profile.drop('Pres_float', axis=1) # remove the Post_Pres column as the values are wrong
    return Ed_physic_profile

def format_ramses_Rdata(ramses,ramses2):
    """
    Function to obtain 2 Table of Ed and Lu in physics units for ONE profile of a BGC-Argo float from a RData file (directly load in the float).

    Parameters
    ----------
    ramses : pandas.DataFrame
        raw data of Ed directly send by the float
    ramses2 : pandas.DataFrame
        raw data of Lu directly send by the float

    Returns
    -------
    Ed_physic_profile : pandas.DataFrame
        Table of Ed values in W.m-2.nm-1 with dimensions : depth x wavelength
    Lu_physic_profile : pandas.DataFrame
        Table of Lu values in W.m-2.nm-1.sr-1 with dimensions : depth x wavelength

    """
    
    """ RAWDATA : Find data in counts(depthxwavelength), integration time(depth), dark in counts(depth) and depth """
    Ed_raw_profile = ramses.iloc[:,0:81]
    Lu_raw_profile = ramses2.iloc[:,0:81]
    
    """ METADONNEES : to find Pixels configuration (PixelStart, stop and Binning) """
    PixelStart = 5
    PixelStop = 144
    PixelBinning = 2
        

    """ CALIBRATION FILES : to find equation and coefficients to translate counts into physics units. """
    # open calibration files
    cal_Ed = pd.read_table('/home/lou/Documents/These/phd_axe1/Calibration_RAMSES/Data/RAMSES-TRIOS/01600028/SAM_8718_01600028_AllCal.txt', sep='\t')
    cal_Lu = pd.read_table('/home/lou/Documents/These/phd_axe1/Calibration_RAMSES/Data/RAMSES-TRIOS/01600050/SAM_877C_01600050_AllCal.txt', sep='\t')
    
    # correct the "+NAN" into NaN numpy
    cal_Ed[ cal_Ed=="+NAN" ] = np.nan
    cal_Lu[ cal_Lu=="+NAN" ] = np.nan
    
    # convert everything into numerical type values
    cal_Ed.S =pd.to_numeric(cal_Ed.S)
    cal_Lu.S =pd.to_numeric(cal_Lu.S)
    
    # Rearange Ed Calibration parameter depending on profile configuration (found in metadata but for now in Edouard files)
    Ed_InWater=True
    
    # Averaging Ed calibrations factor PixelBinningxPixelBinning
    Ed_sq = np.arange(PixelStart,PixelStop,PixelBinning)
    Ed_wave = pd.array([ np.mean(cal_Ed.Wave[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning-1) ]) for i in range (len(Ed_sq)) ])
    Ed_B0   = pd.array([ np.mean(cal_Ed.B0[   (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning-1) ]) for i in range (len(Ed_sq)) ])
    Ed_B1   = pd.array([ np.mean(cal_Ed.B1[   (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning-1) ]) for i in range (len(Ed_sq)) ])
    
    if Ed_InWater :
        Ed_S = pd.array([ np.mean(cal_Ed.S[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning-1) ]) for i in range (len(Ed_sq)) ])
    else :
        Ed_S = pd.array([ np.mean(cal_Ed.Sair[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning-1) ]) for i in range (len(Ed_sq)) ])

    # Averaging Ed dark coefficient
    Ed_B0_Dark = cal_Ed.B0[ cal_Ed.Wave==-1 ].mean()
    Ed_B1_Dark = cal_Ed.B1[ cal_Ed.Wave==-1 ].mean()
    
    # Rearange Lu Calibration parameter depending on profile configuration (found in metadata but for now in Edouard files)
    Lu_InWater=True
    # Averaging Lu calibrations factor 2by2
    Lu_sq = np.arange(PixelStart, PixelStop, PixelBinning)    
    Lu_wave = pd.array([ np.mean(cal_Lu.Wave[ (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning-1) ]) for i in range (len(Lu_sq)) ])
    Lu_B0   = pd.array([ np.mean(cal_Lu.B0[   (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning-1) ]) for i in range (len(Lu_sq)) ])
    Lu_B1   = pd.array([ np.mean(cal_Lu.B1[   (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning-1) ]) for i in range (len(Lu_sq)) ])
    
    if Lu_InWater :
        Lu_S = pd.array([ np.mean(cal_Lu.S[ (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning-1) ]) for i in range (len(Lu_sq)) ])
    else :
        Lu_S = pd.array([ np.mean(cal_Lu.Sair[ (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning-1) ]) for i in range (len(Lu_sq)) ])

    # Averaging Lu dark coefficient
    Lu_B0_Dark = cal_Lu.B0[ cal_Lu.Wave==-1 ].mean()
    Lu_B1_Dark = cal_Lu.B1[ cal_Lu.Wave==-1 ].mean()
    
    
    """ APPLY CALIBRATION """
    # Create the global table to save radiometric data of the profile into physics units
    Ed_physic_profile = pd.DataFrame(columns=np.round(Ed_wave))
    Ed_physic_profile.insert(0,'ramses_depth2', Ed_raw_profile.ramses_depth2[~Ed_raw_profile.ramses_depth2.isna()])
    
    # Fill the table with counts data converted
    for z in Ed_raw_profile.ramses_depth2[~Ed_raw_profile.ramses_depth2.isna()] :
    
        # Récupération du spectre à la profondeur z
        Ed_z_counts = Ed_raw_profile[ Ed_raw_profile.ramses_depth2==z ] 
        
        # Apply ra_single function to translate into physic units   
        Ed_z_physic = ra_single_RData(Ed_z_counts,Ed_B0,Ed_B1,Ed_S,Ed_B0_Dark,Ed_B1_Dark)
        
        # Add into the global table
        Ed_physic_profile.loc[Ed_physic_profile.ramses_depth2==z, Ed_physic_profile.columns[1:]] = Ed_z_physic.reshape((1,-1))

    # concat with float data
    Ed_physic_profile = pd.concat([Ed_raw_profile.iloc[:,0:11],Ed_physic_profile.iloc[:,1:]], axis=1)

    # Create the global table to save radiometric data into physics units
    Lu_physic_profile = pd.DataFrame(columns=np.round(Lu_wave))
    Lu_physic_profile.insert(0,'ramses_depth2', Lu_raw_profile.ramses_depth2[~Lu_raw_profile.ramses_depth2.isna()])
    
    
    # Fill the table with counts data converted
    for z in Lu_raw_profile.ramses_depth2[~Lu_raw_profile.ramses_depth2.isna()] :
    
        # Récupération du spectre à la profondeur z
        Lu_z_counts = Lu_raw_profile[ Lu_raw_profile.ramses_depth2==z ] 
        
        # Apply ra_single function to translate into physic units
        Lu_z_physic = ra_single_RData(Lu_z_counts,Lu_B0,Lu_B1,Lu_S,Lu_B0_Dark,Lu_B1_Dark)
        
        # Add into the global table
        Lu_physic_profile.loc[Lu_physic_profile.ramses_depth2==z, Lu_physic_profile.columns[1:]] = Lu_z_physic.reshape((1,-1))
        
    # concat with float data
    Lu_physic_profile = pd.concat([Lu_raw_profile.iloc[:,0:11],Lu_physic_profile.iloc[:,1:]], axis=1)

    return Ed_physic_profile, Lu_physic_profile

def format_ramses_buoy_mode(filename,cyc,metaname,calEd_name,calLu_name,PixelBinning='auto', PixelStop='auto'):
    """
    Function to obtain 2 Table of Ed and Lu in physics units for ONE profile of a BGC-Argo float netcdf Sfile from GDAC/aux in buoy mode (MEASUREMENT_CODE=711).
    
    Parameters
    ----------
    filename : str
        path of the netcdf file of the float's measurements in buoy mode in counts (gdac/aux/coriolis/)
    metaname : str
        path of the netcdf fileof the float's metadata (gdac/aux/coriolis/)
    calEd_name : str
        path of the txt file with the calibration coefficient of Ed sensor for this float.
        (supposed to be in the metadata, for now it is in Edouard computer)
    calLu_name : str
        path of the txt file with the calibration coefficient of Lu sensor for this float.
        (supposed to be in the metadata, for now it is in Edouard computer)
    PixelBinning : int
        security to allow user to fixe manually PixelBinning in case he knows that it is the wrong one in the config meta file
        (ex : wmo=4903660_013). Default='auto' --> means that we keep the PixelBinning given by the meta file.
    
    Returns
    -------
    Ed_physic_profile : pandas.DataFrame
        Table of Ed values in W.m-2.nm-1 with dimensions : depth x wavelength
    Lu_physic_profile : pandas.DataFrame
        Table of Lu values in W.m-2.nm-1.sr-1 with dimensions : depth x wavelength
    
    """
    
    """ RAWDATA : Find data in counts(depthxwavelength), integration time(depth), dark in counts(depth) and depth """
    # open raw data in counts for one profile
    file = xr.open_dataset(filename)
    
    # Find the n_measurement list of interest
    # n_meas = file.N_MEASUREMENT[(file.CYCLE_NUMBER==cyc)&(file.PRES<0.10)].values
    n_meas = file.N_MEASUREMENT[(file.CYCLE_NUMBER==cyc)&(file.MEASUREMENT_CODE==711)].values
    
    
    ## Extract Ed data into a table with : depth, int_time, dark_count, raw_count_lambda1, ..., raw_count_lambda2
    Ed_raw_profile = pd.DataFrame({ 'JULD':file.JULD.loc[{'N_MEASUREMENT':n_meas}].values,
                                   'Post_Pres':file.RADIOMETER_DOWN_IRR_POST_PRES.loc[{'N_MEASUREMENT':n_meas}].values,
                                       'Post_Tilt':file.RADIOMETER_DOWN_IRR_POST_INCLINATION.loc[{'N_MEASUREMENT':n_meas}].values/10,
                                        'Int_Time':file.RADIOMETER_DOWN_IRR_INTEGRATION_TIME.loc[{'N_MEASUREMENT':n_meas}].values,
                                        'Dark_count':file.RADIOMETER_DOWN_IRR_DARK_AVERAGE.loc[{'N_MEASUREMENT':n_meas}].values })
    Ed_raw_profile = pd.concat([Ed_raw_profile, pd.DataFrame(file.RAW_DOWNWELLING_IRRADIANCE.loc[{'N_MEASUREMENT':n_meas}].values)], axis=1)
    
    
    ## Extract Lu data into a table with : depth, int_time, dark_count, raw_count_lambda1, ..., raw_count_lambda2
    Lu_raw_profile = pd.DataFrame({ 'JULD':file.JULD.loc[{'N_MEASUREMENT':n_meas}].values,
                                   'Post_Pres':file.RADIOMETER_UP_RAD_POST_PRES.loc[{'N_MEASUREMENT':n_meas}].values,
                                       'Post_Tilt':(180-file.RADIOMETER_UP_RAD_POST_INCLINATION.loc[{'N_MEASUREMENT':n_meas}].values)/10,
                                        'Int_Time':file.RADIOMETER_UP_RAD_INTEGRATION_TIME.loc[{'N_MEASUREMENT':n_meas}].values,
                                        'Dark_count':file.RADIOMETER_UP_RAD_DARK_AVERAGE.loc[{'N_MEASUREMENT':n_meas}].values })
    Lu_raw_profile = pd.concat([Lu_raw_profile, pd.DataFrame(file.RAW_UPWELLING_RADIANCE.loc[{'N_MEASUREMENT':n_meas}].values)], axis=1)
    
    
    
    """ METADONNEES : to find Pixels configuration (PixelStart, stop and Binning) """
    # open meta data 
    meta = xr.open_dataset(metaname)
    
    # Find Config parameters index of RAMSES 1 and 2
    index_Arc = np.where(meta.LAUNCH_CONFIG_PARAMETER_NAME.values==b'CONFIG_RamsesArcOutputPixelBegin_NUMBER                                                                                         ')[0][0]
    index_Acc = np.where(meta.LAUNCH_CONFIG_PARAMETER_NAME.values==b'CONFIG_RamsesAccOutputPixelBegin_NUMBER                                                                                         ')[0][0]
    
    # Find Config parameters thanks to the index
    [PixelStart_Acc, PixelStop_Acc, PixelBinning_Acc] = meta.LAUNCH_CONFIG_PARAMETER_VALUE.values[index_Acc:index_Acc+3]
    [PixelStart_Arc, PixelStop_Arc, PixelBinning_Arc] = meta.LAUNCH_CONFIG_PARAMETER_VALUE.values[index_Arc:index_Arc+3]
    
    if PixelBinning != 'auto' :
        PixelBinning_Acc, PixelBinning_Arc = PixelBinning, PixelBinning
        
    if PixelStop != 'auto' :
        PixelStop_Acc, PixelStop_Arc = PixelStop, PixelStop
        
    
    """ CALIBRATION FILES : to find equation and coefficients to translate counts into physics units. """
    # open calibration files
    cal_Ed = pd.read_table(calEd_name, sep='\t')
    cal_Lu = pd.read_table(calLu_name, sep='\t')
    
    # correct the "+NAN" into NaN numpy
    cal_Ed[ cal_Ed=="+NAN" ] = np.nan
    cal_Lu[ cal_Lu=="+NAN" ] = np.nan
    
    # convert everything into numerical type values
    cal_Ed.S =pd.to_numeric(cal_Ed.S)
    cal_Lu.S =pd.to_numeric(cal_Lu.S)
    cal_Ed.Sair =pd.to_numeric(cal_Ed.Sair)
    cal_Lu.Sair =pd.to_numeric(cal_Lu.Sair)
    
    # Rearange Ed Calibration parameter depending on profile configuration (found in metadata but for now in Edouard files)
    Ed_InWater=False
    
    # Averaging Ed calibrations factor PixelBinningxPixelBinning
    Ed_sq = np.arange(PixelStart_Acc,PixelStop_Acc,PixelBinning_Acc)
    
    # ajout d'une condition pour éviter les problèmes de shape dans le cas PixelBinning=1
    if PixelBinning==1 :
        Ed_sq = np.arange(PixelStart_Acc,PixelStop_Acc+1,PixelBinning_Acc)
        
    Ed_wave = pd.array([ np.mean(cal_Ed.Wave[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])
    Ed_B0   = pd.array([ np.mean(cal_Ed.B0[   (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])
    Ed_B1   = pd.array([ np.mean(cal_Ed.B1[   (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])
    
    if Ed_InWater :
        Ed_S = pd.array([ np.mean(cal_Ed.S[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])
    else :
        Ed_S = pd.array([ np.mean(cal_Ed.Sair[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Ed_sq)) ])
    
    # Averaging Ed dark coefficient
    Ed_B0_Dark = cal_Ed.B0[ cal_Ed.Wave==-1 ].mean()
    Ed_B1_Dark = cal_Ed.B1[ cal_Ed.Wave==-1 ].mean()
    Ed_B1_Dark
    
    # Rearange Lu Calibration parameter depending on profile configuration (found in metadata but for now in Edouard files)
    Lu_InWater=True
    
    # Averaging Lu calibrations factor 2by2
    Lu_sq = np.arange(PixelStart_Arc,PixelStop_Arc,PixelBinning_Arc)
    
    # ajout d'une condition pour éviter les problèmes de shape dans le cas PixelBinning=1
    if PixelBinning==1 :
        Lu_sq = np.arange(PixelStart_Acc,PixelStop_Acc+1,PixelBinning_Acc)
    
    Lu_wave = pd.array([ np.mean(cal_Lu.Wave[ (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])
    Lu_B0   = pd.array([ np.mean(cal_Lu.B0[   (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])
    Lu_B1   = pd.array([ np.mean(cal_Lu.B1[   (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])
    
    if Lu_InWater :
        Lu_S = pd.array([ np.mean(cal_Lu.S[ (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])
    else :
        Lu_S = pd.array([ np.mean(cal_Lu.Sair[ (cal_Lu.N>=Lu_sq[i]) & (cal_Lu.N<=Lu_sq[i]+PixelBinning_Acc-1) ]) for i in range (len(Lu_sq)) ])
    
    # Averaging Lu dark coefficient
    Lu_B0_Dark = cal_Lu.B0[ cal_Lu.Wave==-1 ].mean()
    Lu_B1_Dark = cal_Lu.B1[ cal_Lu.Wave==-1 ].mean()
    
    
    """ APPLY CALIBRATION """
    # Create the global table to save radiometric data of the profile into physics units
    Ed_physic_profile = pd.DataFrame(columns=np.round(Ed_wave))
    Ed_physic_profile.insert(0,'Post_Pres', Ed_raw_profile.Post_Pres[~Ed_raw_profile.Post_Pres.isna()])
    Ed_physic_profile.insert(0,'Post_Tilt', Ed_raw_profile.Post_Tilt[~Ed_raw_profile.Post_Tilt.isna()])
    Ed_physic_profile.insert(0,'JULD', Ed_raw_profile.JULD[~Ed_raw_profile.Post_Pres.isna()])
    
    # Ed_raw_profile = Ed_raw_profile[Ed_physic_profile.Post_Pres.notna()]
    # return Ed_raw_profile
    for index, row in Ed_raw_profile[Ed_raw_profile.Post_Pres.notna()].iterrows():
        # Extraire une seule ligne et la transformer en DataFrame
        Ed_z_counts = pd.DataFrame(row).transpose()

        # Apply ra_single function to translate into physic units   
        Ed_z_physic = ra_single_buoy_mode(Ed_z_counts,Ed_B0,Ed_B1,Ed_S,Ed_B0_Dark,Ed_B1_Dark)
        Ed_z_physic = np.stack(Ed_z_physic)
        
        if ~np.all(np.isnan(Ed_z_physic)):
            # Add into the global table
            Ed_physic_profile.loc[index, Ed_physic_profile.columns[3:]] = Ed_z_physic.reshape((1,-1))
    

    # Create the global table to save radiometric data into physics units
    Lu_physic_profile = pd.DataFrame(columns=np.round(Lu_wave))
    Lu_physic_profile.insert(0,'Post_Pres', Lu_raw_profile.Post_Pres[~Lu_raw_profile.Post_Pres.isna()])
    Lu_physic_profile.insert(0,'Post_Tilt', Lu_raw_profile.Post_Tilt[~Lu_raw_profile.Post_Tilt.isna()])
    Lu_physic_profile.insert(0,'JULD', Lu_raw_profile.JULD[~Lu_raw_profile.Post_Pres.isna()])
    
    for index, row in Lu_raw_profile[Lu_raw_profile.Post_Pres.notna()].iterrows():
        # Extraire une seule ligne et la transformer en DataFrame
        Lu_z_counts = pd.DataFrame(row).transpose()

        # Apply ra_single function to translate into physic units   
        Lu_z_physic = ra_single_buoy_mode(Lu_z_counts,Lu_B0,Lu_B1,Lu_S,Lu_B0_Dark,Lu_B1_Dark)
        Lu_z_physic = np.stack(Lu_z_physic)
        
        if ~np.all(np.isnan(Lu_z_physic)):
            # Add into the global table
            Lu_physic_profile.loc[index, Lu_physic_profile.columns[3:]] = Lu_z_physic.reshape((1,-1))
        
    return Ed_physic_profile, Lu_physic_profile

def format_ramses_RamsesData(file,typ,cal_name) :
    """
    Function to obtain a pandas DataFrame of Ed or Lu in physics units from another pandas.DataFrame in counts.
    This DataFrame in counts is obtainded from a direct acquisition by a Ramses log on the computer.

    Parameters
    ----------
    typ : string
        'Ed' : for Ed values 
        'Lu' :  for Lu values
    file : pandas.DataFrame 
        NScan / Duration / IntegrationTime / Temperature / Tilt1 / Tilt2 / Pressure / 0 ... 254

    Returns
    -------
    Ed_physic_profile : pandas.DataFrame
        144 wavelengths in physic units + pressure

    """
    
    """ RAWDATA : Find data in counts(depthxwavelength), integration time(depth), dark in counts(depth) and depth """
    ## Extract Ed data into a table with : depth, int_time, dark_count, raw_count_lambda1, ..., raw_count_lambda2
    Ed_raw_profile = pd.DataFrame({ 'Post_Pres':file.Pressure,
                                        'Int_Time':file.IntegrationTime,
                                        'Dark_count':file.loc[:,'237':'254'].mean(axis=1) })
    Ed_raw_profile = pd.concat([Ed_raw_profile, file.loc[:,'5':'144']], axis=1)
    
    """ CALIBRATION FILES : to find equation and coefficients to translate counts into physics units. """
    # open calibration files
    # open calibration files
    path_trios = '/home/lou/Documents/These/phd_axe1/Calibration_RAMSES/Data/RAMSES-TRIOS/'
    cal_Ed = pd.read_table(path_trios+cal_name, sep='\t')
    
    # correct the "+NAN" into NaN numpy
    cal_Ed[ cal_Ed=="+NAN" ] = np.nan
    
    # convert everything into numerical type values
    cal_Ed.S =pd.to_numeric(cal_Ed.S)
    cal_Ed.Sair =pd.to_numeric(cal_Ed.Sair)
    
    # Rearange Ed Calibration parameter depending on profile configuration (found in metadata but for now in Edouard files)
    Ed_InWater=False
    
    # Averaging Ed calibrations factor PixelBinningxPixelBinning
    PixelStart = 5
    PixelStop = 144
    PixelBinning =  1
    Ed_sq = np.arange(PixelStart,PixelStop,PixelBinning)


    # ajout d'une condition pour éviter les problèmes de shape dans le cas PixelBinning=1
    if PixelBinning==1 :
        Ed_sq = np.arange(PixelStart,PixelStop+1,PixelBinning)
        
    Ed_wave = pd.array([ np.mean(cal_Ed.Wave[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning-1) ]) for i in range (len(Ed_sq)) ])
    Ed_B0   = pd.array([ np.mean(cal_Ed.B0[   (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning-1) ]) for i in range (len(Ed_sq)) ])
    Ed_B1   = pd.array([ np.mean(cal_Ed.B1[   (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning-1) ]) for i in range (len(Ed_sq)) ])
    
    if Ed_InWater :
        Ed_S = pd.array([ np.mean(cal_Ed.S[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning-1) ]) for i in range (len(Ed_sq)) ])
    else :
        Ed_S = pd.array([ np.mean(cal_Ed.Sair[ (cal_Ed.N>=Ed_sq[i]) & (cal_Ed.N<=Ed_sq[i]+PixelBinning-1) ]) for i in range (len(Ed_sq)) ])

    # Averaging Ed dark coefficient
    Ed_B0_Dark = cal_Ed.B0[ cal_Ed.Wave==-1 ].mean()
    Ed_B1_Dark = cal_Ed.B1[ cal_Ed.Wave==-1 ].mean()


    """ APPLY CALIBRATION """
    # Create the global table to save radiometric data of the profile into physics units
    Ed_physic_profile = pd.DataFrame(columns=np.round(Ed_wave))
    Ed_physic_profile.insert(0,'Post_Pres', Ed_raw_profile.Post_Pres[~Ed_raw_profile.Post_Pres.isna()])
    
    
    for index, row in Ed_raw_profile[Ed_raw_profile.Post_Pres.notna()].iterrows():
        # Extraire une seule ligne et la transformer en DataFrame
        Ed_z_counts = pd.DataFrame(row).transpose()
    
        # Apply ra_single function to translate into physic units   
        Ed_z_physic = ra_single_RamsesData(Ed_z_counts,Ed_B0,Ed_B1,Ed_S,Ed_B0_Dark,Ed_B1_Dark)
        Ed_z_physic = np.stack(Ed_z_physic)
        
        # Add into the global table
        if ~np.all(np.isnan(Ed_z_physic)):
            Ed_physic_profile.loc[index,Ed_physic_profile.columns[1:]] = Ed_z_physic.reshape((1,-1))

    return Ed_physic_profile


#%% CREATE PROFILE DATA FILES (Ed & Lu FOR EACH FLOAT) 

def create_profile_files(n_float, profile_dir, meta_dir, save_to) :
    
    """ Localize and open all needed files """
    # metadata files names
    meta_name = n_float+'_meta_aux.nc'
    
    # find calibration file names associated to this float sensors :
    tab = pd.read_table('/home/lou/Documents/These/phd_axe1/Calibration_RAMSES/Data/WMOvsNSerie.txt', sep='\t') 
    n_series = tab[ tab.WMO==int(n_float) ]
    
    # open calibration files (retrieved from Edouard)
    path_cal = '/home/lou/Documents/These/phd_axe1/Calibration_RAMSES/Data/RAMSES-TRIOS/'
    path_cal_Ed = path_cal+n_series.N_Serie[ n_series.EdLu=='Ed' ].iloc[0]+'/'
    path_cal_Lu = path_cal+n_series.N_Serie[ n_series.EdLu=='Lu' ].iloc[0]+'/'
    
    file_Ed = glob.glob('*AllCal*',root_dir=path_cal_Ed)
    file_Lu = glob.glob('*AllCal*',root_dir=path_cal_Lu)
    
    # list of the float's profiles sorted in alphabetic order
    float_dir = profile_dir+n_float+'/'
    files_name = sorted(os.listdir(float_dir))
    
    
    """ Cycle by cycle, write a csv file with ramses data in physical units """
    Ed_physic, Lu_physic = pd.DataFrame(), pd.DataFrame()

    for file_name in files_name :
        print(file_name)
        # open netcdf
        file = xr.open_dataset(float_dir+file_name) #engine='argo'
    
        # Find index in STATION_PARAMETERS values where Radiance and Irradiance are to know what is the N_PROF of Ed and Lu.
        # string de 64 caractères pour station parameter
        if len(np.where(file.STATION_PARAMETERS.values==b'RAW_DOWNWELLING_IRRADIANCE                                      ')[0]) == 0 :
            print('No Irradiance in {}'.format(file_name))
            continue
        if len(np.where(file.STATION_PARAMETERS.values==b'RAW_UPWELLING_RADIANCE                                          ')[0]) == 0 :
            print('No Radiance in {}'.format(file_name))
            continue
            
        Ed_n_prof = np.where(file.STATION_PARAMETERS.values==b'RAW_DOWNWELLING_IRRADIANCE                                      ')[0][0]
        Lu_n_prof = np.where(file.STATION_PARAMETERS.values==b'RAW_UPWELLING_RADIANCE                                          ')[0][0]
    
        # format ramses data
        # careful : some float have different pixel binning or pixel stop parameters. 
        if n_float=='4903660' and int(file_name[-10:-7]) > 12 :
            Ed_physic_profile, Lu_physic_profile = format_ramses(float_dir+file_name,meta_dir+meta_name,path_cal_Ed+file_Ed[0],
                                                                       path_cal_Lu+file_Lu[0], Ed_n_prof, Lu_n_prof, PixelBinning=1)
        elif n_float=='1902578' :
            Ed_physic_profile, Lu_physic_profile = format_ramses(float_dir+file_name,meta_dir+meta_name,path_cal_Ed+file_Ed[0],
                                                                       path_cal_Lu+file_Lu[0], Ed_n_prof, Lu_n_prof, PixelStop=144)
        
        else :
            Ed_physic_profile, Lu_physic_profile = format_ramses(float_dir+file_name,meta_dir+meta_name,path_cal_Ed+file_Ed[0],
                                                                   path_cal_Lu+file_Lu[0], Ed_n_prof, Lu_n_prof)
            
        
        # format into a complete dataframe
        profile_Ed = pd.DataFrame({'CRUISE': [file_name[1:8]]*Ed_physic_profile.shape[0],
                                   'CYCLE': [int(file_name[-10:-7])]*Ed_physic_profile.shape[0],
                                   'WMO': [file_name[1:-7]]*Ed_physic_profile.shape[0],
                                   'TIME': [file.JULD.sel(N_PROF=Ed_n_prof).values]*Ed_physic_profile.shape[0],
                                   'LON': [file.LONGITUDE.sel(N_PROF=Ed_n_prof).values]*Ed_physic_profile.shape[0],
                                   'LAT': [file.LATITUDE.sel(N_PROF=Ed_n_prof).values]*Ed_physic_profile.shape[0],
                                   'PRES_FLOAT' : file.PRES.sel(N_PROF=Ed_n_prof).values[0:Ed_physic_profile.shape[0]] })
        
        profile_Lu = pd.DataFrame({'CRUISE': [file_name[1:8]]*Lu_physic_profile.shape[0],
                                   'CYCLE': [int(file_name[-10:-7])]*Lu_physic_profile.shape[0],
                                   'WMO': [file_name[1:-7]]*Lu_physic_profile.shape[0],
                                    'TIME': [file.JULD.sel(N_PROF=Lu_n_prof).values]*Lu_physic_profile.shape[0],
                                   'LON': [file.LONGITUDE.sel(N_PROF=Lu_n_prof).values]*Lu_physic_profile.shape[0],
                                   'LAT': [file.LATITUDE.sel(N_PROF=Lu_n_prof).values]*Lu_physic_profile.shape[0],
                                   'PRES_FLOAT' : file.PRES.sel(N_PROF=Lu_n_prof).values[0:Lu_physic_profile.shape[0]] })
        
        Ed_profile = pd.concat([profile_Ed,Ed_physic_profile],axis=1)
        Lu_profile = pd.concat([profile_Lu,Lu_physic_profile],axis=1)
        
        # add to global table of the float
        Ed_physic = pd.concat([Ed_physic,Ed_profile])
        Lu_physic = pd.concat([Lu_physic,Lu_profile])
    
       
    # save csv file
    Ed_physic.to_csv(save_to+file_name[1:8]+'_Ed.csv', index=False)
    Lu_physic.to_csv(save_to+file_name[1:8]+'_Lu.csv', index=False)
    
    print('All files have been converted into a physics units table')
    return Ed_physic, Lu_physic

#%% CREATE BUOY DATA FILES (Ed & Lu FOR EACH FLOAT)

def create_buoy_files(n_float, nc_traj_path, meta_dir, save_to) :
    
    """ Localize and open all needed files """
    # metadata file name
    meta_name = str(n_float)+'_meta_aux.nc'
    
    # find calibration file names associated to this float sensors :
    tab = pd.read_table('/home/lou/Documents/These/phd_axe1/Calibration_RAMSES/Data/WMOvsNSerie.txt') 
    n_series = tab[ tab.WMO==int(n_float) ]
    
    # open calibration files (retrieved from Edouard)
    path_cal = '/home/lou/Documents/These/phd_axe1/Calibration_RAMSES/Data/RAMSES-TRIOS/'
    path_cal_Ed = path_cal+n_series.N_Serie[ n_series.EdLu=='Ed' ].iloc[0]+'/'
    path_cal_Lu = path_cal+n_series.N_Serie[ n_series.EdLu=='Lu' ].iloc[0]+'/'
    
    file_Ed = glob.glob('*AllCal*',root_dir=path_cal_Ed)
    file_Lu = glob.glob('*AllCal*',root_dir=path_cal_Lu)

    # file_name netcdf
    file_name = '{}_Rtraj_aux.nc'.format(n_float)
    buoy_nc = xr.open_dataset(nc_traj_path+file_name)
    
    
    """ Cycle by cycle, write a csv file with ramses data in physical units """
    Ed_physic, Lu_physic = pd.DataFrame(), pd.DataFrame()
    
    for cyc in np.unique(buoy_nc.CYCLE_NUMBER.values) :
        # print(cyc)
    
        # Find the n_measurement list of interest
        n_meas = buoy_nc.N_MEASUREMENT[(buoy_nc.CYCLE_NUMBER==cyc)&(buoy_nc.PRES<0.10)].values
        # n_meas = buoy_nc.N_MEASUREMENT[(buoy_nc.CYCLE_NUMBER==cyc)&(buoy_nc.MEASUREMENT_CODE==711)].values
    
        # format ramses data
        if n_float==4903660 and cyc > 12 :
            print('condition cyc>13')
            Ed_physic_profile, Lu_physic_profile = format_ramses_buoy_mode(nc_traj_path+file_name, cyc, meta_dir+meta_name, path_cal_Ed+file_Ed[0],
                                                                    path_cal_Lu+file_Lu[0], PixelBinning=1)
        if n_float==4903660 and cyc <= 12 :
            continue
            
        elif n_float==1902578 :
            Ed_physic_profile, Lu_physic_profile = format_ramses_buoy_mode(nc_traj_path+file_name, cyc, meta_dir+meta_name, path_cal_Ed+file_Ed[0],
                                                                   path_cal_Lu+file_Lu[0], PixelStop=144)
            
        else :
            Ed_physic_profile, Lu_physic_profile = format_ramses_buoy_mode(nc_traj_path+file_name, cyc, meta_dir+meta_name, path_cal_Ed+file_Ed[0],
                                                                   path_cal_Lu+file_Lu[0])
    
        
        # format into a complete dataframe
        length_Ed = buoy_nc.PRES.loc[{'N_MEASUREMENT':n_meas}].values[0:Ed_physic_profile.shape[0]].shape[0]
        length_Lu = buoy_nc.PRES.loc[{'N_MEASUREMENT':n_meas}].values[0:Lu_physic_profile.shape[0]].shape[0]

        profile_Ed = pd.DataFrame({'CRUISE': [file_name[:7]]*length_Ed,
                                    'CYCLE': [cyc]*length_Ed,
                                    'WMO': [file_name[:7]+'_'+ str(int(cyc)).zfill(3)]*length_Ed,
                                    # 'LON':buoy_nc.LON.loc[{'N_MEASUREMENT':n_meas}].values[0:Ed_physic_profile.shape[0]],
                                    # 'LAT':buoy_nc.LAT.loc[{'N_MEASUREMENT':n_meas}].values[0:Ed_physic_profile.shape[0]],
                                    'PRES_FLOAT' : buoy_nc.PRES.loc[{'N_MEASUREMENT':n_meas}].values[0:Ed_physic_profile.shape[0]] })
        
        profile_Lu = pd.DataFrame({'CRUISE': [file_name[:7]]*length_Lu,
                                    'CYCLE': [cyc]*length_Lu,
                                    'WMO': [file_name[:7]+'_'+ str(int(cyc)).zfill(3)]*length_Lu,
                                    # 'LON':buoy_nc.LON.loc[{'N_MEASUREMENT':n_meas}].values[0:Lu_physic_profile.shape[0]],
                                    # 'LAT':buoy_nc.LAT.loc[{'N_MEASUREMENT':n_meas}].values[0:Lu_physic_profile.shape[0]],
                                    'PRES_FLOAT' : buoy_nc.PRES.loc[{'N_MEASUREMENT':n_meas}].values[0:Lu_physic_profile.shape[0]] })

        # reset index to concat
        Ed_physic_profile.reset_index(drop=True, inplace=True)
        Lu_physic_profile.reset_index(drop=True, inplace=True)
        
        # concat profile properties and hyperspectral data
        Ed_profile = pd.concat([profile_Ed,Ed_physic_profile],axis=1)
        Lu_profile = pd.concat([profile_Lu,Lu_physic_profile],axis=1)
        
        # add to global table of the float
        Ed_physic = pd.concat([Ed_physic,Ed_profile])
        Lu_physic = pd.concat([Lu_physic,Lu_profile])


    # save csv file
    Ed_physic.to_csv(save_to+file_name[:7]+'_Ed.csv', index=False)
    Lu_physic.to_csv(save_to+file_name[:7]+'_Lu.csv', index=False)
    
    print('All files have been converted into a physics units table')
    return Ed_physic, Lu_physic
    
    
    
#%% PUT PRESSURE RAMSES TO THE LEVEL OF THE FLOAT CTD (1.78M FOR Lu & 0.17M FOR Ed)
def recal_pres_ramses(dtb, EdLu):
    """
    Function to recal the pressure sensor of RAMSES
    using the pressure sensor of the float.

    Parameters
    ----------
    dtb : pandas.DataFrame
        Table with the following columns ['CRUISE','CYCLE','WMO','TIME','LON','LAT','PRES_FLOAT','Post_Pres' +wavelengths]
        containing just one cycle of the float and Ed OR Lu spectra.
    EdLu : str
        To indicate if the dtb contains Ed or Lu spectra.

    Returns
    -------
    P_recal : pandas.Series
        a new vector with the new pressure of the RAMSES sensor recaled

    """

    # Distance between the pressure sensor and the Ed/Lu sensors
    delta_Ed = -0.17 # 17cm d'écart
    delta_Lu = 1.78 # 178cm d'écart
   
    # Choose the good one
    if EdLu == 'Ed' :
        delta = delta_Ed
    if EdLu == 'Lu' :
        delta = delta_Lu

    # find pressure at the surface
    P_surf = dtb.PRES_FLOAT.values.min()
    
    # recal pressure Ed/Lu at the surface
    P_surf_recal = P_surf + delta

    # compute offset between pressure Ed/Lu recaled and pressure Ed/Lu raw
    delta_P = P_surf_recal - np.nanmin(dtb.Post_Pres.values)
    
    # apply offset on all the pressure vector
    P_recal = dtb.Post_Pres + delta_P

    return P_recal


#%% VISUALISATION FUNCTIONS 
def plot_spectravswave(dtb,ylabel,xmin,xmax,ymin,ymax,title,to_save=None):
    """
    Function to plot a spectra : Lu/Ed vs Wavelength vs Depth(=colorbar)

    Parameters
    ----------
    dtb : pd.DataFrame
        with one colonne Post_Pres + one colonne per wavelength
    ylabel : str
        ylabel that will be shown on the  figure
    xmin : int
        xaxis minimum
    xmax : int
        xaxis maximum
    ymin : int
        yaxis minimum
    ymax : int
        yaxis maximum
    title : str
        title of the figure
    to_save : str, optional
        path+namefile where the figure will be save. If you don't want to save, enter None.
        The default is None.

    Returns
    -------
    None.

    """
    # Trier les DataFrames en fonction de la pression 'Post_Pres' en ordre décroissant
    dtb_inv = dtb.sort_values(by='Post_Pres', ascending=False)

    # setup the normalization and the colormap
    nValues=dtb_inv.Post_Pres.unique()
    normalize = mcolors.Normalize(vmin=nValues.min(), vmax=nValues.max())
    colormap = plt.cm.colors.ListedColormap(cmocean.cm.haline(normalize(nValues))).reversed()

    # plot figure
    plt.figure()
    for i in range(len(dtb_inv.Post_Pres)):
        plt.plot(pd.to_numeric(dtb_inv.columns[1:]), dtb_inv.iloc[i,1:], label=dtb_inv.Post_Pres.iloc[i], c=colormap(i), linewidth=0.5)
    
    # setup axis
    plt.xlim(xmin,xmax)
    plt.xlabel('Wavelength (nm)')
    plt.ylim(ymin,ymax)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.title(title)
    
    # setup the colorbar
    scalarmappaple = plt.cm.ScalarMappable(norm=normalize, cmap=colormap.reversed() )
    scalarmappaple.set_array(nValues)
    plt.colorbar(scalarmappaple, label='depth (db)')
    
    # Adjust the layout to prevent overlap of titles and labels
    plt.tight_layout()
    
    # savefig
    if to_save is not None :
        plt.savefig(to_save, dpi=300, bbox_inches='tight')
    

#%% OTHERS VARIABLES COMPUTING (Rrs)
def compute_Rrs(Ed,Lu,cmin_Ed,cmax_Ed,cmin_Lu,cmax_Lu):
    """
    Function to compute Rrs from a spectra of Ed and a spectra of Lu.
    
    Parameters
    ----------
    Ed : pandas.DataFrame
        Table of a column of Pressure named 'Post_Pres' and N number of columns
        associated to each wavelength/ Ramses pixels with Ed values
    Lu : pandas.DataFrame
        Table of a column of Pressure named 'Post_Pres' and N number of columns
        associated to each wavelength/ Ramses pixels with Lu values.
    cmin_Ed : int
        index of the first column with Ed values (minimum wavelength)
    cmax_Ed : int
        index of the last column with Ed values (maximum wavelength)
    cmin_Lu : int
        index of the first column with Lu values (minimum wavelength)
    cmax_Lu : int
        index of the last column with Lu values (maximum wavelength)

    Returns
    -------
    df_Rrs : pd.DataFrame
        Remote-sensing reflectance computed as Lu/Ed for each wavelength
        and each pressure.
        The dataframe has also N columns of Rrs and a column of Pressure.

    """
    
    # rename PRES_FLOAT column
    Ed = Ed.rename(columns={'PRES_FLOAT': 'PRES_FLOAT_Ed'})
    Lu = Lu.rename(columns={'PRES_FLOAT': 'PRES_FLOAT_Lu'})
    
    # remove nan into radiometric data
    Ed.dropna(inplace=True)
    Lu.dropna(inplace=True)
    
    # Concaténer les DataFrames en vérifiant les valeurs de PRES_FLOAT
    Ed_sorted = Ed.sort_values(by='PRES_FLOAT_Ed')
    Lu_sorted = Lu.sort_values(by='PRES_FLOAT_Lu')
    
    # Merge Ed & Lu
    merged = pd.merge_asof(Ed_sorted, Lu_sorted, 
                               left_on='PRES_FLOAT_Ed', right_on='PRES_FLOAT_Lu', 
                               suffixes=('_Ed', '_Lu'))
    
    # Compute Rrs
    Rrs = pd.DataFrame(merged.iloc[:,cmin_Lu:cmax_Lu].to_numpy()/merged.iloc[:,cmin_Ed:cmax_Ed].to_numpy(), columns=Ed.columns[cmin_Ed:cmax_Ed])
    
    
    # Réinitialiser les index des deux DataFrames avant de les concaténer
    Ed.reset_index(drop=True, inplace=True)
    Rrs.reset_index(drop=True, inplace=True)
    
    
    # Concat Post_Pres vector and Rrs
    df_Rrs =  pd.concat([Ed.iloc[:,:cmin_Ed-1], Rrs],  axis=1)
        
    return df_Rrs


#%% CONVERT IMU DATA IN PHYSICAL UNTIS AND COMPUTE HEADING ANGLE
def imu_process_heading(raw_mag, meta):

    # load calibration coefficients in meta data file
    mx0 = pd.to_numeric(meta.FLOAT_META_DATA_VALUE.values[meta.FLOAT_META_DATA_NAME.values==b'IMU_MAGNETOMETER_X_COR_OFFSET                                                                                                   '][0])
    my0 = pd.to_numeric(meta.FLOAT_META_DATA_VALUE.values[meta.FLOAT_META_DATA_NAME.values==b'IMU_MAGNETOMETER_Y_COR_OFFSET                                                                                                   '][0])
    mz0 = pd.to_numeric(meta.FLOAT_META_DATA_VALUE.values[meta.FLOAT_META_DATA_NAME.values==b'IMU_MAGNETOMETER_Z_COR_OFFSET                                                                                                   '][0])
    
    hi1 = pd.to_numeric(meta.FLOAT_META_DATA_VALUE.values[meta.FLOAT_META_DATA_NAME.values==b'IMU_COMPASS_HARD_IRON_COR_OFFSET1                                                                                               '][0])
    hi2 = pd.to_numeric(meta.FLOAT_META_DATA_VALUE.values[meta.FLOAT_META_DATA_NAME.values==b'IMU_COMPASS_HARD_IRON_COR_OFFSET2                                                                                               '][0])
    
    si11 = pd.to_numeric(meta.FLOAT_META_DATA_VALUE.values[meta.FLOAT_META_DATA_NAME.values==b'IMU_COMPASS_SOFT_IRON_COR_MATRIX11                                                                                              '][0])
    si12 = pd.to_numeric(meta.FLOAT_META_DATA_VALUE.values[meta.FLOAT_META_DATA_NAME.values==b'IMU_COMPASS_SOFT_IRON_COR_MATRIX12                                                                                              '][0])
    si21 = pd.to_numeric(meta.FLOAT_META_DATA_VALUE.values[meta.FLOAT_META_DATA_NAME.values==b'IMU_COMPASS_SOFT_IRON_COR_MATRIX21                                                                                              '][0])
    si22 = pd.to_numeric(meta.FLOAT_META_DATA_VALUE.values[meta.FLOAT_META_DATA_NAME.values==b'IMU_COMPASS_SOFT_IRON_COR_MATRIX22                                                                                              '][0])
    
    # Compensation simple et Orientation
    PhyMagx = raw_mag.Raw_Mx + mx0
    PhyMagy = raw_mag.Raw_Mz + mz0
    PhyMagz = raw_mag.Raw_My + my0
    
    # Compensation compas
    PhyMagx += hi1
    PhyMagy += hi2
    
    PhyMagx = PhyMagx * si11 + PhyMagy * si12
    PhyMagy = PhyMagx * si21 + PhyMagy * si22
    
    # Calcul de l'angle
    fHead = np.arctan2(PhyMagy, PhyMagx)
    
    # On retourne le résultat
    return np.degrees(fHead)


def process_raw_imu(data, meta):
    data2 = data.copy()
    data2['heading'] = np.nan
        
    for i in range(len(data)) :
        fheading = imu_process_heading(data.iloc[i,:], meta)          
        data2.loc[i,'heading'] = fheading

    return data2



if __name__=='__main__' :
    
    # Base filepath
    root = '/home/lou/Documents/These/phd_axe1/Calibration_RAMSES/Data/'
    meta_dir = root + 'Meta_netcdf/'
    profile_dir = root + 'Profiles_netcdf/'
    
    # Test flotteur OISO
    file_name = '4903660/'+'R4903660_013_aux.nc'
    meta_name = '4903660_meta_aux.nc'
        
    # open calibration file :
    path_cal_Ed = '/home/lou/Documents/These/phd_axe1/Calibration_RAMSES/Data/RAMSES-TRIOS/01600045/'
    path_cal_Lu = '/home/lou/Documents/These/phd_axe1/Calibration_RAMSES/Data/RAMSES-TRIOS/0160004B/'
    
    file_Ed = glob.glob('*AllCal*',root_dir=path_cal_Ed)
    file_Lu = glob.glob('*AllCal*',root_dir=path_cal_Lu)
    
    
    # format ramses data
    Ed_physic_profile, Lu_physic_profile = format_ramses(profile_dir+file_name,meta_dir+meta_name,path_cal_Ed+file_Ed[0],path_cal_Lu+file_Lu[0],PixelBinning=1)
    
    # save file
    
    
    # PLOT TO CHECK
    #Ed
    # setup the normalization and the colormap
    nValues=Ed_physic_profile.Post_Pres
    normalize = mcolors.Normalize(vmin=nValues.min(), vmax=nValues.max())
    colormap = plt.get_cmap(cmocean.cm.haline, len(Ed_physic_profile.Post_Pres)).reversed()
    
    # plot figure
    plt.figure()
    for i in range(len(Ed_physic_profile.Post_Pres)):
        plt.plot(Ed_physic_profile.columns[1:], Ed_physic_profile.iloc[i,1:], label=Ed_physic_profile.Post_Pres.iloc[i], c=colormap(i), linewidth=0.5)
    
    # setup axis
    plt.xlim(300,800)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Ed ($W.nm^{-1}.m^{-2}$)')
    plt.yscale('log')
    
    # setup the colorbar
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap )
    scalarmappaple.set_array(nValues)
    plt.colorbar(scalarmappaple, label='depth (db)')
    
    #Lu
    # setup the normalization and the colormap
    nValues=Lu_physic_profile.Post_Pres
    normalize = mcolors.Normalize(vmin=nValues.min(), vmax=nValues.max())
    colormap = plt.get_cmap(cmocean.cm.haline, len(Lu_physic_profile.Post_Pres)).reversed()
    
    # plot figure
    plt.figure()
    for i in range(len(Lu_physic_profile.Post_Pres)):
        plt.plot(Lu_physic_profile.columns[1:], Lu_physic_profile.iloc[i,1:], label=Lu_physic_profile.Post_Pres.iloc[i], c=colormap(i), linewidth=0.5)
    
    # setup axis
    plt.xlim(300,800)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Lu ($W.nm^{-1}.m^{-2}.sr^{-1}$)')
    plt.yscale('log')
    
    # setup the colorbar
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap )
    scalarmappaple.set_array(nValues)
    plt.colorbar(scalarmappaple, label='depth (db)')