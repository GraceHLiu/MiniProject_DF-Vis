##################################################################
### QC the MODIS reflectance data and fill in the low quality  ###
### pixels with temporally neibored pixels                     ###
##################################################################
import os
import glob
import QualityControl_GL
import numpy as np
from datetime import date, datetime
import pandas as pd
from scipy.signal import savgol_filter
import shutil

SiteName = 'HREC'
MODIS_ClippedData_dir = '/z0/lh349796/Rangeland/MODIS/I_Cliped_Data/'+ SiteName + '/'
MODIS_Data_dir = '/z0/lh349796/Rangeland/MODIS/II_Cliped_Repro_Resap_Data/'+ SiteName + '/'
MODIS_CleanData_dir = '/z0/lh349796/Rangeland/MODIS/III_Combined_Data/'+ SiteName + '/'
MODIS_InterData1_dir = '/z0/lh349796/Rangeland/MODIS/IIII_Input_Data/'+ SiteName + '_temp/'
MODIS_InterData2_dir = '/z0/lh349796/Rangeland/MODIS/IIII_Input_Data/' + SiteName + '/'
ProductName = ['MOD', 'MYD']

if not os.path.exists(MODIS_CleanData_dir):
    os.makedirs(MODIS_CleanData_dir)

# ## ~Step1: Read in the data
Dates = []
for File in os.listdir(MODIS_Data_dir):
    if File.endswith('nir.tiff'):
        Dates.append(File[8:15])
Dates = sorted(set(Dates))
MissedDates = []
for Date in Dates:
    for PD in ProductName:
        print 'Working on date ' + Date + '...'
        if len(glob.glob(MODIS_Data_dir + PD +'09GQ_'+ Date + '_red.tiff'))>0:
            Red_dir = glob.glob(MODIS_Data_dir + PD +'09GQ_'+ Date + '_red.tiff')[0]
            NIR_dir = glob.glob(MODIS_Data_dir + PD + '09GQ_'+ Date + '_nir.tiff')[0]
            if len(glob.glob(MODIS_Data_dir + PD + '09GA_'+ Date + '_qc.tiff'))>0:
                QC_dir = glob.glob(MODIS_Data_dir + PD + '09GA_' + Date + '_qc.tiff')[0]
            else:# use the other satellite's data if the designated one is not avaliable
                QC_dir = glob.glob(MODIS_Data_dir + '*09GA_' + Date + '_qc.tiff')[0]


            Red_Raw = QualityControl_GL.raster2array(Red_dir, 1)
            NIR_Raw = QualityControl_GL.raster2array(NIR_dir, 1)
            QC = QualityControl_GL.raster2array(QC_dir, 1)

            NDVI = 1.0 * (NIR_Raw-Red_Raw)/(NIR_Raw+Red_Raw)
            # QC[QC == 2] = 1
            QC[QC != 1 ] = 0 # the aerosol layer 00 climatology 01 low 10 medium 11 high

            NDVI_nan = NDVI*QC
            NDVI_nan[NDVI_nan==0] = np.nan
            outFile = os.path.join(MODIS_Data_dir,PD+ '09GQ_'+ Date + '_ndvi.tiff')
            print 'Saving ' + os.path.basename(outFile) + '...'
            QualityControl_GL.array2raster(Red_dir, outFile, NDVI_nan)
            # Calculating the raw NDVI and save in clip_dir
            outFile = os.path.join(MODIS_ClippedData_dir, PD + '09GQ_' + Date + '_ndvi.tiff')
            print 'Saving ' + os.path.basename(outFile) + '...'
            QualityControl_GL.array2raster(Red_dir, outFile, NDVI)
        else:
            print MODIS_Data_dir + PD +'09GQ_'+ Date + '_red.tiff does not exist!'
            MissedDates.append(MODIS_Data_dir + PD +'09GQ_'+ Date + '_red.tiff')
print 'The missing data are: '
for misseddate in  MissedDates:
    print misseddate
# ~Step2: Combine the MOD and MYD for the same day (spatially)
## save the maximum if both exist
MissedDates = []
for Date in Dates:
    if os.path.exists(MODIS_Data_dir + ProductName[0] +'09GQ_'+ Date + '_ndvi.tiff') and os.path.exists(MODIS_Data_dir + ProductName[1] +'09GQ_'+ Date + '_ndvi.tiff'):
        print '~~both MOD and MYD data exist for date ' + Date
        MOD = QualityControl_GL.raster2array(MODIS_Data_dir + ProductName[0] +'09GQ_'+ Date + '_ndvi.tiff', 1)
        MYD = QualityControl_GL.raster2array(MODIS_Data_dir + ProductName[1] +'09GQ_'+ Date + '_ndvi.tiff', 1)
        MIX = np.fmax(MOD, MYD)
        QualityControl_GL.array2raster(MODIS_Data_dir + ProductName[0] +'09GQ_'+ Date + '_ndvi.tiff',
                                       os.path.join(MODIS_CleanData_dir, 'MIX09GQ_'+ Date + '_ndvi.tiff'),
                                       MIX)
    elif ~os.path.exists(MODIS_Data_dir + ProductName[0] +'09GQ_'+ Date + '_ndvi.tiff') and os.path.exists(MODIS_Data_dir + ProductName[1] +'09GQ_'+ Date + '_ndvi.tiff'):
        print '~~no MOD file exists for date ' + Date
        # os.system('cp ' + MODIS_Data_dir + ProductName[1] + '09GQ_'+ Date + '_ndvi.tiff ' + MODIS_CleanData_dir + ProductName[1] + '09GQ_'+ Date + '_ndvi.tiff')
        os.system('cp ' + MODIS_Data_dir + ProductName[1] + '09GQ_' + Date + '_ndvi.tiff ' + MODIS_CleanData_dir + 'MIX09GQ_' + Date + '_ndvi.tiff')
    elif os.path.exists(MODIS_Data_dir + ProductName[0] +'09GQ_'+ Date + '_ndvi.tiff') and ~os.path.exists(MODIS_Data_dir + ProductName[1] +'09GQ_'+ Date + '_ndvi.tiff'):
        print '~~no MYD file exists for date ' + Date
        # os.system('cp '+ MODIS_Data_dir + ProductName[0] + '09GQ_' + Date + '_ndvi.tiff ' + MODIS_CleanData_dir + ProductName[0] + '09GQ_' + Date + '_ndvi.tiff')
        os.system('cp ' + MODIS_Data_dir + ProductName[0] + '09GQ_' + Date + '_ndvi.tiff ' + MODIS_CleanData_dir + 'MIX09GQ_' + Date + '_ndvi.tiff')
    else:
        print 'no MODIS file exists for date '+ Date + '!!!'
        MissedDates.append(Date)
print 'The missing date are: '
for misseddate in MissedDates:
    print misseddate
### ~Step3: Fill in the NA pixels using temporal interpolation
print 'reading in the MODIS mixed data...'
if not os.path.exists(MODIS_InterData1_dir):
    os.makedirs(MODIS_InterData1_dir)
if not os.path.exists(MODIS_InterData2_dir):
    os.makedirs(MODIS_InterData2_dir)
date_str =[]
filelist = []
for file in os.listdir(MODIS_CleanData_dir):
    if file.endswith("ndvi.tiff") and file.startswith('MIX'):
        file_path = os.path.join(MODIS_CleanData_dir, file)
        date_str.append(file[8:15])
        filelist.append(file_path)
datelist_str = sorted(list(set(date_str)))
filelist = sorted(filelist)
## convert datelist to a datetime object
datelist = []
for date_var in datelist_str:
    temp = datetime.strptime(date_var, '%Y%j')
    datelist.append(date(temp.year, temp.month, temp.day))
## Check if there is any missing dates and add empty raster for missing dates
if len(MissedDates)!=0:
    for misseddate in MissedDates:
        print 'creating NAN raster for date '+ misseddate
        OutFile = os.path.join(MODIS_InterData1_dir, "MYD09GA_" + misseddate + "_simu_ndvi.tiff")
        InFile = os.path.join(MODIS_CleanData_dir, filelist[0])
        raster = QualityControl_GL.raster2array(InFile, 1)
        raster[~np.isnan(raster)] = np.nan
        QualityControl_GL.array2raster(InFile, OutFile, raster)

else:
    print 'there is no missing dates in the combined dataset'
    for file in filelist:
        print 'copying file' + os.path.basename(file) + 'from III to IIII_temp'
        os.system('cp ' + file + ' ' + MODIS_InterData1_dir + os.path.basename(file))
print 'converting the TIFFs to numpy array...'
Multi_raster_path = []
for raster in os.listdir(MODIS_InterData1_dir):
    if raster.endswith('ndvi.tiff'):
        raster_path = os.path.join(MODIS_InterData1_dir, raster)
        Multi_raster_path.append(raster_path)
Multi_raster_path = sorted(Multi_raster_path)
Multi_raster = []
for raster_path in Multi_raster_path:
    Multi_raster.append(QualityControl_GL.raster2array(raster_path, 1))
Raster_array = np.array(Multi_raster)
Raster_dim = Raster_array.shape
print 'Conversion finished!'

print 'Getting ready to interpolate and smooth...'
# Interpolation and Smoothing
for row in np.arange(Raster_dim[1]):
    for col in np.arange(Raster_dim[2]):
        print 'interpolating pixel (' + str(row) + ',' + str(col) + ')...'
        time_series = pd.Series(Raster_array[:, row, col])
        #print time_series
        # interpolate missing data
        ## do spatial interpolation if there is less than 3 non nans in a time series
        if np.count_nonzero(~np.isnan(time_series)) <= 3:
            time_series = QualityControl_GL.SpatialInterpolation(col, row, Raster_array)
            #print time_series
        time_series_interp = time_series.interpolate(method="linear")
        #print time_series_interp
        print 'applying the SavGol filter to pixel (' + str(row) + ',' + str(col) + ')...'
        # apply SavGol filter
        time_series_savgol = savgol_filter(time_series_interp, window_length=21, polyorder=2)
        time_series_savgol = savgol_filter(time_series_savgol, window_length=7, polyorder=3)
        #print time_series_savgol
        Raster_array[:, row, col] = time_series_savgol
        print 'interpolation and smoothing for pixel (' + str(row) + ',' + str(col) + ') is finished!'
print 'interpolation and smoothing complete!'

print 'Getting ready to save the results...'
filelist = []
date_str = []
for file in os.listdir(MODIS_InterData1_dir):
    if file.endswith("ndvi.tiff"):
        date_str.append(file[8:15])
        filelist.append(file)
datelist_str = sorted(list(set(date_str)))
filelist = sorted(set(filelist))

for date in np.arange(Raster_dim[0]):
    OutFile = os.path.join(MODIS_InterData2_dir, filelist[date])
    InFile = os.path.join(MODIS_InterData1_dir, filelist[0])
    raster = Raster_array[date, :, :]
    print 'Saving ' + OutFile + '...'
    QualityControl_GL.array2raster(InFile, OutFile, raster)
print 'code running complete!'

shutil.rmtree(MODIS_InterData1_dir)