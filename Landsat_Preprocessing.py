# ########################################################################################################################
# ###~~~~~~~~~~~~~~STEP ONE Unzip Clipping~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`###
# ########################################################################################################################
import os
import glob


# # Define Directories
SiteName = 'SLO'
Projection = 'EPSG:32610' #WGS84 utm zone 10 N #'EPSG:26910'# NAD83  utm zone 10 N
box = '740070  3931100 745174 3935183'## xmin ymin xmax ymax
# need to add the te_srs parameter in gralwarp because 042035 is in UTM 11 
Path = ['042','043']
Row = ['035']
Landsat_org_tar_Path = '/z0/lh349796/Rangeland/landsat_data/espa-haxliu@ucdavis.edu-11112018-130407-329/' #Zipped file directory
Landsat_Raw_Path = '/z0/lh349796/Rangeland/landsat_data/Raw_data/espa-haxliu@ucdavis.edu-11112018-130407-329/' # Where extracted data are saved
Landsat_Org_mos_Path = '' # Where mossaiced data are saved
Reproject_Path = '/z0/lh349796/Rangeland/landsat_data/'+ SiteName +'/I_Clipped_data/Landsat8_full_temp'
Path1 = '' #Where files need reproject and clip
Path2 = '' #Where files need reproject and clip
Landsat_Inputs_Path = '/z0/lh349796/Rangeland/landsat_data/'+ SiteName +'/I_Clipped_data/Landsat8_full' #Where reprojected and clipped data are saved

if not os.path.exists(Landsat_Raw_Path):
    os.makedirs(Landsat_Raw_Path)
if not os.path.exists(Landsat_Inputs_Path):
    os.makedirs(Landsat_Inputs_Path)
if not os.path.exists(Reproject_Path):
    os.makedirs(Reproject_Path)
# # ~Prepare Landsat Spatial file #
# unzip XXX.tar.gz
# for file in os.listdir(Landsat_org_tar_Path):
#    if file.endswith("tar.gz"):
#        print "Extracting "+file
#        FileName = os.path.join(Landsat_org_tar_Path, file)
#        os.system("tar xzvf " + FileName + " -C " +Landsat_Raw_Path)
## Loop through file directory, create a list of date and band
date =[]
# #band =[]
#
for file in os.listdir(Landsat_Raw_Path):
    if file.endswith(".tif"):
        date.append(file[17:25])
        #band.append(file[22:])
datelist = list(set(date))

print datelist
#print bandlist
###For a certain date, if both file exist, mosiac and move, else don't do anything.
# for d in datelist:
#    for b in bandlist:
#        file_1_Path =  glob.glob(Landsat_Raw_Path + "LT5045033"+d+"*_sr_"+b)
#        #folder572 = glob.glob(file_path_beginning + '*57221*')
#        file_2_Path =  os.path.join(Landsat_Raw_Path,"LC8044033"+d+"LGN00_"+b)
#        out_file_Path =  os.path.join(Landsat_Org_mos_Path,'LC8'+d+"LGN00_"+b)
#        if os.path.isfile(file_1_Path) and os.path.isfile(file_2_Path):
#            print('Merging L8 day '+d+'band '+b)
#            os.system("gdal_merge.py -o "+ out_file_Path + " -n -9999 " + file_1_Path+" "+file_2_Path)

#~Step3: Reproject & Clip XXX.tif
In_Directory = [Landsat_Raw_Path]
for directory in In_Directory:
    for file in os.listdir(directory):
        if file.endswith("bqa.tif") or file.endswith('ndvi.tif'):
            current_file = os.path.join(directory, file)
            export_file = os.path.join(Reproject_Path , file)
            print("cliping data...")
            os.system("gdalwarp -t_srs " + Projection + " " + current_file + " " + export_file)
            export_file_new = os.path.join(Landsat_Inputs_Path, file)
            os.system("gdalwarp -te " + box + " " + export_file + " " + export_file_new)
            print("done!")
# #
# # ########################################################################################################################
# # ###~~~~~~~~~`~~STEP 2 Qualitity Control~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
# # ########################################################################################################################
import os, glob,math,json,ogr, osr
from datetime import datetime
from osgeo import gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import numpy as np

import sys
# # Define Directories
Landsat_Clip_Path = '/z0/lh349796/Rangeland/landsat_data/' + SiteName + '/I_Clipped_data/Landsat8_full' # Where the clipped data are saved
Landsat_Clean_Path = '/z0/lh349796/Rangeland/landsat_data/' + SiteName + '/II_Cfmasked_data/Landsat8_full' # Where the cleaned data are saved
projection = 'EPSG:32610' #WGS84 UTM 10N

## Define functions
def raster2array(rasterfn,i):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(i)
    return band.ReadAsArray()

def array2raster(rasterfn,newRasterfn,array):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())

### Loop through file directory, create a list of date and band
if not os.path.exists(Landsat_Clean_Path):
    os.makedirs(Landsat_Clean_Path)
date =[]
for file in os.listdir(Landsat_Clip_Path):
    #print file
    if file.endswith("ndvi.tif"):
        file_path = os.path.join(Landsat_Clip_Path, file)
        #print file_path
        date.append(file[17:25])


datelist = list(set(date))

# #### use the landsat QA tool (https://landsat.usgs.gov/landsat-qa-tools) to generate cloud mask using the *bqa.tif data
# ### standard used is: cloud pixel with high level confidence and cirrus pixels with low, medium or high level confidence

for file in os.listdir(Landsat_Clip_Path):
    if file.endswith('bqa.tif'):
        infile = os.path.join(Landsat_Clip_Path,file)
        os.system('/z0/lh349796/Rangeland/landsat_data/unpack_oli_qa --ifile=' + infile + ' --ofile=' + infile[:-7] + 'cmask.tif --fill --cloud_shadow=high --cloud=high --cirrus=low --combine')

##~ step 1: For a certain date, if both file exist, calculate the cleaned data, else don't do anything.
for d in datelist:
    for r in Row:
        for p in Path:
            file_path_beginning = os.path.join(Landsat_Clip_Path, 'LC')
            NDVI_dir = glob.glob(file_path_beginning + '*' + p + r +  '*'  +d+'*sr_ndvi*')
            cfmask_dir = glob.glob(file_path_beginning + '*' + p + r + '*' +d+'*cmask*')
            if not NDVI_dir or not cfmask_dir:
                print 'empty'
                continue
            else:
                print 'both exist'
        	NDVI = raster2array(NDVI_dir[0],1)
        	cfmask = raster2array(cfmask_dir[0],1)
        	NDVI_cleaned = NDVI * 0.0001
        	NDVI_cleaned[cfmask == 1] = np.nan
        	outFile = os.path.join(Landsat_Clean_Path ,os.path.basename(NDVI_dir[0])[:-4]+ '_cleaned.tif')
        	print("Applying the cfmask to the NDVI image on " +d+"...")
        	array2raster(NDVI_dir[0],outFile,NDVI_cleaned)
        	print("done!")

# ########################################################################################################################
# ###~~~~~~~~~`~~STEP 3 Interpolation and Smoothing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
# ########################################################################################################################
# import gdal
# import os
# import osr
# from datetime import date, datetime, timedelta
# import numpy as np
# import pandas as pd
# from scipy.signal import savgol_filter
# import shutil
#
#
# Landsat_cfmasked_dir = '/z0/lh349796/Rangeland/landsat_data/' + SiteName + '/II_Cfmasked_data/Landsat8_full' #where the cfmasked landsat data are saved
# Landsat_inter_1_dir = '/z0/lh349796/Rangeland/landsat_data/' + SiteName + '/III_Processed_data/Landsat8_full/Interpolated_Nan_cubic' #save the whole time series with NANs
# Landsat_inter_2_dir = '/z0/lh349796/Rangeland/landsat_data/' + SiteName + '/III_Processed_data/Landsat8_full/Interpolated_final_cubic' #save the final products
# # Landsat_cfmasked_dir = 'C:/Users/Lenovo/Desktop/Landsat8' #where the cfmasked landsat data are saved
# # Landsat_inter_1_dir = 'C:/Users/Lenovo/Desktop/Landsat8/Interpolated_1_cubic' #save the whole time series with NANs
# # Landsat_inter_2_dir = 'C:/Users/Lenovo/Desktop/Landsat8/Interpolated_2_cubic' #Nsave the final products
#
# # Define functions
# def raster2array(rasterfn,i):
#     raster = gdal.Open(rasterfn)
#     band = raster.GetRasterBand(i)
#     return band.ReadAsArray()
#
# def array2raster(rasterfn,newRasterfn,array):
#     raster = gdal.Open(rasterfn)
#     geotransform = raster.GetGeoTransform()
#     originX = geotransform[0]
#     originY = geotransform[3]
#     pixelWidth = geotransform[1]
#     pixelHeight = geotransform[5]
#     cols = raster.RasterXSize
#     rows = raster.RasterYSize
#
#     driver = gdal.GetDriverByName('GTiff')
#     outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
#     outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
#     outband = outRaster.GetRasterBand(1)
#     outband.WriteArray(array)
#     outRasterSRS = osr.SpatialReference()
#     outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
#     outRaster.SetProjection(outRasterSRS.ExportToWkt())
#
# # for generating a list of date with delta interval
# def perdelta(start, end, delta):
#     curr = start
#     while curr < end:
#         yield curr
#         curr += delta
#
# # Setup Directory Functions#
# # Function to create directories - written by Andy J.Y. Wong
# def createFolders(Path, Objects):
#     for folder in Objects:
#         folder_Path = os.path.join(Path, folder)
#         if not os.path.exists(folder_Path):
#             os.mkdir(folder_Path)
#     return
#
# # Function to remove files in a defined directories  -  written by Andy J.Y. Wong
# def RemoveFilesInFolder(Path):
#     import os
#     for file in os.listdir(Path):
#         target = os.path.join(Path, file)
#         if os.path.isfile(target):
#             os.remove(target)
#     return
#
# # function to do spatial interpolation - written by Han Liu
# def SpatialInterpolation(col,row,Raster_array):
#     if col == 0 and row == 0:
#         time_series = pd.Series(np.nanmean(np.array([Raster_array[:, row + 1, col],
#                                                      Raster_array[:, row, col + 1],
#                                                      Raster_array[:, row + 1, col + 1]]), axis=0))
#     elif col != 0 and row == 0 and col != (Raster_dim[2] - 1):
#         time_series = pd.Series(np.nanmean(np.array([Raster_array[:, row, col - 1],
#                                                      Raster_array[:, row, col + 1],
#                                                      Raster_array[:, row + 1, col - 1],
#                                                      Raster_array[:, row + 1, col],
#                                                      Raster_array[:, row + 1, col + 1]]), axis=0))
#     elif row == 0 and col == (Raster_dim[2] - 1):
#         time_series = pd.Series(np.nanmean(np.array([Raster_array[:, row, col - 1],
#                                                      Raster_array[:, row + 1, col],
#                                                      Raster_array[:, row + 1, col - 1]]), axis=0))
#     elif col == 0 and row != 0 and row != (Raster_dim[1] - 1):
#         time_series = pd.Series(np.nanmean(np.array([Raster_array[:, row - 1, col],
#                                                      Raster_array[:, row + 1, col],
#                                                      Raster_array[:, row - 1, col + 1],
#                                                      Raster_array[:, row, col + 1],
#                                                      Raster_array[:, row + 1, col + 1]]), axis=0))
#     elif col != 0 and row != 0 and row != (Raster_dim[1] - 1) and col != (Raster_dim[2] - 1):
#         time_series = pd.Series(np.nanmean(np.array([Raster_array[:, row - 1, col - 1],
#                                                      Raster_array[:, row - 1, col],
#                                                      Raster_array[:, row - 1, col + 1],
#                                                      Raster_array[:, row, col - 1],
#                                                      Raster_array[:, row, col + 1],
#                                                      Raster_array[:, row + 1, col - 1],
#                                                      Raster_array[:, row + 1, col],
#                                                      Raster_array[:, row + 1, col + 1]]), axis=0))
#     elif col == (Raster_dim[2] - 1) and row != 0 and row != (Raster_dim[1] - 1):
#         time_series = pd.Series(np.nanmean(np.array([Raster_array[:, row - 1, col - 1],
#                                                      Raster_array[:, row - 1, col],
#                                                      Raster_array[:, row, col - 1],
#                                                      Raster_array[:, row + 1, col - 1],
#                                                      Raster_array[:, row + 1, col]]), axis=0))
#     elif row == (Raster_dim[1] - 1) and col == 0:
#         time_series = pd.Series(np.nanmean(np.array([Raster_array[:, row - 1, col],
#                                                      Raster_array[:, row - 1, col + 1],
#                                                      Raster_array[:, row, col + 1]]), axis=0))
#     elif col != 0 and col != (Raster_dim[2] - 1) and row == (Raster_dim[1] - 1):
#         time_series = pd.Series(np.nanmean(np.array([Raster_array[:, row - 1, col - 1],
#                                                      Raster_array[:, row - 1, col],
#                                                      Raster_array[:, row - 1, col + 1],
#                                                      Raster_array[:, row, col - 1],
#                                                      Raster_array[:, row, col + 1]]), axis=0))
#     elif row == (Raster_dim[1] - 1) and col == (Raster_dim[2] - 1):
#         time_series = pd.Series(np.nanmean(np.array([Raster_array[:, row - 1, col - 1],
#                                                      Raster_array[:, row - 1, col],
#                                                      Raster_array[:, row, col - 1]]), axis=0))
#     return time_series
#
# # functions end
#
# # ~ Step 1 ~  Read in the data and create NAN tiffs for missing dates
# ## Loop through file directory, create a list of date  and file
# print 'reading in the landsat dataset...'
# if not os.path.exists(Landsat_inter_1_dir):
#     os.makedirs(Landsat_inter_1_dir)
# if not os.path.exists(Landsat_inter_2_dir):
#     os.makedirs(Landsat_inter_2_dir)
# date_str =[]
# filelist = []
# for file in os.listdir(Landsat_cfmasked_dir):
#     if file.endswith("cleaned.tif"):
#         file_path = os.path.join(Landsat_cfmasked_dir, file)
#         #print file_path
#         date_str.append(file[17:25])
#         filelist.append(file)
# datelist_str = sorted(list(set(date_str)))
# filelist = sorted(filelist, key=lambda x: x.split('_')[3])
#
# ## convert datelist to a datetime object
# datelist = []
# for date_var in datelist_str:
#     temp = datetime.strptime(date_var, '%Y%m%d')
#     datelist.append(date(temp.year, temp.month, temp.day))
# ## Check the 16 days interval and add empty raster
# date_start = datelist[0]
# date_end = datelist[len(datelist)-1]
# datelist_whole = []
# for result in perdelta(date_start, date_end, timedelta(days=1)):
#     datelist_whole.append(result)
# datelist_whole.append(date_end)
#
# ## check the daily interval and add empty raster
# i = 0 #for looping through the landsat dates
# for date_whole in datelist_whole:
#     day_of_year = date_whole.strftime('%Y%j')
#     if date_whole == datelist[i]:
#         print "landsat data existed for date " + str(datelist[i])
#         # check if NDVI == 0 replace them with NAN (in the quality control part couldy pixels are replaced with 0)
#         InFile = os.path.join(Landsat_cfmasked_dir, filelist[i])
#         OutFile = os.path.join(Landsat_inter_1_dir, filelist[i][0:17] + day_of_year + "_sr_ndvi_cleaned.tif")
#         image = raster2array(InFile,1)
#         image[image == 0] = np.nan
#         array2raster(InFile, OutFile, image)
#         i += 1
#     elif datelist[i] > date_whole:
#         print "creating a scene for date " + str(date_whole) + " ..."
#         OutFile = os.path.join(Landsat_inter_1_dir, filelist[i][0:17] + day_of_year + "simu_sr_ndvi_cleaned.tif")
#         InFile = os.path.join(Landsat_cfmasked_dir, filelist[i])
#         raster = raster2array(InFile, 1)
#         raster[~np.isnan(raster)] = np.nan
#         array2raster(InFile, OutFile, raster)
#
# print 'Landset dataset are readed'
# # ~ Step 2 ~  Interpolation & Smoothing (SG filter)
# # read in rasters as a 3d arrary
# print 'converting the TIFFs to numpy array...'
# Multi_raster_path = []
# for raster in os.listdir(Landsat_inter_1_dir):
#     if raster.endswith('cleaned.tif'):
#         Multi_raster_path.append(raster)
# Multi_raster_path = sorted(Multi_raster_path, key=lambda x: x.split('_')[3])
# Multi_raster = []
# for raster_path in Multi_raster_path:
#     Multi_raster.append(raster2array(os.path.join(Landsat_inter_1_dir, raster_path), 1))
# Raster_array = np.array(Multi_raster)
# Raster_dim = Raster_array.shape
# print 'Conversion finished!'
# print 'Getting ready to interpolate and smooth...'
# # Interpolation and Smoothing
# for row in np.arange(Raster_dim[1]):
#     for col in np.arange(Raster_dim[2]):
#         print 'interpolating pixel (' + str(row) + ',' + str(col) + ')...'
#         time_series = pd.Series(Raster_array[:, row, col])
#         # interpolate missing data
#         ## do spatial interpolation if there is less than 3 non nans in a time series
#         if np.count_nonzero(~np.isnan(time_series)) <= 3:
#             time_series = SpatialInterpolation(col, row, Raster_array)
#         time_series_interp = time_series.interpolate(method="linear")
#         print 'applying the SavGol filter to pixel (' + str(row) + ',' + str(col) + ')...'
#         # apply SavGol filter
#         #time_series_savgol = savgol_filter(time_series_interp, window_length=15, polyorder=2)
#         #time_series_savgol = savgol_filter(time_series_savgol, window_length=5, polyorder=3)
#         time_series_savgol = savgol_filter(time_series_interp, window_length=5, polyorder=2)
#
#         Raster_array[:, row, col] = time_series_savgol
#         print 'interpolation and smoothing for pixel (' + str(row) + ',' + str(col) + ') is finished!'
# print 'interpolation and smoothing complete!'
# # ~ Step 3 ~ Save the processed rasters
# print 'Getting ready to save the results...'
# # filelist = []
# # date_str = []
# # for file in os.listdir(Landsat_inter_1_dir):
# #     if file.endswith("cleaned.tif"):
# #         file_path = os.path.join(Landsat_inter_1_dir, file)
# #         date_str.append(file[17:25])
# #         filelist.append(file)
# # datelist_str = sorted(list(set(date_str)))
# # filelist = sorted(set(filelist))
#
# for date in np.arange(Raster_dim[0]):
#     OutFile = os.path.join(Landsat_inter_2_dir, Multi_raster_path[date][:-20] + "_ndvi_processed.tif")
#     InFile = os.path.join(Landsat_inter_1_dir, Multi_raster_path[i])
#     raster = Raster_array[date, :, :]
#     print 'Saving ' + OutFile + '...'
#     array2raster(InFile, OutFile, raster)
# print 'code running complete!'
#
# #shutil.rmtree(Landsat_inter_1_dir)
