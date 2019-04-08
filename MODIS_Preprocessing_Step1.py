import os
import shutil

Sitename = 'HREC'
Folder = 'Lower_h08v05'
# Define Directories
Modis_Raw_Path = '/z0/lh349796/Rangeland/MODIS/Download/'+Folder+'/'#'/z0/lh349796/Rangeland/MODIS/Download/HREC' # Where downloaded raw data are saved
Modis_QC_Path = '/z0/lh349796/Rangeland/MODIS/Download/'+Folder+'/'#'/z0/lh349796/Rangeland/MODIS/Download/HREC/'
Modis_QC_Convert_temp_Path = '/z0/lh349796/Rangeland/MODIS/Download/'+Folder+'/Unpacked_temp'
Modis_QC_Convert_Path = '/z0/lh349796/Rangeland/MODIS/Download/'+Folder+'/Unpacked'
Modis_Clip_Path = '/z0/lh349796/Rangeland/MODIS/I_Cliped_Data/'+Sitename
Modis_Inputs_Path = '/z0/lh349796/Rangeland/MODIS/II_Cliped_Repro_Resap_Data/'+Sitename  #Where reprojected, clipped and resampled data are saved
projection = 'EPSG:32610'
Box = '-10644729 4339998 -10625913 4332777' #open both MODIS and Landsat in ArcMap and mannual select the coords
# smallx bigy bigx smally
Landsat_Box = '489189 4313353 498523 4319954' #ulx uly lrx lry:
ProductName = ['MOD','MYD']#ProductName = ['MOD','MYD']

if not os.path.exists(Modis_QC_Convert_Path):
    os.mkdir(Modis_QC_Convert_Path)
if not os.path.exists(Modis_QC_Convert_temp_Path):
    os.mkdir(Modis_QC_Convert_temp_Path)
if not os.path.exists(Modis_Clip_Path):
    os.mkdir(Modis_Clip_Path)
if not os.path.exists(Modis_Inputs_Path):
    os.mkdir(Modis_Inputs_Path)

##~Step1: Clip and convert XXX.hdf to Tiff
# In_Directory = [Modis_QC_Path]
# for directory in In_Directory:
#     for file in os.listdir(directory):
#         if file.endswith('.hdf'):
#             inFile = os.path.join(directory,file)
#             outFile = os.path.join(Modis_QC_Convert_temp_Path,file)
#             outFile_projected = os.path.join(Modis_QC_Convert_Path,file)
#             ###unpack the aerosol quantity bits 00 climatology 01 low 10 medium 11 high
#             os.system('/opt/LDOPE-1.7/bin/unpack_sds_bits -meta -sds=state_1km_1 -bit=6-7 -of=' + outFile + ' ' + inFile)
#             os.system('/opt/LDOPE-1.7/bin/cp_proj_param -ref=' + inFile + ' -of=' + outFile_projected + ' ' + outFile)


for file in os.listdir(Modis_QC_Convert_Path):
    # Access the files in the given directory, clip and save the result in another directory
    for PD in ProductName:
        if file.endswith(".hdf"):
            file_location = os.path.join(Modis_QC_Convert_Path, file)
            ## Clipping
            if file.startswith(PD +'09GA'):
                print "Clipping file " + file
                #QC_Raw = 'HDF4_EOS:EOS_GRID:"' + file_location + '":MODIS_Grid_2D:state_1km_1_bits_6-7'
                QC_Raw = file_location
                QC_Clip = os.path.join(Modis_Clip_Path, PD + '09GA_' + file[9:16] + '_qc.tiff')
                os.system('gdal_translate -of GTiff -projwin ' + Box + ' "' + QC_Raw + '" "' + QC_Clip + '"')
for file in os.listdir(Modis_Raw_Path):
    # Access the files in the given directory, clip and save the result in another directory
    for PD in ProductName:
        if file.endswith(".hdf"):
            file_location = os.path.join(Modis_Raw_Path, file)
            ## Clipping
            if file.startswith(PD + '09GQ'):
                print "Clipping file " + file
                Red_Raw = 'HDF4_EOS:EOS_GRID:"' + file_location + '":MODIS_Grid_2D:sur_refl_b01_1'
                NIR_Raw = 'HDF4_EOS:EOS_GRID:"' + file_location + '":MODIS_Grid_2D:sur_refl_b02_1'
                Red_Clip = os.path.join(Modis_Clip_Path, PD + '09GQ_' + file[9:16] + '_red.tiff')
                NIR_Clip = os.path.join(Modis_Clip_Path, PD + '09GQ_' + file[9:16] + '_nir.tiff')
                os.system('gdal_translate -of GTiff -projwin ' + Box + ' "' + Red_Raw + '" "' + Red_Clip + '"')
                os.system('gdal_translate -of GTiff -projwin ' + Box + ' "' + NIR_Raw + '" "' + NIR_Clip + '"')
print "done!"
## ~Step2: Reproject and Resample to Landsat
In_Directory = [Modis_Clip_Path]
for directory in In_Directory:
    for file in os.listdir(directory):
        if file.endswith('.tiff'):
            current_file = os.path.join(directory, file)
            export_file = os.path.join(Modis_Inputs_Path, file)
            print "Reprojecting and resampling data " + file
            os.system('gdalwarp -overwrite -t_srs "' + projection + '" -tr 30 30 -r near -te ' + Landsat_Box + ' "' + current_file + '" "' + export_file + '"')
            print "done!"

shutil.rmtree(Modis_QC_Convert_temp_Path)
#shutil.rmtree(Modis_Clip_Path)




