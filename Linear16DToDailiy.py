# This code takes in the 16D SSTARFM interpolated NDVI and interpolate in to daily NDVI
# Then calculates daily APAR
import gdal, osr
import os
import glob
import numpy as np
import pandas as pd
import logging
from datetime import date, datetime, timedelta
SiteNames = ['SLO']#,'HREC','SFREC','SJER','Hwy36']
NDVI_16d_dir = '/z0/lh349796/Rangeland/STARFM'
CIMIS_Rs_dir = '/z0/lh349796/Rangeland/CIMIS_Spatial/III_Tiff_Clipped/Rs'
NDVI_daily_dir = '/z0/lh349796/Rangeland/STARFM'
APAR_daily_dir = '/z0/lh349796/Rangeland/STARFM'
# Define functions
def raster2array(rasterfn, i):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(i)
    return band.ReadAsArray()
def array2raster(rasterfn, newRasterfn, array):
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
def perdelta(start, end, delta):
    '''
    for generating a list of date with delta interval
    '''
    curr = start
    while curr < end:
        yield curr
        curr += delta
# Setting up the logging file
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename= os.path.join(os.path.dirname(os.path.realpath(__file__)),os.path.basename(__file__)[:-3] + '.log'),
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
logging.info('Getting ready to interpolate and smooth...')
logger1 = logging.getLogger('NDVI')
logger2 = logging.getLogger('APAR')
Datelist = []
for site in SiteNames:
    logging.info('Started working on '+ site + '...')
    Working_dir = os.path.join(NDVI_16d_dir,site+'_Fusion/SSTARFM_full_Smoothed')
    Working_CIMIS_dir = os.path.join(CIMIS_Rs_dir,site)
    Working_outputNDVI_dir = os.path.join(NDVI_daily_dir,site+'_Fusion/DailyNDVI_LinearSmoothed')
    Working_outputAPAR_dir = os.path.join(NDVI_daily_dir,site+'_Fusion/DailyAPAR_LinearSmoothed')
    if not os.path.exists(Working_outputNDVI_dir):
        os.makedirs(Working_outputNDVI_dir)
    if not os.path.exists(Working_outputAPAR_dir):
        os.makedirs(Working_outputAPAR_dir)
    Datelist = []
    for file in os.listdir(Working_dir):
        if file.endswith('.tif'):
            d = datetime.strptime(file.split('_')[2],'%Y%m%d')
            Datelist.append(date(d.year, d.month, d.day))
    Datelist = sorted(set(Datelist))
    Datelist_1 = Datelist[:-1]
    Datelist_2 = Datelist[1:]
    RasterStack = []
    for d1,d2 in zip(Datelist_1,Datelist_2):
        for d in perdelta(d1, d2, timedelta(days=1)):
            logger1.info('working on '+ d.strftime('%Y%m%d') + ' using ' + d1.strftime('%Y%m%d') + ' and ' + d2.strftime('%Y%m%d') + '...'+ site)
            file = glob.glob(os.path.join(Working_dir,'*'+d.strftime('%Y%m%d')+'_NDVI.tif'))
            if(file):
                Raster = raster2array(file[0],1)
            else:
                Raster = raster2array(glob.glob(os.path.join(Working_dir,'*'+d1.strftime('%Y%m%d')+'_NDVI.tif'))[0], 1)
                Raster[Raster != np.nan] = np.nan
            RasterStack.append(Raster)
    RasterStack = np.array(RasterStack)
    logger1.info('interpolating '+site+' pixel by pixel...')
    for row in np.arange(RasterStack.shape[1]):
        for col in np.arange(RasterStack.shape[2]):
            time_series = pd.Series(RasterStack[:, row, col])
            ## apply SavGol filter
            time_series_interp = time_series.interpolate(method="linear")
            RasterStack[:, row, col] = time_series_interp
    Output_Datelist = [Datelist[0] + timedelta(days=x) for x in range(0, (Datelist[-1]-Datelist[0]).days)]
    for layer in np.arange(RasterStack.shape[0]):
         OutFile = os.path.join(Working_outputNDVI_dir, 'LinearDailySmoothed_'+site + '_' + Output_Datelist[layer].strftime('%Y%m%d') + '_NDVI.tif')
         InFile = glob.glob(os.path.join(Working_dir,'*'+d1.strftime('%Y%m%d')+'_NDVI.tif'))[0]
         logger1.info('Saving NDVI' + OutFile + '...')
         ndvi_inte = RasterStack[layer, :, :]
         array2raster(InFile, OutFile, ndvi_inte)
         ## find the correspoinding CIMIS Spatial
         if glob.glob(os.path.join(Working_CIMIS_dir, Output_Datelist[layer].strftime('%Y%m%d') +'Rs.tif')):
             cimis_rs = raster2array(glob.glob(os.path.join(Working_CIMIS_dir,Output_Datelist[layer].strftime('%Y%m%d')+'Rs.tif'))[0],1)
             apar = cimis_rs * 0.5 * ndvi_inte
             OutFile = os.path.join(Working_outputAPAR_dir,'LinearDailySmoothed_' + site + '_' + Output_Datelist[layer].strftime('%Y%m%d') + '_APAR.tif')
             logger2.info('Saving APAR' + OutFile + '...')
             array2raster(InFile, OutFile, apar)
         else:
             logger2.info('No CIMIS Rs exists for ' + Output_Datelist[layer].strftime('%Y%m%d'))

logging.info('done!')
