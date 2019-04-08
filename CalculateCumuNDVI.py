# This code calculates cumulated APAR using daily APAR using the baseline method and save the beginning of end of growing season
import os
from datetime import date, datetime, timedelta
import gdal,osr
import glob
import logging
import numpy as np

#NDVI_base = 0.3# Instead of 0.3 I'm going to use mean NDVI in Jul, Aug, and Sep
SiteName = ['Hwy36','HREC','SFREC','SJER','SLO']
APAR_dir = '/z0/lh349796/Rangeland/STARFM'
NDVI_dir = '/z0/lh349796/Rangeland/STARFM'
Cumu_APAR_dir = '/z0/lh349796/Rangeland/STARFM'
Grow_start = '1001'#start of water year
Grow_end = '0531'#start of water year
Y_start = 2014
Y_end = 2018

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
for site in SiteName:
    logging.info('Start working on '+ site)
    Working_APAR_dir = os.path.join(APAR_dir,site+'_Fusion','DailyAPAR_LinearSmoothed')
    Working_NDVI_dir = os.path.join(APAR_dir, site + '_Fusion', 'DailyNDVI_LinearSmoothed')
    Working_CumuAPAR_dir = os.path.join(APAR_dir,site+'_Fusion','CumuAPAR_LinearSmoothed')
    if not os.path.exists(Working_CumuAPAR_dir ):
        os.makedirs(Working_CumuAPAR_dir)
    for year in range(Y_start,Y_end):
        d_start = str(year) + Grow_start
        D_start = datetime.strptime(d_start, '%Y%m%d')
        d_end = str(year+1) + Grow_end
        D_end = datetime.strptime(d_end, '%Y%m%d')
        logging.info('Working on ' + d_start + ', '+ d_end + ' ' +site )
        Baseline = 0
        Basline_index = 0
        logging.info('Calculating the Baseline value...')
        for date in perdelta(datetime.strptime(str(year)+'0701', '%Y%m%d'), datetime.strptime(str(year)+'0930', '%Y%m%d'), timedelta(days=1)):
            if glob.glob(os.path.join(Working_NDVI_dir,'LinearDailySmoothed_'+site + '_' + date.strftime('%Y%m%d')+ '_NDVI.tif')):
                Basline_index = Basline_index + 1
                Baseline = Baseline + raster2array(os.path.join(Working_NDVI_dir,'LinearDailySmoothed_'+site + '_' + date.strftime('%Y%m%d')+ '_NDVI.tif'),1)
        Baseline = 1.0*Baseline/Basline_index
        Baseline[Baseline<0.3]=0.3 # keep 0.3 or the summer average, whichever one is higher
        print Baseline.shape
        Maxline = Baseline
        logging.info('Calulating the maximum NDVI value..')
        for date in perdelta(datetime.strptime(str(year+1)+'0301', '%Y%m%d'), datetime.strptime(str(year+1)+'0530', '%Y%m%d'), timedelta(days=1)):
            if glob.glob(os.path.join(Working_NDVI_dir,'LinearDailySmoothed_'+site + '_' + date.strftime('%Y%m%d')+ '_NDVI.tif')):
                NDVI = raster2array(os.path.join(Working_NDVI_dir,'LinearDailySmoothed_'+site + '_' + date.strftime('%Y%m%d')+ '_NDVI.tif'),1)
                Maxline = np.maximum.reduce([Maxline,NDVI])
            else:
                logging.warning('no NDVI exists on '+ date.strftime('%Y%m%d'))
        CumuAPAR = 0
        for date in perdelta(D_start, D_end, timedelta(days=1)):
            if glob.glob(os.path.join(Working_APAR_dir,'LinearDailySmoothed_'+site + '_' + date.strftime('%Y%m%d')+ '_APAR.tif')):
                NDVI = raster2array(os.path.join(Working_NDVI_dir,'LinearDailySmoothed_'+site + '_' + date.strftime('%Y%m%d')+ '_NDVI.tif'),1)
                Baseline[np.equal(NDVI,Maxline)] = Maxline[np.equal(NDVI,Maxline)]
                NDVI[NDVI >= Baseline] = 1
                NDVI[NDVI < Baseline] = 0
                APAR = raster2array(os.path.join(Working_APAR_dir,'LinearDailySmoothed_'+site + '_' + date.strftime('%Y%m%d')+ '_APAR.tif'),1)
                CumuAPAR = CumuAPAR + NDVI*APAR
                Paramfile = os.path.join(Working_NDVI_dir,'LinearDailySmoothed_'+site + '_' + date.strftime('%Y%m%d')+ '_NDVI.tif')
                Outfile = os.path.join(Working_CumuAPAR_dir,'LinearDailySmoothed_'+site + '_' + date.strftime('%Y%m%d')+ '_CumuAPAR.tif')
                array2raster(Paramfile,Outfile,CumuAPAR)
                Outfile = os.path.join(Working_CumuAPAR_dir,'LinearDailySmoothed_'+site + '_' + date.strftime('%Y%m%d')+ '_Baseline.tif')
                array2raster(Paramfile, Outfile, NDVI)
            else:
                logging.warning('No APAR found for '+ date.strftime('%Y%m%d') + ' '+site)







