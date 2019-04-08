# this code fuse the Landsat and the MODIS data and make Landsat data with 16 day interval
# the input landsat image series has to be have 16-day interval
# The algorithm only works when the input Landsat series starts and end with a
# 'full' image
import gdal
import os
import osr
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
# import shutil
import glob
import logging
SiteNames = ['SLO']#,'SFREC','SLO','SJER']#'Hwy36'
PathRow = '04*03*'
# preset data fusion parameters
Moving_Window_Size = 25 # has to be an odd number and bigger than 17 (500/30 = 16.6666)
Num_Class = 5 # number of class being classified in the unsupervised classification before data fusion
for SiteName in SiteNames:
    # where the input coarse resolution data is saved
    PS_Ori_dir = '/z0/lh349796/Rangeland/MODIS/IIII_Input_Data/'+ SiteName #'/z0/lh349796/Rangeland/MODIS/IIII_Input_Data/HREC_16_17'#'/z0/lh349796/Rangeland/PlanetScope/IIII_Interpolated_data/Camatta/2017growing_30cm/Interpolated_final_cubic'
    # where the input fine resolution data is saved
    Drone_Ori_dir = '/z0/lh349796/Rangeland/landsat_data/' + SiteName + '/II_Cfmasked_data/Landsat8_full'#'/z0/lh349796/Rangeland/landsat_data/HREC/II_Cfmasked_data/Landsat8_2018'#'/z0/lh349796/Rangeland/UAS/Camatta/II_Clipped_Resampled_NDVI_30cm/2017'
    # where the fused data is saved
    Drone_Inte_dir = '/z0/lh349796/Rangeland/STARFM/' + SiteName + '_Fusion/SSTARFM_full'#'/z0/lh349796/Rangeland/landsat_data/HREC/III_Processed_data/Landsat8_2018/SSTARFM_NDVI'#'C:/Users/Grace Liu/Box Sync/research/sUAS/JastroProposal/IIII_Processed_data/30cm_CModel/NDVI_RapidEye_Intepolated_I'
    # where the output smoothed data is saved
    Drone_Smooth_dir = '/z0/lh349796/Rangeland/STARFM/' + SiteName + '_Fusion/SSTARFM_full_Smoothed'#'/z0/lh349796/Rangeland/landsat_data/HREC/III_Processed_data/Landsat8_2018/SSTARFM_NDVI_smoothed'#'C:/Users/Grace Liu/Box Sync/research/sUAS/JastroProposal/IIII_Processed_data/30cm_CModel/NDVI_RapidEye_Smoothed_II'
    if not os.path.exists(Drone_Smooth_dir):
        os.makedirs(Drone_Smooth_dir)
    if not os.path.exists(Drone_Inte_dir):
        os.makedirs(Drone_Inte_dir)
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
    def nearest(items, pivot):
        '''
        This function will return the datetime in items which is the closest to the date pivot
        '''
        return min(items, key=lambda x: abs(x - pivot))
    def savitzky_golay_filtering(timeseries, wnds=[11, 7], orders=[2, 4], debug=True):
        '''
        A simple method for reconstructing a high quality NDVI time-series data set
        based on the Savitzky-Golay filter", Jin Chen et al. 2004
        '''
        interp_ts = pd.Series(timeseries)
        interp_ts = interp_ts.interpolate(method='linear', limit=14)
        smooth_ts = interp_ts
        wnd, order = wnds[0], orders[0]
        F = 1e8
        W = None
        it = 0
        while True:
            smoother_ts = savgol_filter(smooth_ts, window_length=wnd, polyorder=order)
            diff = smoother_ts - interp_ts
            sign = diff > 0
            if W is None:
                W = 1 - np.abs(diff) / np.max(np.abs(diff)) * sign
                wnd, order = wnds[1], orders[1]
            fitting_score = np.sum(np.abs(diff) * W)
            print it, ' : ', fitting_score
            if fitting_score > F:
                break
            else:
                F = fitting_score
                it += 1
            smooth_ts = smoother_ts * sign + interp_ts * (1 - sign)
        if debug:
            return smooth_ts, interp_ts
        return smooth_ts
    def perdelta(start, end, delta):
        '''
        for generating a list of date with delta interval
        '''
        curr = start
        while curr <= end:
            yield curr
            curr += delta
    def ExtractMovingWindow(Wsize,Raster,RR,CC):
        '''
        a function that extracts a Wsize*Wsize raster
        centered at [Row,Col] from the original raster
        '''
        if Wsize % 2 == 0:
            return 0
        Length = (Wsize-1)/2
        if RR < Length:
            if CC < Length:
                Out_Raster = Raster[0:(RR + Length + 1), 0:(CC + Length + 1)]
            else:
                Out_Raster = Raster[0:(RR + Length + 1), (CC - Length):(CC + Length + 1)]
        elif CC < Length:
            Out_Raster = Raster[(RR - Length):(RR + Length + 1), 0:(CC + Length + 1)]
        else:
            Out_Raster = Raster[(RR - Length):(RR + Length + 1), (CC - Length):(CC + Length + 1)]
        return Out_Raster
    def ReturnMovingWindowCenter(Wsize,Raster,RR,CC):
        '''
        a function that extracts the center pixel of a MW generated by  ExtractMovingWindow
        '''
        if Wsize % 2 == 0:
            return 0
        Length = (Wsize-1)/2
        if RR < Length:
            if CC < Length:
                Out_Pixel = Raster[RR, CC]
            else:
                Out_Pixel = Raster[RR, Length]
        elif CC < Length:
            Out_Pixel = Raster[Length, CC]
        else:
            Out_Pixel = Raster[Length, Length]
        return Out_Pixel
    def load_nearest_img(type, dlist, d, before = True):
        '''
        return the nearest available landsat or modis image
        :param type: string 'Landsat' or 'MODIS'
        :param datelist: a datelist to loop through for finding the nearest date
        :param d: datetime indicating the base date
        :param before: boolean finding the nearest date before or after the base date
        :return: image in array format of the nearest date, and the date of the image
        '''
        if before:
            # find a earlier image closest to the target date
            if d <= datelist[0]:
                print 'load_nearest_img error'
                return 0,0
            date = nearest([i for i in dlist if i < d], d)
            if type == 'Landsat':
                Base_drone_filename_before = glob.glob(
                    os.path.join(Drone_Ori_dir, 'LC08_L1*_' + PathRow + '_' + date.strftime('%Y%m%d') + '_*ndvi_cleaned.tif'))[0]
                Base_image = raster2array(os.path.join(Drone_Ori_dir, Base_drone_filename_before), 1)
            elif type == 'MODIS':
                Base_rapideye_filename_before = glob.glob(
                    os.path.join(PS_Ori_dir, 'MIX09GQ_' + date.strftime('%Y%j') + '*_ndvi.tiff'))[0]
                Base_image = raster2array(Base_rapideye_filename_before, 1)
            else:
                print 'error in input type, neither Landsat nor MODIS'
                return 0
        else:
            if d >= datelist[-1]:
                print 'load_nearest_img error'
                return 0,0
            # find a later landsat and MODIS image closest to the target date
            date = nearest([i for i in dlist if i > d], d)
            if type == 'Landsat':
                Base_drone_filename_after = glob.glob(
                os.path.join(Drone_Ori_dir, 'LC08_L1*_' + PathRow + '_' + date.strftime('%Y%m%d') + '_*ndvi_cleaned.tif'))[0]
                Base_image = raster2array(os.path.join(Drone_Ori_dir, Base_drone_filename_after), 1)
            elif type == 'MODIS':
                Base_rapideye_filename_after = glob.glob(
                    os.path.join(PS_Ori_dir, 'MIX09GQ_' + date.strftime('%Y%j') + '*_ndvi.tiff'))[0]
                Base_image = raster2array(Base_rapideye_filename_after, 1)
            else:
                print 'error in input type, neither Landsat nor MODIS'
                return 0
        return Base_image,date
    # ##################################################
    ### for logging
    # set up logging to file - see previous section for more details
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
    # ##################################################
    ## get dates for the whole RapidEye time series
    date_str = []
    for file in os.listdir(PS_Ori_dir):
        if file.endswith('ndvi.tiff'):
            date_str.append(file.split('_')[1])
    datelist_str = sorted(list(set(date_str)))
    ## convert datelist_str to a datetime object
    datelist = [] # the MODIS data datelist
    for date_var in datelist_str:
        temp = datetime.strptime(date_var, '%Y%j')
        datelist.append(date(temp.year, temp.month, temp.day))
    ## get dates of available drone images
    date_str_drone = []
    for file in os.listdir(Drone_Ori_dir):
        if file.endswith("cleaned.tif"):
            date_str_drone.append(file[17:25])
    ## convert date_str_drone to a datetime object
    datelist_drone = []
    for date_var in date_str_drone:
        temp = datetime.strptime(date_var, '%Y%m%d')
        datelist_drone.append(date(temp.year, temp.month, temp.day))
    datelist_drone = sorted(datelist_drone) # datelist of Landsat data
    datelist_predicted = perdelta(datelist_drone[0], datelist_drone[-1], timedelta(days=16)) # a datelist with 16 d interval from the Landsat datelist
    logging.info('Getting ready to interpolate and smooth...')
    logger1 = logging.getLogger('Date_loop')
    logger2 = logging.getLogger('Pixel_loop')
    logger3 = logging.getLogger('Nearest_loop')
    logger4 = logging.getLogger('Smoothing')
    ### Interpolation and Smoothing
    for d in perdelta(datelist_drone[5], datelist_drone[-2], timedelta(days=16)):#loop through the 16 d datelist starting from the second scene
        if (d-timedelta(days=15)) < datelist[0]: # if d is not 15 days later than the first element in MODIS datelist, we cant do anything
            # this is because the landsat images series is longer than the MODIS image series
            continue
        elif (d+timedelta(days=15)) > datelist[-1]:
            break
        logger1.info('Fusing the Landsat image on date ' + d.strftime('%m/%d/%Y'))
        # the file name of Landsat data on the predicted date
        Target_drone_filename = 'LC08_L1*_' + PathRow + '_' + d.strftime('%Y%m%d') + '_*ndvi_cleaned.tif'
        Target_rapideye_filename = glob.glob(os.path.join(PS_Ori_dir, 'MIX09GQ_' + d.strftime('%Y%j') + '_ndvi.tiff'))[0]
        Target_rapideye_image = raster2array(os.path.join(PS_Ori_dir, Target_rapideye_filename), 1)
        Saving_drone_filename = 'SSTARFM_HREC_' + d.strftime('%Y%m%d') + '_NDVI.tif'
        OutFile_dir = os.path.join(Drone_Inte_dir,Saving_drone_filename)
        ### Step 1~ check if we have a existing Landsat file on d
        if len(glob.glob(os.path.join(Drone_Ori_dir, Target_drone_filename))) > 0:# if exist, read it in
            logger2.info('Landsat image for date ' + d.strftime('%m/%d/%Y') + ' exists...')
            Target_drone_image = raster2array(glob.glob(os.path.join(Drone_Ori_dir, Target_drone_filename))[0], 1)
            ### Step~2: check if a window has missing data in the target Landsat scene
            if np.count_nonzero(np.isnan(Target_drone_image)) <= 10: #check if the image has missing data in the target Landsat scene np.count_nonzero(np.isnan(data))
                logger2.info('Landsat image for date '+ d.strftime('%m/%d/%Y') + ' exists...skipping...' +  str(np.count_nonzero(np.isnan(Target_drone_image))))
                os.system('cp ' + os.path.join(Drone_Ori_dir, Target_drone_filename) + ' ' + os.path.join(Drone_Inte_dir, Saving_drone_filename))
                continue
        elif len(glob.glob(os.path.join(Drone_Ori_dir, Target_drone_filename))) == 0: # if does not exist, create a new one
            logger2.info('Landsat image for date ' + d.strftime('%m/%d/%Y') + ' does not exist...')
            Target_drone_image = raster2array(os.path.join(PS_Ori_dir, Target_rapideye_filename), 1)
            Target_drone_image[Target_drone_image != 9999] = np.nan # create an scene with all zeros
        # loop through the windows in the targeted scene
        for Row in range(0, Target_drone_image.shape[0]):  # fuse the pixels one by one range(100, 150):#, Moving_Window_Size):  #
            for Col in range(0, Target_drone_image.shape[1]): # range(0, 50):#, Moving_Window_Size):  #
                if  ~np.isnan(Target_drone_image[Row,Col]):  # go to the next MW if the current one is not polluted
                    continue
                # find a earlier landsat and MODIS image closest to the target date
                Base_drone_image_before, Drone_before_d = load_nearest_img('Landsat', datelist_drone, d, before=True)
                Base_rapideye_image_before, Rap_before_d = load_nearest_img('MODIS', datelist_drone, d, before=True)
                # find a later landsat and MODIS image closest to the target date
                Base_drone_image_after, Drone_after_d = load_nearest_img('Landsat', datelist_drone, d, before=False)
                Base_rapideye_image_after, Rap_after_d = load_nearest_img('MODIS', datelist_drone, d, before=False)
                logger2.info('Fusing MW centered at Row ' + str(Row) + ' Col ' + str(Col) + ' for date ' + str(d) + '...')
                MW_target = ExtractMovingWindow(Moving_Window_Size, Target_drone_image, Row, Col)
                # if np.count_nonzero(np.isnan(MW_target)) == 0:  # go to the next MW if the current one is not polluted
                #     continue
                MW_before = ExtractMovingWindow(Moving_Window_Size, Base_drone_image_before, Row, Col)
                MW_after = ExtractMovingWindow(Moving_Window_Size, Base_drone_image_after, Row, Col)
                #logger3.debug('MW_before'+ str(MW_before.shape))
                ### Step~3: check if a window has two scenes (before and after) of available data in the input landsat time series
                # check if there's missing values in the MW for the before and after drone images
                if np.count_nonzero(np.isnan(MW_before)) > 15:
                    #logger2.info('the nearest ('+ Drone_before_d.strftime('%Y%m%d') +') window has missing values')
                    # if yes, find the nearest before image with no missing value in the MW
                    datelist_drone_new = datelist_drone[:]
                    while(1):
                        #logger3.info('initial date ' + Drone_before_d.strftime('%Y%m%d'))
                        datelist_drone_new = [x for x in datelist_drone_new if x != Drone_before_d]
                        Base_drone_image_before, Drone_before_d = load_nearest_img('Landsat', datelist_drone_new, d,before=True)
                        MW_before = ExtractMovingWindow(Moving_Window_Size, Base_drone_image_before, Row, Col)
                        Base_rapideye_image_before, Rap_before_d = load_nearest_img('MODIS', datelist_drone_new, d,before=True)
                        #logger3.info('Number of missing value pixel in the MW on date '+ Drone_before_d.strftime('%Y%m%d') + ' : ' + str(np.count_nonzero(np.isnan(MW_before))))
                        if np.count_nonzero(np.isnan(MW_before)) <= 15 or Drone_before_d == 0:
                            #logger3.info('output date' + Drone_before_d.strftime('%Y%m%d'))
                            break
                if np.count_nonzero(np.isnan(MW_after)) > 15:
                    #logger2.info('the nearest (' + Drone_after_d.strftime('%Y%m%d') + ') window has missing values')
                    # if yes, find the nearest after image with no missing value in the MW
                    datelist_drone_new = datelist_drone[:]
                    while (1):
                        #logger3.info('initial date '+ Drone_after_d.strftime('%Y%m%d'))
                        datelist_drone_new = [x for x in datelist_drone_new if x != Drone_after_d]
                        Base_drone_image_after, Drone_after_d = load_nearest_img('Landsat', datelist_drone_new, d,before=False)
                        MW_after = ExtractMovingWindow(Moving_Window_Size, Base_drone_image_after, Row, Col)
                        Base_rapideye_image_after, Rap_after_d = load_nearest_img('MODIS', datelist_drone_new, d,before=False)
                        #logger3.info('Number of missing value pixel in the MW on date ' + Drone_after_d.strftime('%Y%m%d') + ' : ' + str(np.count_nonzero(np.isnan(MW_after))))
                        if np.count_nonzero(np.isnan(MW_after)) <= 15 or Drone_after_d == 0:
                            #logger3.info('output date' + Drone_after_d.strftime('%Y%m%d'))
                            break
                # if couldn't both before and after image, continue to the next MW:
                if  Drone_after_d == 0 or Drone_before_d == 0:
                    logger2.warning('no available MW_after or/and MW_before was found for all images!!')
                    logger2.info('Continuing to the next MW...')
                    continue
                else:
                    ### Step~4: if everything exists, start data fusion
                    MW_Rap_target = ExtractMovingWindow(Moving_Window_Size, Target_rapideye_image, Row, Col)
                    ### only keep the pixels that has similar NDVI (+/-0.1) value with the center pixel
                    CenterValue = ReturnMovingWindowCenter(Moving_Window_Size, MW_Rap_target, Row, Col)
                    #MW_Rap_target[MW_Rap_target > (CenterValue + 0.05)] = np.nan
                    #MW_Rap_target[MW_Rap_target < (CenterValue - 0.05)] = np.nan
                    # if len(np.unique(MW_Rap_target[~np.isnan(MW_Rap_target)]))<=1: in case if only the center pixel is kept
                    #     MW_Rap_target = ExtractMovingWindow(Moving_Window_Size, Target_rapideye_image, Row, Col)
                    #     Values = np.unique(MW_Rap_target)
                    L3 = np.reshape(MW_Rap_target, np.product(MW_Rap_target.shape))
                    MW_Rap_before = ExtractMovingWindow(Moving_Window_Size, Base_rapideye_image_before, Row, Col)
                    #MW_Rap_before[np.isnan(MW_Rap_target)] = np.nan
                    L1 = np.reshape(MW_Rap_before, np.product(MW_Rap_before.shape))  # reshaping the original raster to vector
                    MW_Rap_after = ExtractMovingWindow(Moving_Window_Size, Base_rapideye_image_after, Row, Col)
                    #MW_Rap_after[np.isnan(MW_Rap_target)] = np.nan
                    L2 = np.reshape(MW_Rap_after, np.product(MW_Rap_after.shape))
                    logger2.info('fusing date ' + d.strftime('%Y-%m-%d') + ' using date '+ Rap_before_d.strftime('%Y-%m-%d') + ' and date ' + Rap_after_d.strftime('%Y-%m-%d'))
                    # calculate the coorelation between before-target and after-target
                    df13 = pd.DataFrame(np.column_stack((L1, L3)), columns=list('AB'))
                    df13[df13 == 0] = np.nan
                    df23 = pd.DataFrame(np.column_stack((L2, L3)), columns=list('AB'))
                    df23[df23 == 0] = np.nan
                    Corr13 = df13.corr()['A']['B']
                    Corr23 = df23.corr()['A']['B']
                    if Corr13 < 0:
                        Corr13 = 0
                        W13 = Corr13 / (Corr23 + Corr13)
                        W23 = 1 - W13
                        if Corr23 < 0:
                            Corr23 = 0
                            W13 = 0
                            W23 = 0
                    elif Corr23 < 0:
                        Corr23 = 0
                        W13 = Corr13 / (Corr23 + Corr13)
                        W23 = 1 - W13
                    else:
                        W13 = Corr13 / (Corr23 + Corr13)
                        W23 = 1 - W13
                    ## 05312018 update: add time weight
                    T13 = 1 - 1.0 * (d - Rap_before_d).days / (Rap_after_d - Rap_before_d).days
                    T23 = 1 - 1.0 * (Rap_after_d - d).days / (Rap_after_d - Rap_before_d).days
                    W13 = W13 + T13
                    W23 = W23 + T23
                    W13 = W13 / (W13 + W23)
                    W23 = 1 - W13
                    logger3.info('w13 = ' + str(W13) + ' w23 = ' + str(W23))
                    #size = (Moving_Window_Size-1)/2
                    # Target_drone_image[Row-size:Row+size+1 ,Col-size :Col+size+1] = W13 * (MW_Rap_target + MW_before - MW_Rap_before) + W23 * (
                    #         MW_Rap_target + MW_after - MW_Rap_after)
                    Predicted_MW = (W13 * (MW_Rap_target + MW_before - MW_Rap_before) + W23 * (MW_Rap_target + MW_after - MW_Rap_after))
                    Target_drone_image[Row , Col] = ReturnMovingWindowCenter(Moving_Window_Size,Predicted_MW,Row,Col)

        logger1.info('finished fusing the image on date ' + d.strftime('%Y-%m-%d'))
        logger1.info('Saving the result as '+ Saving_drone_filename)
        logger1.info('Number of unfused pixels: ' + str(np.count_nonzero(Target_drone_image == 0)))
        array2raster(os.path.join(PS_Ori_dir, Target_rapideye_filename),
                                     os.path.join(Drone_Inte_dir, Saving_drone_filename),
                                     Target_drone_image)
    logging.info('data fusion complete!')
    logging.info('geting ready to smoothing')
    #################################################
    #################################################
    # Smoothing I'm not going to do the smoothing because its a 16d ts and smoothing will resulting in deviation of NDVI value on the days with avaliable data
    logging.info('Smoothing Started...')
    logging.info('converting the TIFFs to numpy array...')
    Multi_raster_path = []
    for raster in os.listdir(Drone_Inte_dir):
        if raster.endswith('NDVI.tif'):
            raster_path = os.path.join(Drone_Inte_dir, raster)
            Multi_raster_path.append(raster_path)
    Multi_raster_path = sorted(Multi_raster_path)
    Multi_raster = []
    for raster_path in Multi_raster_path:
        logger4.info('reading in file' + raster_path)
        Multi_raster.append(raster2array(raster_path, 1))
    Raster_array = np.array(Multi_raster)
    Raster_dim = Raster_array.shape
    logging.info('Conversion finished!')
    for row in np.arange(Raster_dim[1]):
        for col in np.arange(Raster_dim[2]):
            logger4.info('applying the SavGol filter to pixel (' + str(row) + ',' + str(col) + ')...')
            time_series = pd.Series(Raster_array[:, row, col])
            ## apply SavGol filter
            #time_series_savgol = savgol_filter(time_series, window_length=5, polyorder=2)
            time_series_savgol = savitzky_golay_filtering(time_series, wnds=[11, 7], orders=[2, 4], debug=False)
            Raster_array[:, row, col] = time_series_savgol
            logger4.info('interpolation and smoothing for pixel (' + str(row) + ',' + str(col) + ') is finished!')

    logging.info('interpolation and smoothing complete!')
    # # ts = pd.DataFrame(Raster_array[:, row, col])
    # # ts.to_csv(os.path.join(Drone_Smooth_dir,'R'+str(row)+'C'+str(col)+'.csv'))
    #
    # saving the results
    filelist = []
    for file in os.listdir(Drone_Inte_dir):
        if file.endswith("NDVI.tif"):
            file_path = os.path.join(Drone_Inte_dir, file)
            filelist.append(file)
    filelist = sorted(set(filelist))
    for date_var in np.arange(Raster_dim[0]):
        OutFile = os.path.join(Drone_Smooth_dir, filelist[date_var])
        InFile = os.path.join(Drone_Inte_dir, filelist[1])
        raster = Raster_array[date_var, :, :]
        logger4.info('Saving ' + OutFile + '...')
        array2raster(InFile, OutFile, raster)




    ### Step~4: if everything exists, start data fusion
    # Row = 129
    # Col = 6
    # Rap_before_d = datetime.strptime('2016282', '%Y%j')
    # Rap_after_d = datetime.strptime('2017060', '%Y%j')
    # d = datetime.strptime('2016298', '%Y%j')
    # Target_rapideye_image = raster2array('D:\\Box Sync\\research\\Satellite\\Data\\Datafusion\\MODIS\\MIX09GQ_2016298_ndvi.tiff',1)
    # Base_rapideye_image_before = raster2array('D:\\Box Sync\\research\\Satellite\\Data\\Datafusion\\MODIS\\MIX09GQ_2016282_ndvi.tiff',1)
    # Base_rapideye_image_after = raster2array('D:\\Box Sync\\research\\Satellite\\Data\\Datafusion\\MODIS\\MIX09GQ_2017060_ndvi.tiff',1)
    # MW_Rap_target = ExtractMovingWindow(Moving_Window_Size, Target_rapideye_image, Row, Col)
    # ### only keep the pixels that has similar NDVI (+/-0.1) value with the center pixel
    # CenterValue = ReturnMovingWindowCenter(Moving_Window_Size, MW_Rap_target, Row, Col)
    # MW_Rap_target[MW_Rap_target > (CenterValue + 0.05)] = np.nan
    # MW_Rap_target[MW_Rap_target < (CenterValue - 0.05)] = np.nan
    # len(np.unique(MW_Rap_target[~np.isnan(MW_Rap_target)]))
    # # if len(np.unique(MW_Rap_target[~np.isnan(MW_Rap_target)]))<=1: in case if only the center pixel is kept
    # #     MW_Rap_target = ExtractMovingWindow(Moving_Window_Size, Target_rapideye_image, Row, Col)
    # #     Values = np.unique(MW_Rap_target)
    # L3 = np.reshape(MW_Rap_target, np.product(MW_Rap_target.shape))
    # MW_Rap_before = ExtractMovingWindow(Moving_Window_Size, Base_rapideye_image_before, Row, Col)
    # MW_Rap_before[np.isnan(MW_Rap_target)] = np.nan
    # L1 = np.reshape(MW_Rap_before, np.product(MW_Rap_before.shape))  # reshaping the original raster to vector
    # MW_Rap_after = ExtractMovingWindow(Moving_Window_Size, Base_rapideye_image_after, Row, Col)
    # MW_Rap_after[np.isnan(MW_Rap_target)] = np.nan
    # L2 = np.reshape(MW_Rap_after, np.product(MW_Rap_after.shape))
    # #logger2.info('fusing date ' + d.strftime('%Y-%m-%d') + ' using date '+ Rap_before_d.strftime('%Y-%m-%d') + ' and date ' + Rap_after_d.strftime('%Y-%m-%d'))
    # # calculate the coorelation between before-target and after-target
    # df13 = pd.DataFrame(np.column_stack((L1, L3)), columns=list('AB'))
    # df13[df13 == 0] = np.nan
    # df23 = pd.DataFrame(np.column_stack((L2, L3)), columns=list('AB'))
    # df23[df23 == 0] = np.nan
    # Corr13 = df13.corr()['A']['B']
    # Corr23 = df23.corr()['A']['B']
    # if Corr13<0:
    #     Corr13=0
    # if Corr23 < 0:
    #     Corr23 = 0
    # W13 = Corr13 / (Corr23 + Corr13)
    # W23 = 1 - W13
    # ## 05312018 update: add time weight
    # T13 = 1 - 1.0 * (d - Rap_before_d).days / (Rap_after_d - Rap_before_d).days
    # T23 = 1 - 1.0 * (Rap_after_d - d).days / (Rap_after_d - Rap_before_d).days
    # W13 = W13 + T13
    # W23 = W23 + T23
    # W13 = W13 / (W13 + W23)
    # W23 = 1 - W13