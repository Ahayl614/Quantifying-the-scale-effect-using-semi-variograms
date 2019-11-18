# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:53:53 2017

@author: CHENL
"""

import pandas as pd
import pickle as pkl

from pyproj import Proj, transform
import numpy as np
from scipy.stats import skew
import math,random

import time
import os
import matplotlib.pyplot as plt
from matplotlib import colors

import getNSR


def get_init_param(cityName):
    """
    return data preprocessing parameters for each city

    """
    odir = '../WEIBO_CHECKINS_2014/'
    cityName = cityName.upper()
    if cityName == 'BJ':
        gridLeft = 431539.613100;
        gridBottom = 4400490.445800;
        proj = 'epsg:32650'
        odir = 'WEIBO_CHECKINS_BEIJING.csv'
        titleName = 'Beijing'
    elif cityName == 'SH':
        gridLeft = 341578.279641;
        gridBottom = 3442811.112736;
        proj = 'epsg:32651'
        odir = 'WEIBO_CHECKINS_SHANGHAI.csv'
        titleName = 'Shanghai'
    elif cityName == 'CD':
        gridLeft = 395910.414867;
        gridBottom = 3380011.996190;
        proj = 'epsg:32648'
        odir = 'WEIBO_CHECKINS_CHENGDU.csv'
        titleName = 'Chengdu'
    elif cityName == 'WH':
        gridLeft = 226962.988499;
        gridBottom = 3369428.592151;
        proj = 'epsg:32650'
        odir = 'WEIBO_CHECKINS_WUHAN.csv'
        titleName = 'Wuhan'
    else:
        print ('There is no pre-processing information for '+str(cityName))
    return gridLeft, gridBottom, proj, odir, titleName

def update_study_area(RegionLeft,RegionBottom):
    rho = random.uniform(0, 100)
    theta = random.uniform(0, 0.5 * math.pi)
    delta_x = rho * math.cos(theta)
    delta_y = rho * math.sin(theta)
    RegionLeft = RegionLeft + delta_x
    RegionBottom = RegionBottom + delta_y
    return RegionLeft,RegionBottom,round(delta_x,2),round(delta_y,2)

def proj_trans(lon, lat, proj):
    p1 = Proj(init='epsg:4326')   # Geographic coordinate system WGS1984
    p2 = Proj(init=proj)          # Projected coordinate system WGS_1984_UTM_Zone_50N(bj/wh)

    lon_val = lon.values
    lat_val = lat.values
    x1, y1 = p1(lon_val, lat_val)
    x2, y2 = transform(p1, p2, x1, y1, radians=True)
    return x2, y2

def get_grid_id(cellSize, gridNum, x, y,gridLeft,gridBottom):
    xidArr = np.ceil((x - gridLeft) / cellSize);
    yidArr = np.ceil((y - gridBottom) / cellSize);
    outIndex = np.array([], dtype=np.bool)

    for i in range(0, len(xidArr)):
        if (xidArr[i] < 1) | (xidArr[i] > gridNum):
            outIndex = np.append(outIndex, True)
        else:
            outIndex = np.append(outIndex, False)
    for j in range(0, len(yidArr)):
        if (yidArr[j] < 1) | (yidArr[j] > gridNum):
            outIndex[j] = True

    grid_id = (yidArr - 1) * gridNum + xidArr - 1

    grid_id[outIndex] = -1
    # grid_id(0,gridNum*gridNum-1 = 899)
    totalNum = gridNum * gridNum - 1
    grid_id[(grid_id < 0) | (grid_id > totalNum)] = -1
    grid_id = grid_id.astype(np.int)
    return grid_id

def get_grid_inf(gridNum):
    gridid_arr = np.arange(gridNum * gridNum)
    xid_arr = gridid_arr // gridNum + 1
    yid_arr = gridid_arr % gridNum + 1
    return gridid_arr, xid_arr, yid_arr

def caculate_variance():
    start = time.time()
    count = 0
    lags_list = []
    dsquare_list = []
    for i in range(1, gridNum * gridNum):
        count += i
        o = gridinf.iloc[i]
        d_arr = gridinf.iloc[:i]
        lags = np.sqrt(pow((d_arr.gridx.tolist() - o.gridx), 2) + pow((d_arr.gridy.tolist() - o.gridy), 2)) * csize

        dif = abs(sdd_df.iloc[:i, 0] - sdd_df.iloc[i, 0])
        dif_square = pow(dif, 2)
        lags_list.extend(lags)
        dsquare_list.extend(dif_square)

        if (count % 100000 == 0):
            end = time.time()
            print(count / 100000, 'timeconsum:', end - start)
            start = time.time()
    return lags_list, dsquare_list

def _init():
    global MAXRANGE
    global cityNames
    global csizeList
    global fileNoList
    MAXRANGE = 30000
    cityNames = ['WH']
    csizeList = range(600, 2001, 100)
    fileNoList = range(19, 21, 1)
    try:
        return cityNames,csizeList,fileNoList,MAXRANGE
    except NameError:
        print('Variable is not defined. Please initialize the variable first.')



if __name__ == '__main__':
    cityNames, csizeList, fileNoList, MAXRANGE = _init()
    for cityId, cityName in enumerate(cityNames):

        RegionLeft, RegionBottom, proj, odir_checkins, titleName = get_init_param(cityName)

        for fileNo in fileNoList:
            # slightly move the study area
            RegionLeft, RegionBottom, delta_x, delta_y = update_study_area(RegionLeft,RegionBottom)
            print('FileNo: '+str(fileNo),'cityname: '+str(cityName))
            print('The offsets in the x and y directions are '+str(delta_x) + ' m and ' + str(delta_y)+' m respectively.')

            # read data and filter data
            df = pd.read_csv(odir_checkins, low_memory=False, )
            df = df[df['checkin_num'] > 0]
            # print(len(df[df['checkin_num'] > 0]),' valid records')

            x, y = proj_trans(df['lon'], df['lat'], proj)
            df['x'],df['y'] = x,y

            #aggregate and filter data
            for csize in csizeList:
                gridNum = int( MAXRANGE / csize)
                gridpos = get_grid_inf(gridNum)
                gridinf = pd.DataFrame(index = gridpos[0], columns=['gridx', 'gridy'])
                gridinf['gridx'], gridinf['gridy'] = gridpos[1], gridpos[2]

                df['gridId'] = get_grid_id(csize, gridNum, x, y,RegionLeft,RegionBottom)
                df['poi_num'] = 1
                checkin_by_grid = df['checkin_num'].groupby(df['gridId']).sum()
                poi_by_grid = df['poi_num'].groupby(df['gridId']).sum()

                # sdd_df restores aggregated data
                sdd_df = pd.concat([checkin_by_grid, poi_by_grid, gridinf], axis=1).fillna(0)
                sdd_df.rename(columns={'checkin_num': 'checkin_sum', 'poi_num': 'poi_sum'}, inplace=True)
                if (sdd_df.index[0] == -1):
                    sdd_df.drop(sdd_df.index[0], inplace=True)     # delete record of the grid which id is -1.

                # calculate semi-variances at each lag distance
                lags, dsquare = caculate_variance()

                vari_df = pd.DataFrame(columns=['lags', 'vari', 'num'])
                vari_df['lags'] = lags
                vari_df['vari'] = dsquare
                vari_df['num'] = 1

                dir = '../variance/'+str(cityName)+'/'+str(fileNo)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                filepath = dir + '/'+str(cityName)+'_vari_'+str(csize)+'.pkl'
                with open(filepath,'wb') as file:
                    pkl.dump(vari_df,file)
            print('The ' + str(fileNo) + ' file for '+ str(cityName) + ' is finished!')





