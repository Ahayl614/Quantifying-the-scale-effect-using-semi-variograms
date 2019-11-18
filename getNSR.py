import pandas as pd
import pickle as pkl

import numpy as np
from scipy.stats import skew
from scipy.optimize import curve_fit
from math import exp,sqrt,e
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import os

import getVariance

def fun_gaus(x,a,b,c):
    para = [-pow((item-b),2)/2/pow(c,2) for item in x]
    return [a*exp(item) for item in para]

def model_gaus(x,c0,c1,a):
    return [c0+c1*(1-exp(-pow(item/a,2))) for item in x]

def model_exp(x,c0,c1,a):
    return [c0+c1*(1-exp(-(item/a))) for item in x]

def model_spher(x,c0,c1,a):
    return [c0+c1*(1.5*item/a - 0.5*pow((item/a),3)) for item in x]

def model_poly(x,a,b,c):
    return [ a*pow(item,2)+b*item+c for item in x]

def model_pure(x,c):
    return [c for item in x]

def get_fit_para(modelType,laglist,semilog_list):
    dta_list = semilog_list
    if modelType == 'poly':
        p_fit, pcov = curve_fit(model_poly, laglist, semilog_list)
        a,b,c = p_fit.tolist()
        y = model_poly(laglist , a,b,c)
        c0 = c
        c1 = -b**2/4/a
        r = -b/2/a
        k =  round(c0 / (c0+c1),3)
        R2 = round(r2_score(dta_list, y), 3)
        return c0,c1,k,R2,y,r
    elif modelType == 'gaus':
        p_fit,pcov = curve_fit(model_gaus, laglist, semilog_list)
        c0, c1, a = p_fit.tolist()
        y = model_gaus(laglist, *p_fit)
        k = round(c0 / (c0 + c1), 3)
        R2 = round(r2_score(dta_list,y), 3)  #r2_score(y_true,y_pred)
        return c0,c1,k,R2,y
    elif modelType == 'pure':
        p_fit,pcov = curve_fit(model_pure, laglist, semilog_list)
        c = p_fit.tolist()
        y = model_pure(laglist, *p_fit)
        R2 = round(r2_score(dta_list,y), 3)  #r2_score(y_true,y_pred)
        return c,R2,y

if __name__ == '__main__':

    cityNames, csizeList, fileNoList, MAXRANGE = getVariance._init()
    MAXLAG = int(MAXRANGE/2)

    for cityName in cityNames:

        for fileNo in fileNoList:
            nsr_poly_list = []
            nsr_gaus_list = []
            c0_poly_list = []
            c1_poly_list = []
            r_poly_list = []
            gamma_poly_list = []

            odir = '../variance/' + str(cityName) + '/' + str(fileNo) + '/'
            for csize in csizeList:
                filename = str(cityName)+'_vari_'+str(csize)+'.pkl'
                filepath = os.path.join(odir,filename)
                with open(filepath, 'rb') as file:
                    data = pkl.load(file)
                df = pd.DataFrame(data=data)
                df = df[df['vari'] > 0]
                # df['vari'] = df['vari']/pow(cellsize/1000,4)   #/K^4
                tolerance = csize * 0.4
                bin_width = int(tolerance * 2)

                skewraw_list = []
                skewlog_list = []
                laglist = []
                lenlist = []
                semilog_list = []
                semiraw_list = []
                mean_list = []
                median_list = []
                count = -1

                for lag in range(csize, MAXLAG, bin_width):
                    count += 1
                    dta_each_bin = df[(df['lags'] > lag - tolerance) & (df['lags'] < lag + tolerance)]
                    num_dta = len(dta_each_bin)
                    rawdta = dta_each_bin['vari']
                    logdta = np.log(rawdta)
                    # semiraw = round(np.mean(rawdta), 2) / 2
                    # semiraw_list.append(semiraw)
                    semilog = round(np.mean(logdta), 2) / 2
                    semilog_list.append(semilog)
                    if (num_dta < 30):
                        print(csize, ' There is not statistically significant due to limited amount of data!')
                        break
                    else:
                        laglist.append(lag / 1000)
                        skew_raw = skew(rawdta)
                        skew_log = round(skew(logdta), 2)
                        lenlist.append(num_dta)
                        skewraw_list.append(skew_raw)
                        skewlog_list.append(skew_log)

                dta_list = semilog_list
                # fit curve and calculate NSR
                c0_poly, c1_poly, nsr_poly, R2_poly, y_poly, r_poly = get_fit_para('poly', laglist, dta_list)
                nsr_poly_list.append(nsr_poly)
                c0_poly_list.append(c0_poly)
                c1_poly_list.append(c1_poly)
                r_poly_list.append(r_poly)
                # gamma_1 = round(model_poly([csize/1000],a_poly,b_poly,c_poly)[0],3)
                # gamma_poly_list.append(gamma_1)
                c0, c1, nsr_gaus, R2_gaus, y_gaus = get_fit_para('gaus', laglist, dta_list)
                nsr_gaus_list.append(nsr_gaus)

            filepath_nsr = odir + str(cityName) + '_NSR' + '.pkl'
            nsr_dict = {'cellsize': csizeList, 'NSR_poly': nsr_poly}
            with open(filepath_nsr, 'wb') as file:
                pkl.dump(nsr_dict, file)

            # draw the correlation between the NSR and the cell size
            plt.scatter(csizeList, nsr_poly_list)
            plt.plot(csizeList, nsr_poly_list, label='Poly')
            # plt.plot(csize_list, nsr_gaus_list, label='Gauss', linestyle='--')
            fsize = 18
            plt.xlabel('cellsize (m)', fontsize=fsize)
            plt.ylabel('NSR', fontsize=fsize)
            plt.title(str(cityName),fontsize=fsize)
            plt.legend(fontsize=14)
            plt.show()

