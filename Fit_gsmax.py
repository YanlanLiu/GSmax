#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:03:18 2021

@author: yanlanliu

Requires FLUEXNET2015 dataset and inverted stomatal conductance using Penmen-Monteith equation as inputs
"""


import numpy as np
import pandas as pd
import pickle
from scipy.optimize import curve_fit
from datetime import datetime
import calendar
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)

datapath= '/Volumes/Elements/Data/GSmax/'
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial 
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)]) #b=a matrix of k^i for k in window range
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv) #np.linalg.pinv gives the inverse matrix
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals)) #concatenate joins a sequence of arrays
    return np.convolve( m[::-1], y, mode='valid') #convolve joins the two sequences

def toTimestamp(d_list): # for interpolation
  return np.array([calendar.timegm(d.timetuple()) for d in d_list])

def isoutlier(y,multiplier=2):
    return np.abs(y-np.nanmedian(y))>multiplier*np.nanstd(y)

def calR2(y,yhat):
    mask = ~np.isnan(y+yhat)
    y = y[mask]
    yhat = yhat[mask]
    return 1-np.sum((y-yhat)**2)/np.sum((y-np.mean(y))**2)

def calrmse(y,yhat):
    mask = ~np.isnan(y+yhat)
    y = y[mask]
    yhat = yhat[mask]
    return np.sqrt(np.mean((y-yhat)**2))/np.mean(y)

def line(x, g0, g1, m): # x -> [gpp,vpd]; return gc; Eq. 4 in Li et al., 2019, AFM
    return g0+g1*x[:,0]/(x[:,1]**m)

import cvxpy as cvx
import mosek

def quadratic_reg(t,vpd,gs,vpd_trd): #quadratic_reg(0.95,vpd[filter_all],gs,1)
    vpd_sid = np.argsort(vpd)
    gs_sorted = gs[vpd_sid]
    xx = vpd[vpd_sid]
    trd_id = np.argmin(np.abs(xx-vpd_trd))
    # t = 0.95
    fit = cvx.Variable(len(gs_sorted))
    tau = cvx.Parameter(value=t,nonneg=True)
    mu = cvx.Parameter(value=1e4 * min(t, 1-t),nonneg=True)
    f1 = cvx.sum(0.5 * cvx.abs(gs_sorted - fit) + (tau - 0.5) * (gs_sorted - fit))
    f2 = mu * cvx.norm(cvx.diff(fit, 2), 2)
    objective = cvx.Minimize(f1 + f2)
    problem = cvx.Problem(objective)
    problem.solve(solver='MOSEK')
    plt.plot(fit.value)
    return fit.value[trd_id],xx,fit.value

SiteInfo = pd.read_csv(datapath+'FLUXNET2015_site_listing.csv')

PARA = np.zeros([len(SiteInfo),3])+np.nan #['G0','G1','m']
R2 = np.zeros([len(SiteInfo),4])+np.nan
R2_all = np.zeros([len(SiteInfo),2])+np.nan
GSREF = np.zeros([len(SiteInfo),])+np.nan
GSmax = np.zeros([len(SiteInfo),3])+np.nan
Nsample = np.zeros([len(SiteInfo),])+np.nan
G0 = np.zeros([len(SiteInfo),4])+np.nan 
MLAI = np.zeros([len(SiteInfo),])+np.nan
RMSE = np.zeros([len(SiteInfo),4])+np.nan
RMSE00 = np.zeros([len(SiteInfo),4])+np.nan
VPD_gsmax = np.zeros([len(SiteInfo),2])+np.nan

mybounds = ([0., 0., 0], [.5, .05, 1.5])

g0_list = [.1,.25,.4]
g1_list = [.01,.025,.04]
m_list = [.4,.7,1.0]
x0_list  =[[g0_list[i],g1_list[j],m_list[k]] for i in range(3) for j in range(3) for k in range(3)]

        

site_sm = 0
gstrd = 0.50
lai_trd = 6 # 4, 6, 8
for sid in range(0,len(SiteInfo)):
    sitename = SiteInfo['SITE_ID'][sid]
    print(sid,sitename)
    df_LAI = pd.read_csv(datapath+'Fluxnet_LAI/LAI_'+sitename+'.csv')
    if 'Lai' not in list(df_LAI): continue
    tt_lai = np.array([datetime.strptime(tmp,'%Y_%m_%d') for tmp in np.array(df_LAI['system:index'])])
    df = pd.read_csv(datapath+'For_Alex_gc_calculation/'+sitename+'.csv')
    tt_fluxnet = [datetime(df['YEAR'].iloc[i],df['MONTH'].iloc[i],df['DAY'].iloc[i],df['HOUR'].iloc[i],df['MIN'].iloc[i]) for i in range(len(df))]

    # identify bad quality LAI (qc>1), and rescale
   
    Lai_QC = df_LAI['FparLai_QC'].fillna(999)
    qc0 = Lai_QC.apply(lambda x:int(format(int(x),'b').zfill(7)[0],2)) # 0: Good quality (main algorithm with or without saturation)
    qc2 = Lai_QC.apply(lambda x:int(format(int(x),'b').zfill(7)[2],2)) # 0: Detectors apparently fine for up to 50% of channels 1, 2
    qc3 = Lai_QC.apply(lambda x:int(format(int(x),'b').zfill(7)[3:5],2)) # 0: Significant clouds NOT present
    qc = qc0+qc2+qc3

    df_LAI['Lai'][qc!=0] = np.nan
    LAI = np.array(df_LAI['Lai'].interpolate(method='linear'))/10
    
    # Smooth and then interpolate LAI to the time points consistent with Fluxnet obserations
    LAI_itp = np.interp(toTimestamp(tt_fluxnet),toTimestamp(tt_lai),savitzky_golay(LAI,30,1))
    
    # growing season filter. True means within growing season. 
    filter_sns = (LAI_itp>np.quantile(LAI_itp,gstrd))
    LAI_itp[LAI_itp>lai_trd] = lai_trd


    # read meteorological conditions to create filter_met
    nobs_day = 48 if np.abs(df['MIN'][1]-df['MIN'][0])==30 else 24 
    vpd = np.array(df['VPD_F'])/10 #units = kPa
    temp = np.array(df['TA_F']+273.15) # K
    rh = np.array(df['RH']) # K
    gc = np.array(df['gc'])*1e5/8.3145/temp #units = m/s  -> mol/m2/s
    gc_std = np.array(df['gc_std'])*1e5/8.3145/temp; gc_std[gc_std<0] = np.nan
    gpp = np.array(df['GPP_NT_VUT_REF'])#units = umol/m2/s
    netrad = np.array(df['NETRAD']) if 'NETRAD' in list(df) else np.array(df['SW_IN_F']) # W/m2
    netrad = netrad*1.0; netrad[netrad<0] = np.nan
    rain = np.array(df['P_F']) #units = mm
    wind = np.array(df['WS_F']) #units = m/s
    HH = np.array(df['H_F_MDS']) #sensible heat flux
    Hrs = np.array(df['HOUR'])
    LE_F_MDS_QC = np.array(df['LE_F_MDS_QC'])
    nyr = int(len(df)/nobs_day/365)
    yearly_maxrad = np.max(np.nanmean(np.reshape(netrad[:nyr*nobs_day*365],[-1,(nobs_day*365)]),axis=1),axis=0)

    # Meteorological filter: True or 1 -> bad data to be removed
    filter_rain = np.zeros(rain.shape) # no rain -> 0; rainy or shortly after rain -> 1
    for i in np.where(rain>1)[0]: filter_rain[i-4:i+nobs_day*2] = 1
    filter_met = (gc<1e-3)+(gpp<0)+(vpd<.6)+(netrad<.5*yearly_maxrad)+(filter_rain==1)+(wind<1)+(HH<0)+(Hrs<10)+(Hrs>15)+(temp<273.15)
    
    # Outlier filter: True or 1 -> outliers to be removed
    gc[filter_met] = np.nan; gc[isoutlier(gc)] = np.nan; gpp[filter_met] = np.nan; wind[filter_met] = np.nan # to identify outlier using times with valid met data
    filter_outlier = isoutlier(gc)+isoutlier(gpp)+isoutlier(wind)+np.isnan(gc)+np.isnan(gpp)+np.isnan(vpd)+(gc_std>np.nanmean(gc)*3)  
    
    # Combine both filters: True or 1 -> good data to keep
    filter_all = (~filter_met)*(~filter_outlier)*(filter_sns)

    # Fit canopy conductance model
    Nsample[sid,] = int(sum(filter_all))
    if Nsample[sid,]>100: #only calculate performance when there's more than 50 hrs of valid data
    
        x = np.column_stack([gpp,vpd])[filter_all,:]
        y = gc[filter_all]
        
        r2_list = np.zeros([len(x0_list)]); popt_list = np.zeros([len(x0_list),3])
        for ii,x0 in enumerate(x0_list):
            popt_list[ii,:], pcov = curve_fit(line, x, y, p0=np.array(x0),bounds=mybounds)
            r2_list[ii] = calR2(y,line(x,*popt_list[ii,:]))
        R2[sid,:] = np.zeros([1,4])+max(r2_list)
        yhat = line(x,*popt_list[r2_list==max(r2_list),:][0])
        PARA[sid,:] = popt_list[r2_list.argmax(),:]
        b0 = np.zeros([sum(filter_all),])+PARA[sid,0]
        G0[sid,:] = np.zeros([1,4])+PARA[sid,0]
        R2_all[sid,:] = np.zeros([1,2])+max(r2_list)
        
    # Fit canopy conductance model with soil moisture bins    
        if 'SWC_F_MDS_1' in list(df):
            sm = df['SWC_F_MDS_1'][filter_all]; sm[sm<0] = np.nan
            if sum(~np.isnan(sm))>Nsample[sid,]/4 and sum(~np.isnan(sm))>100:
                site_sm = site_sm+1
                trdlist = [0]+[np.nanquantile(sm,qt) for qt in [.25,.50,.75]]+[100]
                print(trdlist)
                
                sy = np.array([])
                syhat = np.array([])
                for subid in range(4):
                    subfilter = (sm>=trdlist[subid]) & (sm<trdlist[subid+1]) 
                    if sum(subfilter)<5: continue
                    x_sub = x[subfilter,:]; y_sub = y[subfilter]
                    
                    
                    r2_list = np.zeros([len(x0_list)]); popt_list = np.zeros([len(x0_list),3])
                    for ii,x0 in enumerate(x0_list):
                        popt_list[ii,:], pcov = curve_fit(line, x_sub, y_sub, p0=np.array(x0),bounds=mybounds)
                        r2_list[ii] = calR2(y_sub,line(x_sub,*popt_list[ii,:]))
                    R2[sid,subid] = max(r2_list)
                    b0[subfilter] = popt_list[r2_list.argmax(),0]
                    G0[sid,subid] = popt_list[r2_list.argmax(),0]
                    yhat_sub = line(x_sub,*popt_list[r2_list==max(r2_list),:][0])
                    RMSE[sid,subid] = calrmse(y_sub,yhat_sub)
                    RMSE00[sid,subid] = calrmse(y_sub,yhat[subfilter])
                    syhat = np.concatenate([syhat,yhat_sub])
                    sy = np.concatenate([sy,y_sub])
                R2_all[sid,1] = calR2(sy,syhat)
        gs = (gc[filter_all]-b0)/LAI_itp[filter_all]
        GSmax[sid,:] = np.array([np.quantile(gs,qt) for qt in [0.85,0.9,0.95]])
        MLAI[sid,] = np.nanmean(LAI_itp[filter_all])

        GSREF[sid],tmp,tmp = quadratic_reg(0.95,vpd[filter_all],gs,1)
        print(GSmax[sid,:],GSREF[sid])


paranames = [r'$g_0$',r'$g_1$',r'$m$']
plt.figure(figsize=(12,4))
for i in range(3):
    plt.subplot('13'+str(i+1))
    sns.distplot(PARA[:,i], kde=False,bins=20)
    plt.xlabel(paranames[i])
    if i==0:plt.ylabel('pdf')
    plt.xlim(mybounds[0][i],mybounds[1][i])
    
#%% Combine with other dataset

BADM = pd.read_csv(datapath+'BADM_all_sites.csv')
CanopyHeight = np.zeros([len(SiteInfo),])+np.nan
for sid in range(len(SiteInfo)): # range(len(10)):
    site_badm = BADM[(BADM['SITE_ID']==SiteInfo['SITE_ID'][sid]) & (BADM['VARIABLE']=='HEIGHTC')] # to findout the rows for each site specifying canopy height
    datavalue = [np.float(itm) for itm in site_badm['DATAVALUE']] # to force the data format to be float
    if len(site_badm)>0: # only calculate the mean when the data is available, use nan otherwise
        CanopyHeight[sid] = np.nanmean(datavalue)
        #print(SiteInfo['SITE_ID'][sid],CanopyHeight[sid])

DRYIND = pd.read_csv(datapath+'Output/DrynessIndex.csv')

df = pd.concat([SiteInfo.drop(columns=['noGs_value=1','MAT','MAP','CANOPY_HEIGHT']),DRYIND.drop(columns=['SITE_ID'])],axis=1) 

# Add gsmax, r2 in gs fitting, and canopy height into the dataframe
df['GSfit_R2'] = R2[:,1]; df['CanopyHeight'] = CanopyHeight;
df['MLAI'] = MLAI
df['Nsample'] = Nsample
df['GSfit_R2_nosm'] = R2[:,0]
df['g0'] = PARA[:,0];df['g1'] = PARA[:,1]; df['m'] = PARA[:,2]
for j in range(3):
    df['GSmax'+str(j)] = GSmax[:,j]; 
for j in range(4):
    df['g0'+str(j)] = G0[:,j]
df['GSREF'] = GSREF

#%%
# Drop sites that do not have data of canopy height or gsmax
df = df.loc[(~df['CanopyHeight'].isna()) & (~df['GSmax0'].isna())].reset_index()

# Group SAV and WSA to SAV, group OSH and CSH to OSH
df['IGBP'][df['IGBP']=='WSA'] = 'SAV'
df['IGBP'][df['IGBP']=='CSH'] = 'OSH'
df['IGBP'][df['IGBP']=='OSH'] = 'SHB'

# find out major land cover types with more than 10 sites
IGBPlist = np.unique(df['IGBP'])
IGBPmajor = IGBPlist[[sum(df['IGBP']==pft)>9 for pft in IGBPlist]]
print('Major land cover types: ', IGBPmajor)
df = df[(df['IGBP']!='WET') & (df['IGBP']!='MF')].reset_index()
print('Total number of valid sites: '+str(len(df)))

df.to_csv('SiteInfo_0624.csv',index=False)

