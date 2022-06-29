#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:54:14 2021

@author: yanlanliu

Used to generate Fig 2


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
import warnings; warnings.simplefilter("ignore")

UNIT_0 = 18e-6 # mol H2O/m2/s -> m/s H2O
UNIT_1 = 1.6*UNIT_0 # mol CO2 /m2/s -> m/s, H2O
UNIT_2 = 1e6 # Pa -> MPa
UNIT_3 = 273.15 # Degree C -> K
UNIT_4 = UNIT_0*3600*24 # mol H2O /m2/s -> mm/day,s H2O
ca = 400 # ppm, atmospheric CO2 concentration


class OptimalBioChem:
    def __init__(self):
        self.koptj = 155.76 #  umol/m2/s
        self.Haj = 43.79 # kJ/mol
        self.Hdj = 200; # kJ/mol
        self.Toptj = 32.19+UNIT_3 # K
        self.koptv = 174.33 # umol/m2/s
        self.Hav = 61.21 # kJ/mol
        self.Hdv = 200 # kJ/mol
        self.Toptv = 37.74+UNIT_3 # K
        self.Coa = 210 # mmol/mol
        self.kai1 = 0.9
        self.kai2 = 0.3

class Constants:
    def __init__(self):
        self.R = 8.31*1e-3 # Gas constant, kJ/mol/K
        self.NA = 6.02e23 # Avogadro's constant, /mol
        self.hc = 2e-25 # Planck constant times light speed, J*s times m/s
        self.wavelen = 500e-9 # wavelength of light, m
        self.Ephoton = self.hc/self.wavelen
        self.ca = 400 # Atmospheric Co2 concentration, ppm
        self.Cpmol = 1005*28.97*1e-3 # J/kg/K*kg/mol -> J/mol/K
        self.lambda0 = 2.26*10**6
        self.gammaV = 100*1005/(self.lambda0*0.622) #in kpa, constant in PM equation
        self.a0 = 1.6 # relative diffusivity of h2o to co2 through stomata
        self.U3 = 273.15

OB = OptimalBioChem()
CONST = Constants()
T2ES  = lambda x: 0.6108*np.exp(17.27*(x)/(x+237.3))# saturated water pressure, kPa

#%%
VPD_kPa = 0.6; RNET = 600; T_C = 25
g1 = 4; Vcmax25 = 56
T2ES  = lambda x: 0.6108*np.exp(17.27*(x)/(x+237.3))# saturated water pressure, kPa
# es = T2ES(T_C)
RH = 1-VPD_kPa/T2ES(T_C)

def cal_gs_medlyn(g1,Vcmax25,T_C,RNET,VPD_kPa):
    TEMP = T_C+CONST.U3 # degree C
    PAR = RNET/(CONST.Ephoton*CONST.NA)*1e6
    Kc = 300*np.exp(0.074*(T_C-25)) # umol/mol
    Ko = 300*np.exp(0.015*(T_C-25)) # mmol/mol
    cp = 36.9+1.18*(T_C-25)+0.036*(T_C-25)**2
    Vcmax0 = Vcmax25*np.exp(50*(TEMP-298)/(298*CONST.R*TEMP)) 
    Jmax = Vcmax0*1.97 # np.exp(1)*Vcmax25
    J = (OB.kai2*PAR+Jmax-np.sqrt((OB.kai2*PAR+Jmax)**2-4*OB.kai1*OB.kai2*PAR*Jmax))/2/OB.kai1
    medlyn_term = 1+g1/np.sqrt(VPD_kPa) # double check
    ci = ca*(1-1/medlyn_term)
    Rd = 0.015*Vcmax0
    a1 = np.array(Vcmax0*(ci-cp)/(ci + Kc*(1+209/Ko)))-Rd
    a2 = np.array(J*(ci-cp)/(4*(ci + 2*cp)))-Rd
    A = np.nanmin(np.column_stack([a1,a2]),axis=1)
    gs = A/(ca-ci)/1.6 # mol H20/m2/s
    return gs


def cal_gs_bb(m,Vcmax25,T_C,RNET,VPD_kPa):
    if np.isnan(Vcmax25):
        gs = np.nan
        # print('nan')
    else:
        RH = 1-VPD_kPa/T2ES(T_C)
        TEMP = T_C+CONST.U3 # degree C
        PAR = RNET/(CONST.Ephoton*CONST.NA)*1e6
        Kc = 300*np.exp(0.074*(T_C-25)) # umol/mol
        Ko = 300*np.exp(0.015*(T_C-25)) # mmol/mol
        cp = 36.9+1.18*(T_C-25)+0.036*(T_C-25)**2
        Vcmax0 = Vcmax25*np.exp(50*(TEMP-298)/(298*CONST.R*TEMP)) 
        Jmax = Vcmax0*1.97 # np.exp(1)*Vcmax25
        J = (OB.kai2*PAR+Jmax-np.sqrt((OB.kai2*PAR+Jmax)**2-4*OB.kai1*OB.kai2*PAR*Jmax))/2/OB.kai1
        Rd = 0.015*Vcmax0
        Rd = 0
        ci = np.arange(cp,ca,0.01)
        a1 = np.array(Vcmax0*(ci-cp)/(ci + Kc*(1+209/Ko)))-Rd
        gs = m*a1/ca*RH+0.01
        err = np.abs(a1-gs*(ca-ci))
        a1 = a1[err==np.nanmin(err)]
        
        
        a2 = np.array(J*(ci-cp)/(4*(ci + 2*cp)))-Rd
        gs = m*a2/ca*RH+0.01
        err = np.abs(a2-gs*(ca-ci))
        a2 = a2[err==np.nanmin(err)]
        
        A = np.nanmin(np.column_stack([a1,a2]),axis=1)
        gs = m*A/ca*RH+0.01
    return gs


VCMAX25_L = np.array([[0,0,0,62.5],[41.0, 57.7, 57.7,0], [55.0, 61.5,0,0], 
             [0, 61.7, 54.0, 0],[0,0,0,0],[0,0,0,78.2], [0,0,0,100.7]])

M_L = 9
G1_L = np.array([[0,0,0,2.35],[4.45, 4.45, 4.45, 0],[4.12, 4.12,0,0],
        [0,4.7,4.7,0],[0,0,0,0],[0,0,0,5.25],[0,0,0,5.79]])


symbol = ['p','^','s','o']
# ENF, DBF, EBF, OSH, SAV, GRA, CRO
VCMAX25_N = np.array([50,60,60,40,40,40,80])
M_N = np.array([6,9,9,9,9,9,9])
GSmax_ml = np.zeros(VCMAX25_L.shape)
GSmax_bb = np.zeros(VCMAX25_L.shape)
for i in range(GSmax_ml.shape[0]):
    for j in range(4):
        GSmax_ml[i,j] = cal_gs_medlyn(G1_L[i,j],VCMAX25_L[i,j],25,700,0.6)
        GSmax_bb[i,j] = cal_gs_bb(M_L,VCMAX25_L[i,j],25,700,0.6)

GS_gldas = 1/np.array([125,100,150,235,70,40,40]) * 1e5/8.3145/(273+25) # m/s -> mol/m2/s
GSmax_ml[GSmax_ml==0] = np.nan #['']
GSmax_bb[GSmax_bb<=0.01] = np.nan

#%%
df = pd.read_csv('SiteInfo_0624.csv')

df['IGBP'][df['IGBP']=='OSH'] ='SHB'
IGBPmajor = ['ENF','DBF','EBF','SHB','SAV','GRA','CRO']

wCV = np.zeros([len(IGBPmajor),])
ymean = np.zeros([len(IGBPmajor),3])
for i,lc in enumerate(IGBPmajor):   
    for j in range(3):
        ysub = df['GSmax'+str(j)][df['IGBP']==lc]
        ymean[i,j] = np.nanmedian(ysub)
    

from sklearn.linear_model import LinearRegression

cc = ['b','orange','g','r','purple','tan','pink']
fill_styles = ['bottom','left','top','full']
markers = ['^','o','v','d','p','X','P']
ms=12

def plot_avg_gsmax(GS_gldas,tag='',legend='True',xlim = [0.056,0.21],ylim=[0.056,0.21]):

    
    biomenames = ['Tropical','Temperate','Boreal','All']
    if len(GS_gldas.shape)==1:
        for i,lc in enumerate(IGBPmajor):
            # if lc=='OSH': lc = 'SHB'
            plt.plot(ymean[i,1],GS_gldas[i],markers[i],label=lc,color=cc[i],markersize=ms)
    else:
            
        for i,lc in enumerate(IGBPmajor):
            # if lc=='OSH': lc = 'SHB'
            for bi in range(4):
                if bi==3:
                    # plt.plot(ymean[i,1],GS_gldas[i,bi],markers[i],label=lc,color=cc[bi],markersize=ms)
                    plt.plot(ymean[i,1],GS_gldas[i,bi],markers[i],label=lc,color=cc[i],markersize=ms,fillstyle=fill_styles[bi])
                else:
                    plt.plot(ymean[i,1],GS_gldas[i,bi],markers[i],color=cc[i],markersize=ms,markerfacecoloralt='w',fillstyle=fill_styles[bi])
        for bi in range(4):
            plt.plot(1,0.4,'s',color='k',markerfacecoloralt='w',fillstyle=fill_styles[bi],markersize=ms,label=biomenames[bi])
    
            # plt.plot(ymean[i,1],GS_gldas[i,1],'o',color=cc[i],markersize=ms)
#         plt.plot([ymean[i,0],ymean[i,2]],np.nanmean(GS_gldas[i,:])*np.array([1,1]),'-o',color=cc[i])

    if legend:plt.legend(bbox_to_anchor=(1.15,1.1))
    plt.xlabel(r'Site-averaged $g_{s,u}$ (mol/m$^2$/s)')
    plt.ylabel('LSM $g_{s,u}$ (mol/m$^2$/s)')
    plt.xlim(xlim);plt.ylim(ylim)
    
    
    if len(GS_gldas.shape)==1:

        tmpfilter = ~np.isnan(GS_gldas)
        
        # tmpfilter[-2:]=False
        r = np.corrcoef(ymean[tmpfilter,1],GS_gldas[tmpfilter])[0][1]
        y = GS_gldas[tmpfilter].flatten()
        x = ymean[tmpfilter,1]
    else:

        tmpfilter = ~np.isnan(np.nanmean(GS_gldas,axis=1))
        # tmpfilter[-2:]=False
        # r = np.corrcoef(ymean[tmpfilter,1],np.nanmean(GS_gldas,axis=1)[tmpfilter])[0][1]
        y = GS_gldas[tmpfilter,:].flatten()
        x = np.transpose(np.tile(ymean[tmpfilter,1],[GS_gldas.shape[1],1])).flatten()
        
    nanfilter = np.isnan(y)
    r = np.corrcoef(x[~nanfilter],y[~nanfilter])[0][1]
    reg = LinearRegression().fit(np.reshape(x[~nanfilter],[-1,1]),y[~nanfilter])
    yy = np.array(xlim)*reg.coef_+reg.intercept_
    plt.plot(xlim,yy,'--k')
    print(r)
    # plt.title(f'r = {r:.2f}')
    

plt.figure(figsize=(16,4.5))
plt.subplots_adjust(wspace=0.4)
plt.subplot(131)
plot_avg_gsmax(GS_gldas,tag='GLDAS ',legend=False,ylim=[0.05,1.23])
# plt.ylim()
plt.subplot(132)
plot_avg_gsmax(GSmax_bb,tag='CLM4.5, BB (mol/m$^2$/s)',legend=False,ylim=[0.23,0.73])
# plt.ylim()
plt.subplot(133)
plot_avg_gsmax(GSmax_ml,tag='CLM5, Medlyn (mol/m$^2$/s)' ,ylim=[0.07,0.51])
