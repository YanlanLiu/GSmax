#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:48:54 2021

@author: yanlanliu

Used to generate Fig 3 and Fig 4
"""

import numpy as np
import pandas as pd
import pickle
from scipy import optimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
import warnings; warnings.simplefilter("ignore")

df = pd.read_csv('SiteInfo_0624.csv')


df['IGBP'][df['IGBP']=='OSH'] = 'SHB'

IGBPmajor = ['ENF','DBF','EBF','SHB','SAV','GRA','CRO']
markers = ['^','o','v','d','p','X','P']
cc = ['b','orange','g','r','purple','tan','pink']

ModelTerms = [['1/CH',''], # canopy height
              ['PET/P','PET_P','PET/ET','PET_ET'],#,'ET/P','ET_P',''], # dryness index, ,
              ['MAT','MAP','']] # mean climate

# Data corresponding to the model terms
varmap = 'MAP_tower'
varpet = 'PET_PT'

Data = [[1/df['CanopyHeight'].values],
        [df[varpet].values/df[varmap].values,df[varpet].values-df[varmap].values,
         df[varpet].values/df['MET'].values,df[varpet].values-df['MET'].values,df['MET'].values/df[varmap].values,df['MET'].values/df[varmap].values],
        [df['MAT_tower'].values,df[varmap].values]]

GSmaxname = 'GSmax1'
ModelList = [[vi,vj,vk] for vi in ModelTerms[0] for vj in ModelTerms[1] for vk in ModelTerms[2]]



#%%

def cal_aic(y,yhat,N,k): 
    # assuming Gaussian noise, one unkown variance per model, beta_MLE=beta_OLS
    sse = np.sum((y-yhat)**2)
    aic = N*(np.log(2*np.pi)+1)+N*np.log(sse/N)+2*k
    return aic

def fit_pft_const(): # benchmark model, one average value per PFT
    Y = np.array(df[GSmaxname])
    beta = np.array([np.mean(Y[df['IGBP']==itm]) for itm in IGBPmajor])
    yhat = np.array([beta[IGBPmajor.index(itm)] for itm in df['IGBP']])
    N = len(Y); k = len(IGBPmajor)*2
    aic = cal_aic(Y,yhat,N,k)
    r2 = np.corrcoef(Y,yhat)[0,1]**2
    return yhat,aic,r2,beta

# yhat,aic,r2,beta = fit_pft_const()
def isoutlier(x,factor=2):
    return np.abs(x-np.nanmedian(x))>factor*np.nanstd(x)

def X_by_model(model,samplefrac,remove_outlier=False,normalize=True): # Generate X matrix for any given model
    if samplefrac<1:
        df0 = df.sample(frac=samplefrac).drop(columns=['level_0','index']).reset_index()
    else:
        df0 = df.copy()
    X = np.ones([len(df0),1])
    Y = np.array(df0[GSmaxname])
    Data = [[1/df0['CanopyHeight'].values],
            [df0[varpet].values/df0[varmap].values,df0[varpet].values-df0[varmap].values,
             df0[varpet].values/df0['MET'].values,df0[varpet].values-df0['MET'].values,df0['MET'].values/df0[varmap].values,df0['MET'].values/df0[varmap].values],
            [df0['MAT_tower'].values,df0[varmap].values]]
    
    IGBP = np.array(df0['IGBP'])
    for subid,sublist in enumerate(ModelTerms):
        if model[subid] != '':
            idx = sublist.index(model[subid])
            X = np.column_stack([X,Data[subid][idx]])
    
    if remove_outlier and X.shape[1]>1:
        outlier = np.any(np.apply_along_axis(isoutlier,0,X[:,1:]),axis=1)
        X = X[~outlier,:]
        Y = Y[~outlier,]
        IGBP = IGBP[~outlier,]
        
    if normalize:
        for i in range(1,X.shape[1]):
            X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])
            
    return X,Y,IGBP

def fit_pft_lm(model,samplefrac=1): # one linear regression per PFT, varying coefficients
    X,Y,IGBP = X_by_model(model,samplefrac,normalize=False,remove_outlier=False)
    yhat = np.zeros(Y.shape)
    beta = []
    for pft in IGBPmajor:
        subset = np.array(IGBP==pft)
        X_sub = X[subset]
        Y_sub = Y[subset]
        mod = sm.OLS(Y_sub,X_sub)
        res = mod.fit()
        yhat[subset] = res.predict()
        beta.append(res.params)
    
    yhat[yhat<0] = 0
    N = X.shape[0]; k = (X.shape[1]+1)*len(IGBPmajor)
    aic = cal_aic(Y,yhat,N,k)
    r2 = np.corrcoef(Y,yhat)[0,1]**2
    return yhat,aic,r2,beta

def rmse_clm(beta,X,Y,IGBP): # rmse of combined linear model
    yhat = np.dot(X,beta[-X.shape[1]:]) #np.dot(X,beta[-4:])
    scalar = [beta[IGBPmajor.index(IGBP[i])] for i in range(len(Y))]
    yhat = yhat*scalar
    rmse = np.mean(np.sqrt((Y-yhat)**2))
    return rmse

def fit_clm(model,samplefrac=1): # fit combined linear fmodel, fixed coefficients and PFT-based scalars
    X,Y,IGBP = X_by_model(model,samplefrac, remove_outlier=False,normalize=True)
    N = X.shape[0]; k = len(IGBPmajor)+X.shape[1]+1
    
    beta0 = np.zeros([len(IGBPmajor)+X.shape[1],])+1e-1
    
    bounds = [(-np.inf,np.inf) for i in range(len(beta0))]
        
    res = optimize.minimize(rmse_clm,beta0,args=(X,Y,IGBP), bounds=bounds)

    beta = res.x
    yhat = np.dot(X,beta[-X.shape[1]:])
    yhat[yhat<0] = 0
    scalar = [beta[IGBPmajor.index(IGBP[i])] for i in range(len(Y))]
    yhat = yhat*scalar
    aic = cal_aic(Y,yhat,N,k)
    r2 = np.corrcoef(Y,yhat)[0,1]**2
    return yhat,aic,r2,beta




def plot_by_pft(Y,yhat,IGBP,R2,AIC,xlabeltag='',titlelegend=False):
    plt.figure(figsize=(6,6))
    for i,lc in enumerate(IGBPmajor):
        plt.scatter(yhat[IGBP==lc],Y[IGBP==lc],label=lc,color=cc[i],marker=markers[i])
    # plt.plot([min(Y),max(Y)],[min(Y),max(Y)],'--k')
    plt.plot([-0.02,0.54],[-0.02,0.54],'--k')
    # plt.plot([0,0.45],[0,0.45],'--k')
    # sns.scatterplot(yhat,Y,hue=df['IGBP'])
    
    plt.xlabel(xlabeltag)
    plt.ylabel(r'$g_{s,u}$ (mol/m$^2$/s)')
    if titlelegend:
        plt.title(f'R$^2$ = {R2:.2f}, AIC = {AIC:.2f}')
        plt.legend(bbox_to_anchor=(1.05,1.05))
    print(R2,AIC)

    plt.xticks(np.arange(0,0.6,0.1))
    plt.yticks(np.arange(0,0.6,0.1))
    
    
    
def print_summary(idx,r2,aic,model,beta):
    summary = f'Model{idx: 1d}: R2={r2:.2f}, AIC={aic:.2f}, {beta[0]:.2f}'
    ibeta = 1
    for itm in model:
        if itm !='':
            summary = summary+" + "+f"{beta[ibeta]:.3f} "+itm
            ibeta = ibeta+1
    print(summary)
    return 0
#%% fit the PFT-average model and combined linear model

yhat,AIC0,R20,beta = fit_pft_const()
Y = np.array(df[GSmaxname]); 
IGBP = np.array(df['IGBP'])
plot_by_pft(Y,yhat,IGBP,R20,AIC0,xlabeltag=r'PFT-averaged $g_{s,u}$ (mol/m$^2$/s)')
plt.xlim([-0.02,0.45]);plt.ylim([-0.02,0.45])


#%%
wCV = np.zeros([len(IGBPmajor),])
ymean = np.zeros([len(IGBPmajor),])
for i,lc in enumerate(IGBPmajor):
    ysub = df[GSmaxname][df['IGBP']==lc]
    wCV[i] = np.std(ysub)/np.mean(ysub)
    ymean[i] = np.mean(ysub)

cCV = np.std(ymean)/np.mean(ymean)

plt.figure(figsize=(4,6))
c = 'b';cm = 'k'
plt.boxplot(wCV,notch=False, patch_artist=True,boxprops=dict(facecolor=c, color=c),
            whiskerprops=dict(color=c),capprops=dict(color=c),flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=cm))
st = 0.6;dd = 0.15
plt.plot([st,st+dd],cCV*np.ones([2,1]),color='r')
plt.xlim([st-dd,1+dd*1.5])
plt.xticks([st+dd/2,1],labels=['Across-PFTs','Within-PFTs'],rotation=30)
plt.ylabel('Coefficient of variation')
plt.ylim([0.2,0.65])


#%%
AIC = np.zeros([len(ModelList),])
R2 = np.zeros([len(ModelList),])
BETA = []
for i,model in enumerate(ModelList):
    yhat,AIC[i],R2[i],beta = fit_clm(model)
    BETA.append(beta)
    
    
#%% plot AIC and R2 of models
model_rank = np.argsort(AIC)
top_model_id = model_rank[:10]
plt.figure()
plt.plot(np.sort(AIC))
plt.plot([0,len(ModelList)],[AIC0,AIC0],'--k')
plt.xlabel('Models ranked by AIC')
plt.ylabel('AIC')
plt.figure()
plt.plot(R2[model_rank])
plt.plot([0,len(ModelList)],[R20,R20],'--k')
plt.xlabel('Models ranked by AIC')
plt.ylabel(r'$R^2$')

            
for rank,i in enumerate(model_rank[0:11]):
    print_summary(rank,R2[i],AIC[i],ModelList[i],BETA[i][len(IGBPmajor):])

#%%
yhat,AIC1,R21,beta = fit_clm(ModelList[model_rank[0]])
plot_by_pft(Y,yhat,IGBP,R21,AIC1,xlabeltag=r'$g_{s,u}$ from scaled model (mol/m$^2$/s)')
plt.xlim([-0.02,0.45]);plt.ylim([-0.02,0.45])


