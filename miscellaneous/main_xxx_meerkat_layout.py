# -*- coding: utf-8 -*-
"""
Created on 10 October 2017

@author: Ashiv Dhondea
"""
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params)

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import math

import AstroFunctions as AstFn
# ------------------------------------------------------------------------------------ #
dframe = pd.read_excel("MeerKAT64v36.wgs84.64x4_edited.xlsx",sheetname="Sheet1")
dframe = dframe.reset_index()

meerkat_id = dframe['ID'][0:64]
meerkat_lat = dframe['Lat'][0:64].astype(dtype=np.float64, copy=True)
meerkat_lon = dframe['Lon'][0:64].astype(dtype=np.float64, copy=True)
# -----------------------------------
altitude_meerkat = 1.038; # [km]
meerkat_ecef = np.zeros([64,3],dtype=np.float64);

baselines = np.zeros([64,64],dtype=np.float64);

for i in range(0,np.shape(meerkat_ecef)[0]):
    meerkat_ecef[i,:] = AstFn.fnRadarSite(math.radians(meerkat_lat[i]),math.radians(meerkat_lon[i]),altitude_meerkat);

for i in range(63,0,-1):
    for j in range(0,i,1):
        baselines[i,j] = np.linalg.norm(np.subtract(meerkat_ecef[i,:],meerkat_ecef[j,:]))

#longest_baseline_indices = np.argmax(baselines);
longest_baseline_indices_unravel = np.unravel_index(baselines.argmax(), baselines.shape)
print longest_baseline_indices_unravel
longest_baseline = np.max(baselines)
print longest_baseline
print baselines[longest_baseline_indices_unravel[0],longest_baseline_indices_unravel[1]]

print baselines[60,48]


lim_lon_min = meerkat_lon.min();
lim_lon_max = meerkat_lon.max();

lim_lat_min = meerkat_lat.min();
lim_lat_max = meerkat_lat.max();

fig = plt.figure(1);
plt.rc('text', usetex=True)
plt.rc('font', family='serif');

map = Basemap(llcrnrlon=lim_lon_min-0.005,llcrnrlat=lim_lat_min-0.01,urcrnrlon=lim_lon_max+0.005,urcrnrlat=lim_lat_max+0.01,resolution='f', projection='cass', lat_0 = 0.0, lon_0 = 0.0) # see http://boundingbox.klokantech.com/

#map.drawmapboundary(fill_color='aqua')
#map.fillcontinents(color='coral',lake_color='aqua')

map.drawmapboundary(fill_color='lightblue')
map.fillcontinents(color='beige',lake_color='lightblue')

parallels = np.arange(-81.,0.,0.02)
# labels = [left,right,top,bottom]
map.drawparallels(parallels,labels=[False,True,False,False],labelstyle='+/-',linewidth=0.2)
meridians = np.arange(10.,351.,0.02)
map.drawmeridians(meridians,labels=[True,False,False,True],labelstyle='+/-',linewidth=0.2)


for i in range(64):
    x,y = map(meerkat_lon[i],meerkat_lat[i]);
    map.plot(x,y,marker='o',markersize=3,color='blue');
for i in range(48,64,1):
    x,y = map(meerkat_lon[i],meerkat_lat[i]);
    plt.text(x, y, r"\textbf{%s}" %meerkat_id[i],fontsize=6,color='navy')
#    plt.annotate(r"\textbf{%s}" %meerkat_id[i],xy = (x,y),color='navy')
    
plt.title(r'\textbf{Location of MeerKAT dishes}', fontsize=12);
fig.savefig('main_xxx_meerkat_layout.pdf',bbox_inches='tight',pad_inches=0.08,dpi=10);
