# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 14:15:39 2017

@author: Ashiv Dhondea
"""

import AstroFunctions as AstFn
import TimeHandlingFunctions as THF

import math
import numpy as np

# Importing what's needed for nice plots.
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params)

from mpl_toolkits.basemap import Basemap

# Libraries needed for time keeping and formatting
import datetime as dt
import pytz
import aniso8601

import pandas as pd # for loading MeerKAT dishes' latlon
# --------------------------------------------------------------------------- #
print 'Loading MeerKAT positions'
dframe = pd.read_excel("MeerKAT64v36.wgs84.64x4_edited.xlsx",sheetname="Sheet1")
dframe = dframe.reset_index()
meerkat_id = dframe['ID'][0:64]
meerkat_lat = dframe['Lat'][0:64]
meerkat_lon = dframe['Lon'][0:64]
# --------------------------------------------------------------------------- #
print 'Sorting out lat lon'
# Location of MeerKAT
lat_meerkat_00 = float(meerkat_lat[0]);
lon_meerkat_00 =  float(meerkat_lon[0]);
altitude_meerkat = 1.038; # [km]

lat_meerkat_01 = float(meerkat_lat[1]);
lon_meerkat_01 = float(meerkat_lon[1]);

lat_meerkat_02 = float(meerkat_lat[2]);
lon_meerkat_02 = float(meerkat_lon[2]);

# Location of Denel Bredasdorp
lat_denel = -34.6; # [deg]
lon_denel = 20.316666666666666; # [deg]
altitude_denel = 0.018;#[km]
# --------------------------------------------------------------------------- #
with open('main_057_iss_00_visibility.txt') as fp:
    for line in fp:
        if 'visibility interval in Tx' in line:
            good_index = line.index('=')
            visibility_interval = line[good_index+1:-1];
            good_index = visibility_interval.index('/');
            start_timestring=visibility_interval[:good_index];
            end_timestring = visibility_interval[good_index+1:];
            
fp.close();
# --------------------------------------------------------------------------- #
print 'Loading data'
timevec = np.load('main_057_iss_02_timevec.npy'); # timevector

lat_sgp4 = np.load('main_057_iss_02_lat_sgp4.npy');
lon_sgp4 = np.load('main_057_iss_02_lon_sgp4.npy');

# discretization step length/PRF
delta_t = timevec[2]-timevec[1];
# time stamps
experiment_timestamps = [None]*len(timevec)
index=0;
with open('main_057_iss_02_experiment_timestamps.txt') as fp:
    for line in fp:
        modified_timestring = line[:-1];
        experiment_timestamps[index] = aniso8601.parse_datetime(modified_timestring);
        index+=1;
fp.close();
experiment_timestamps[0] = experiment_timestamps[0].replace(tzinfo=None)
experiment_timestamps[-1] = experiment_timestamps[-1].replace(tzinfo=None)
title_string = str(experiment_timestamps[0].isoformat())+'/'+str(experiment_timestamps[-1].isoformat());

norad_id = '25544'
# --------------------------------------------------------------------------- #
time_index = np.load('main_057_iss_02_time_index.npy');
tx_el_min_index = time_index[0];
tx_el_max_index = time_index[1];

print 'Tx FoV limits %d and %d' %(tx_el_min_index,tx_el_max_index)
# --------------------------------------------------------------------------- #
tx_beam_indices_best = np.load('main_057_iss_04_tx_beam_indices_best.npy');

tx_beam_index_down = tx_beam_indices_best[0];
tx_bw_time_max = tx_beam_indices_best[1];
tx_beam_index_up = tx_beam_indices_best[2];

tx_beam_index_down_lat = math.degrees(lat_sgp4[tx_beam_index_down]);
tx_beam_index_down_lon = math.degrees(lon_sgp4[tx_beam_index_down]);
tx_beam_index_up_lat = math.degrees(lat_sgp4[tx_beam_index_up]);    
tx_beam_index_up_lon = math.degrees(lon_sgp4[tx_beam_index_up]);  

tx_el_min_index_lat = math.degrees(lat_sgp4[tx_el_min_index]);
tx_el_min_index_lon = math.degrees(lon_sgp4[tx_el_min_index]);

tx_el_max_index_lat = math.degrees(lat_sgp4[tx_el_max_index]);
tx_el_max_index_lon = math.degrees(lon_sgp4[tx_el_max_index]);
# --------------------------------------------------------------------------- #
print 'Finding the relevant epochs'
start_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec,0,experiment_timestamps[0]);
end_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec,len(timevec)-1,experiment_timestamps[0]);

tx_el_min_index_test = THF.fnCalculate_DatetimeEpoch(timevec,tx_el_min_index,experiment_timestamps[0]);
tx_el_max_index_test = THF.fnCalculate_DatetimeEpoch(timevec,tx_el_max_index,experiment_timestamps[0]);

end_epoch_test = end_epoch_test.replace(tzinfo=None);
start_epoch_test = start_epoch_test.replace(tzinfo=None);

tx_el_min_index_test  = tx_el_min_index_test.replace(tzinfo=None);
tx_el_max_index_test = tx_el_max_index_test.replace(tzinfo=None);

tx_beam_index_down_epoch = THF.fnCalculate_DatetimeEpoch(timevec,tx_beam_index_down,experiment_timestamps[0]);
tx_beam_index_up_epoch = THF.fnCalculate_DatetimeEpoch(timevec,tx_beam_index_up,experiment_timestamps[0]);
tx_bw_time_max_epoch = THF.fnCalculate_DatetimeEpoch(timevec,tx_bw_time_max,experiment_timestamps[0]);

tx_beam_index_down_epoch = tx_beam_index_down_epoch.replace(tzinfo=None);
tx_beam_index_up_epoch = tx_beam_index_up_epoch.replace(tzinfo=None);
tx_bw_time_max_epoch  = tx_bw_time_max_epoch .replace(tzinfo=None);

title_string = str(start_epoch_test.isoformat())+'Z/'+str(end_epoch_test .isoformat())+'Z';
# --------------------------------------------------------------------------- #
print 'plotting results'

fig = plt.figure(1);ax = fig.gca();
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
plt.rc('font',family='helvetica');
params = {'legend.fontsize': 8,
    'legend.handlelength': 2}
plt.rcParams.update(params)
map = Basemap(llcrnrlon=10.0,llcrnrlat=-52.0,urcrnrlon=37.,urcrnrlat=-11.,resolution='i', projection='cass', lat_0 = 0.0, lon_0 = 0.0)
map.drawcoastlines()
map.drawcountries()

lon =np.rad2deg(lon_sgp4);
lat = np.rad2deg(lat_sgp4);

x,y = map(lon, lat)
map.plot(x, y, color="blue", latlon=False,linewidth=1.3)

map.drawgreatcircle(tx_el_min_index_lon,tx_el_min_index_lat,lon_denel,lat_denel,linewidth=1,color='crimson')
map.drawgreatcircle(tx_el_max_index_lon,tx_el_max_index_lat,lon_denel,lat_denel,linewidth=1,color='crimson')

map.drawgreatcircle(tx_beam_index_down_lon,tx_beam_index_down_lat,lon_denel,lat_denel,linewidth=1,color='green')
map.drawgreatcircle(tx_beam_index_up_lon,tx_beam_index_up_lat,lon_denel,lat_denel,linewidth=1,color='green')

x,y = map(lon_denel,lat_denel)
map.plot(x,y,marker='o',color='green'); # Denel Bredasdorp lat lon
x2,y2 = map(19.1,-36.2)
plt.annotate(r"\textbf{Tx}", xy=(x2, y2),color='green')
x,y = map(lon_meerkat_00,lat_meerkat_00)
map.plot(x,y,marker='o',color='blue'); # rx lat lon
x2,y2 = map(22,-30)
plt.annotate(r"\textbf{Rx}", xy=(x2, y2),color='blue')
parallels = np.arange(-81.,0.,5.)
# labels = [left,right,top,bottom]
map.drawparallels(parallels,labels=[False,True,False,False],labelstyle='+/-',linewidth=0.2)
meridians = np.arange(10.,351.,10.)
map.drawmeridians(meridians,labels=[True,False,False,True],labelstyle='+/-',linewidth=0.2)
plt.title(r'\textbf{Object %s trajectory during the interval %s}' %(norad_id,title_string), fontsize=12)

#plt.legend(loc='upper right',title=r"Transit through Tx FoV");
#ax.get_legend().get_title().set_fontsize('10')
fig.savefig('main_057_iss_14_map.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10) 
# --------------------------------------------------------------------------- #
fig = plt.figure(2);ax = fig.gca();
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
plt.rc('font',family='helvetica');
params = {'legend.fontsize': 8,
    'legend.handlelength': 2}
plt.rcParams.update(params)
map = Basemap(llcrnrlon=10.0,llcrnrlat=-52.0,urcrnrlon=37.,urcrnrlat=-11.,resolution='i', projection='cass', lat_0 = 0.0, lon_0 = 0.0)
map.drawcoastlines()
map.drawcountries()

lon =np.rad2deg(lon_sgp4);
lat = np.rad2deg(lat_sgp4);

x,y = map(lon, lat)
map.plot(x, y, color="blue", latlon=False,linewidth=1.3)

map.drawgreatcircle(tx_el_min_index_lon,tx_el_min_index_lat,lon_denel,lat_denel,linewidth=1,color='crimson')
map.drawgreatcircle(tx_el_max_index_lon,tx_el_max_index_lat,lon_denel,lat_denel,linewidth=1,color='crimson')

x,y = map(tx_el_min_index_lon,tx_el_min_index_lat)
map.scatter(x,y,marker='*',color='orangered',label=r"%s" %str(tx_el_min_index_test.isoformat()+'Z'));

x,y = map(tx_el_max_index_lon,tx_el_max_index_lat)
map.scatter(x,y,marker='>',color='purple',label=r"%s" %str(tx_el_max_index_test.isoformat()+'Z'));


#map.drawgreatcircle(tx_beam_index_down_lon,tx_beam_index_down_lat,lon_denel,lat_denel,linewidth=1,color='green')
#map.drawgreatcircle(tx_beam_index_up_lon,tx_beam_index_up_lat,lon_denel,lat_denel,linewidth=1,color='green')

x,y = map(lon_denel,lat_denel)
map.scatter(x,y,marker='o',color='green'); # Denel Bredasdorp lat lon
x2,y2 = map(19.1,-36.2)
plt.annotate(r"\textbf{Tx}", xy=(x2, y2),color='green')
x,y = map(lon_meerkat_00,lat_meerkat_00)
map.scatter(x,y,marker='o',color='blue'); # rx lat lon
x2,y2 = map(22,-30)
plt.annotate(r"\textbf{Rx}", xy=(x2, y2),color='blue')
parallels = np.arange(-81.,0.,5.)
# labels = [left,right,top,bottom]
map.drawparallels(parallels,labels=[False,True,False,False],labelstyle='+/-',linewidth=0.2)
meridians = np.arange(10.,351.,10.)
map.drawmeridians(meridians,labels=[True,False,False,True],labelstyle='+/-',linewidth=0.2)
plt.title(r'\textbf{Object %s trajectory during the interval %s}' %(norad_id,title_string), fontsize=12)

plt.legend(loc='upper right',title=r"\textbf{Transit through Tx FoR}");
ax.get_legend().get_title().set_fontsize('11')
fig.savefig('main_057_iss_14_map2.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10)