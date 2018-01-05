# -*- coding: utf-8 -*-
"""
Created on 01 October 2017

@author: Ashiv Dhondea
"""

import AstroFunctions as AstFn
import AstroConstants as AstCnst
import GeometryFunctions as GF
import RadarSystem as RS
import TimeHandlingFunctions as THF
import math
import numpy as np

import datetime as dt
import pytz
import aniso8601

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params)
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import matplotlib as mpl
import matplotlib.patches as patches
from mpl_toolkits.basemap import Basemap

import pandas as pd # for loading MeerKAT dishes' latlon
# --------------------------------------------------------------------------- #
print 'Loading MeerKAT positions'
dframe = pd.read_excel("MeerKAT64v36.wgs84.64x4_edited.xlsx",sheetname="Sheet1")
dframe = dframe.reset_index()

meerkat_id = dframe['ID'][0:64]
meerkat_lat = dframe['Lat'][0:64]
meerkat_lon = dframe['Lon'][0:64]
# --------------------------------------------------------------------------- #
with open('main_meerkat_radar_parameters_doreen.txt') as fp:
    for line in fp:
		if 'centre_frequency' in line:
			good_index = line.index('=')
			centre_frequency = float(line[good_index+1:-1]); 
		if 'HPBW Rx' in line:
			good_index = line.index('=')
			beamwidth_rx = float(line[good_index+1:-1]);
		if 'HPBW Tx' in line:
			good_index = line.index('=')
			beamwidth_tx = float(line[good_index+1:-1]);
		if 'HPBW Tx' in line:
			good_index = line.index('=')
			beamwidth_tx = float(line[good_index+1:-1]);
		if 'bandwidth' in line:
			good_index = line.index('=')
			bandwidth = float(line[good_index+1:-1]); 
fp.close();

# --------------------------------------------------------------------------- #
speed_light = AstCnst.c*1e3; # [m/s]
wavelength = speed_light/centre_frequency; # [m]
# --------------------------------------------------------------------------- #
print 'Loading data'
timevec = np.load('main_057_iss_05_timevec.npy'); # timevector
x_target = np.load('main_057_iss_05_x_target.npy'); # state vector in SEZ frame
theta_GMST = np.load('main_057_iss_05_theta_GMST.npy'); # GMST angles in rad
y_sph_tx = np.load('main_057_iss_05_y_sph_tx.npy'); # spherical measurement vectors in Tx frame
y_sph_rx = np.load('main_057_iss_05_y_sph_rx.npy'); # spherical measurement vectors in Rx frame
y_sph_rx_meerkat_01 = np.load('main_057_iss_05_y_sph_rx_meerkat_01.npy'); 
y_sph_rx_meerkat_02 = np.load('main_057_iss_05_y_sph_rx_meerkat_02.npy');

# discretization step length/PRF
delta_t = timevec[2]-timevec[1];

# time stamps
experiment_timestamps = [None]*len(timevec)
index=0;
with open('main_057_iss_05_experiment_timestamps.txt') as fp:
    for line in fp:
        modified_timestring = line[:-8];
        experiment_timestamps[index] = aniso8601.parse_datetime(modified_timestring);
        index+=1;
fp.close();
experiment_timestamps[-1] = experiment_timestamps[-1].replace(tzinfo=None)
title_string1 = str(experiment_timestamps[0].isoformat())+'/'+str(experiment_timestamps[-1].isoformat());
norad_id = '25544'
# --------------------------------------------------------------------------- #
# Bistatic Radar characteristics
# beamwidth of transmitter and receiver
beamwidth_rx = math.radians(beamwidth_rx);

# Location of MeerKAT
lat_meerkat_00 = float(meerkat_lat[0]);
lon_meerkat_00 =  float(meerkat_lon[0]);
altitude_meerkat = 1.038; # [km]

lat_meerkat_01 = float(meerkat_lat[1]);
lon_meerkat_01 = float(meerkat_lon[1]);

lat_meerkat_02 = float(meerkat_lat[2]);
lon_meerkat_02 = float(meerkat_lon[2]);

lat_meerkat_03 = float(meerkat_lat[3]);
lon_meerkat_03 = float(meerkat_lon[3]);

# Location of Denel Bredasdorp
lat_denel = -34.6; # [deg]
lon_denel = 20.316666666666666; # [deg]
altitude_denel = 0.018;#[km]
# --------------------------------------------------------------------------- #
lat_sgp4 = np.load('main_057_iss_05_lat_sgp4.npy',); 
lon_sgp4 = np.load('main_057_iss_05_lon_sgp4.npy'); 
tx_beam_indices_best = np.load('main_057_iss_07_tx_beam_indices_best.npy');
# --------------------------------------------------------------------------- #
# sort out a few variables
tx_bw_time_max = tx_beam_indices_best[1];
tx_beam_index_down = tx_beam_indices_best[0];
# --------------------------------------------------------------------------- #
# sort out a few variables
tx_bw_time_max = tx_beam_indices_best[1];
tx_beam_index_down = tx_beam_indices_best[0];
tx_beam_index_up = tx_beam_indices_best[2];
# --------------------------------------------------------------------------- #
# sort out a few variables
tx_beam_circ_index = np.load('main_057_iss_08_tx_beam_circ_index.npy');
earliest_pt = tx_beam_circ_index[0];
tx_bw_time_max = tx_beam_circ_index[1];
latest_pt = tx_beam_circ_index[2];
# --------------------------------------------------------------------------- #
rx0_beam_circ_index = np.load('main_057_iss_09_rx0_beam_circ_index.npy');
earliest_pt_rx = rx0_beam_circ_index[0]
index_for_rx0 = rx0_beam_circ_index[1]
latest_pt_rx = rx0_beam_circ_index[2]

rx1_beam_circ_index = np.load('main_057_iss_09_rx1_beam_circ_index.npy');
earliest_pt_rx1 = rx1_beam_circ_index[0]
index_for_rx1 = rx1_beam_circ_index[1]
latest_pt_rx1 = rx1_beam_circ_index[2]

rx2_beam_circ_index = np.load('main_057_iss_09_rx2_beam_circ_index.npy');
earliest_pt_rx2 = rx2_beam_circ_index[0]
index_for_rx2 = rx2_beam_circ_index[1]
latest_pt_rx2 = rx2_beam_circ_index[2]
# --------------------------------------------------------------------------- #
print 'finding relevant epochs'
# Find the epoch of the relevant data points
plot_lim = 6
plt_start_index = tx_beam_index_down - int(plot_lim/delta_t)
plt_end_index = tx_beam_index_up+1 + int(2/delta_t) 

start_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec,plt_start_index,experiment_timestamps[0]);
end_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec,plt_end_index,experiment_timestamps[0]);
tx_beam_index_down_epoch = THF.fnCalculate_DatetimeEpoch(timevec,tx_beam_index_down,experiment_timestamps[0]);
tx_beam_index_up_epoch= THF.fnCalculate_DatetimeEpoch(timevec,tx_beam_index_up,experiment_timestamps[0]);
tx_bw_time_max_epoch = THF.fnCalculate_DatetimeEpoch(timevec,tx_bw_time_max,experiment_timestamps[0]);
earliest_pt_epoch= THF.fnCalculate_DatetimeEpoch(timevec,earliest_pt,experiment_timestamps[0]);
latest_pt_epoch = THF.fnCalculate_DatetimeEpoch(timevec,latest_pt,experiment_timestamps[0]);

earliest_pt_epoch = earliest_pt_epoch.replace(tzinfo=None)
end_epoch_test = end_epoch_test.replace(tzinfo=None);
start_epoch_test = start_epoch_test.replace(tzinfo=None)
title_string = str(start_epoch_test.isoformat())+'/'+str(end_epoch_test .isoformat());
tx_beam_index_down_epoch = tx_beam_index_down_epoch.replace(tzinfo=None);
tx_beam_index_up_epoch = tx_beam_index_up_epoch.replace(tzinfo=None)
tx_bw_time_max_epoch = tx_bw_time_max_epoch.replace(tzinfo=None)
latest_pt_epoch= latest_pt_epoch.replace(tzinfo=None)
# --------------------------------------------------------------------------- #

fig = plt.figure(1);ax = fig.gca();
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
plt.rc('font',family='helvetica');
params = {'legend.fontsize': 8,
    'legend.handlelength': 2}
plt.rcParams.update(params)
map = Basemap(llcrnrlon=3.0,llcrnrlat=-39.0,urcrnrlon=34.,urcrnrlat=-8.,resolution='i', projection='cass', lat_0 = 0.0, lon_0 = 0.0)
map.drawcoastlines()
map.drawcountries()
map.drawmapboundary(fill_color='lightblue')
map.fillcontinents(color='beige',lake_color='lightblue')
lon =np.rad2deg(lon_sgp4);
lat = np.rad2deg(lat_sgp4);

x,y = map(lon[plt_start_index:earliest_pt+1], lat[plt_start_index:earliest_pt+1])
map.plot(x, y, color="blue", latlon=False,linewidth=1)

x,y = map(lon[earliest_pt:latest_pt+1], lat[earliest_pt:latest_pt+1])
map.plot(x, y, color="crimson", latlon=False,linewidth=2,label=r"%s" %str(earliest_pt_epoch.isoformat())+'Z/'+str(latest_pt_epoch.isoformat())+'Z');

x,y = map(lon[latest_pt+1:plt_end_index+1], lat[latest_pt+1:plt_end_index+1])
map.plot(x, y, color="blue", latlon=False,linewidth=1)

x,y = map(lon_denel,lat_denel)
map.plot(x,y,marker='o',color='green'); # Denel Bredasdorp lat lon
x2,y2 = map(20,-34)
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

plt.legend(loc='upper right',title=r"Dwell-time interval");
ax.get_legend().get_title().set_fontsize('10')
fig.savefig('main_057_iss_13_map.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10) 
# --------------------------------------------------------------------------- #
tx_beam_index_down_lat = math.degrees(lat_sgp4[tx_beam_index_down]);
tx_beam_index_down_lon = math.degrees(lon_sgp4[tx_beam_index_down]);
tx_beam_index_up_lat = math.degrees(lat_sgp4[tx_beam_index_up]);    
tx_beam_index_up_lon = math.degrees(lon_sgp4[tx_beam_index_up]);       
    
#earliest_pt_rx   
rx_beam_index_down_lat = math.degrees(lat_sgp4[earliest_pt_rx]);
rx_beam_index_down_lon = math.degrees(lon_sgp4[earliest_pt_rx]);
rx_beam_index_up_lat = math.degrees(lat_sgp4[latest_pt_rx]);    
rx_beam_index_up_lon = math.degrees(lon_sgp4[latest_pt_rx]); 

rx1_beam_index_down_lat = math.degrees(lat_sgp4[earliest_pt_rx1]);
rx1_beam_index_down_lon = math.degrees(lon_sgp4[earliest_pt_rx1]);
rx1_beam_index_up_lat = math.degrees(lat_sgp4[latest_pt_rx1]);    
rx1_beam_index_up_lon = math.degrees(lon_sgp4[latest_pt_rx1]); 

rx2_beam_index_down_lat = math.degrees(lat_sgp4[earliest_pt_rx2]);
rx2_beam_index_down_lon = math.degrees(lon_sgp4[earliest_pt_rx2]);
rx2_beam_index_up_lat = math.degrees(lat_sgp4[latest_pt_rx2]);    
rx2_beam_index_up_lon = math.degrees(lon_sgp4[latest_pt_rx2]); 
# --------------------------------------------------------------------------- #

fig = plt.figure(2);
ax = fig.gca();
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
plt.rc('font',family='helvetica');
params = {'legend.fontsize': 8,
    'legend.handlelength': 2}
plt.rcParams.update(params)
map = Basemap(llcrnrlon=3.0,llcrnrlat=-38.0,urcrnrlon=34.,urcrnrlat=-16.,resolution='i', projection='cass', lat_0 = 0.0, lon_0 = 0.0)
map.drawcoastlines()
lon =np.rad2deg(lon_sgp4);
lat = np.rad2deg(lat_sgp4);

x,y = map(lon[plt_start_index:earliest_pt+1], lat[plt_start_index:earliest_pt+1])
map.plot(x, y, color="blue", latlon=False,linewidth=1)

x,y = map(lon[tx_beam_index_down:tx_beam_index_up+1], lat[tx_beam_index_down:tx_beam_index_up+1])
map.plot(x, y, color="crimson", latlon=False,linewidth=2,label=r"%s" %str(tx_beam_index_down_epoch.isoformat())+'Z/'+str(tx_beam_index_up_epoch.isoformat())+'Z');

x,y = map(lon[tx_beam_index_up+1:plt_end_index+1], lat[tx_beam_index_up+1:plt_end_index+1])
map.plot(x, y, color="blue", latlon=False,linewidth=1)

x_denel,y_denel = map(lon_denel,lat_denel)
map.plot(x_denel,y_denel,marker='o',color='green'); # Denel Bredasdorp lat lon
x2,y2 = map(20,-34)
plt.annotate(r"\textbf{Tx}", xy=(x2, y2),color='green')

tx_beam_index_down_x,tx_beam_index_down_y = map(tx_beam_index_down_lon,tx_beam_index_down_lat )
tx_beam_index_up_x,tx_beam_index_up_y = map(tx_beam_index_up_lon,tx_beam_index_up_lat )

map.drawgreatcircle(tx_beam_index_down_lon,tx_beam_index_down_lat, lon_denel,lat_denel,linewidth=0.5,color='gray')
map.drawgreatcircle(tx_beam_index_up_lon,tx_beam_index_up_lat, lon_denel,lat_denel,linewidth=0.5,color='gray')

map.drawgreatcircle(rx_beam_index_down_lon,rx_beam_index_down_lat, lon_meerkat_00,lat_meerkat_00,linewidth=0.5,color='mediumblue')
map.drawgreatcircle(rx_beam_index_up_lon,rx_beam_index_up_lat,lon_meerkat_00,lat_meerkat_00,linewidth=0.5,color='mediumblue')

map.drawgreatcircle(rx1_beam_index_down_lon,rx1_beam_index_down_lat, lon_meerkat_01,lat_meerkat_01,linewidth=0.5,color='orangered')
map.drawgreatcircle(rx1_beam_index_up_lon,rx1_beam_index_up_lat,lon_meerkat_01,lat_meerkat_01,linewidth=0.5,color='orangered')

map.drawgreatcircle(rx2_beam_index_down_lon,rx2_beam_index_down_lat, lon_meerkat_02,lat_meerkat_02,linewidth=0.5,color='purple')
map.drawgreatcircle(rx2_beam_index_up_lon,rx2_beam_index_up_lat,lon_meerkat_02,lat_meerkat_02,linewidth=0.5,color='purple')


x,y = map(lon_meerkat_00,lat_meerkat_00)
map.plot(x,y,marker='o',color='blue'); # rx lat lon
x2,y2 = map(22,-31)
plt.annotate(r"\textbf{Rx}", xy=(x2, y2),color='blue');
plt.title(r'\textbf{Object %s trajectory during the interval %s}' %(norad_id,title_string), fontsize=12)

plt.legend(loc='upper right',title=r"Dwell-time interval");
ax.get_legend().get_title().set_fontsize('10')
fig.savefig('main_057_iss_13_map2.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10) 
