# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:13:15 2017

@author: Ashiv Dhondea
"""

import math
import numpy as np

# Importing what's needed for nice plots.
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params)
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

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
print 'Loading data'
timevec = np.load('main_057_iss_05_timevec.npy'); # timevector
delta_t = timevec[2]-timevec[1]
# time stamps
experiment_timestamps = [None]*len(timevec)
index=0;
with open('main_057_iss_05_experiment_timestamps.txt') as fp:
    for line in fp:
        modified_timestring = line[:-8];
        experiment_timestamps[index] = aniso8601.parse_datetime(modified_timestring);
        index+=1;
fp.close();

norad_id = '25544'

with open('main_meerkat_radar_parameters_doreen.txt') as fp:
    for line in fp:
		if 'HPBW Tx' in line:
			good_index = line.index('=')
			beamwidth_tx = float(line[good_index+1:-1]);
fp.close();

# beamwidth of transmitter and receiver
beamwidth_tx = math.radians(beamwidth_tx); 
# --------------------------------------------------------------------------- #
## Create datetime objects for the start and end of the visibility interval
time_index = np.load('main_057_iss_05_time_index.npy');
tx_el_min_index = time_index[0];
tx_el_max_index = time_index[1];

start_vis_epoch = experiment_timestamps[tx_el_min_index]
start_vis_epoch = start_vis_epoch.replace(tzinfo=None);

end_vis_epoch = experiment_timestamps[tx_el_max_index]
end_vis_epoch = end_vis_epoch.replace(tzinfo=None);

title_string = str(start_vis_epoch.isoformat())+'/'+str(end_vis_epoch.isoformat());
# --------------------------------------------------------------------------- #
y_sph_rx = np.load('main_057_iss_05_y_sph_rx.npy'); # spherical measurement vectors in Rx frame
y_sph_tx = np.load('main_057_iss_05_y_sph_tx.npy'); # spherical measurement vectors in Tx frame
y_sph_rx_meerkat_01 = np.load('main_057_iss_05_y_sph_rx_meerkat_01.npy'); 
y_sph_rx_meerkat_02 = np.load('main_057_iss_05_y_sph_rx_meerkat_02.npy'); 
# --------------------------------------------------------------------------- #
## Bistatic Radar characteristics
rx_az_min = 0.; # [deg]
rx_az_max = 360.; # [deg]
rx_el_min = 15.; # [deg]
rx_el_max = 88.; # [deg]
# --------------------------------------------------------------------------- #
# M000
rx_el_min_range = np.where( y_sph_rx[1,tx_el_min_index:tx_el_max_index+1] >= math.radians(rx_el_min))
rx_el_min_index = rx_el_min_range[0][0] + tx_el_min_index;
rx_el_max_range = np.where( y_sph_rx[1,rx_el_min_index:tx_el_max_index+1] >= math.radians(rx_el_min)) # 06/09/17: debugged
rx_el_max_index = rx_el_max_range[0][-1]+rx_el_min_index;
# M001
rx_el_min_range_meerkat_01 = np.where( y_sph_rx_meerkat_01[1,tx_el_min_index:tx_el_max_index+1] >= math.radians(rx_el_min))
rx_el_min_index_meerkat_01 = rx_el_min_range_meerkat_01[0][0] + tx_el_min_index;
rx_el_max_range_meerkat_01 = np.where( y_sph_rx_meerkat_01[1,rx_el_min_index_meerkat_01:tx_el_max_index+1] >= math.radians(rx_el_min))# 06/09/17: debugged
rx_el_max_index_meerkat_01 = rx_el_max_range_meerkat_01[0][-1]+rx_el_min_index_meerkat_01;
# M002
rx_el_min_range_meerkat_02 = np.where( y_sph_rx_meerkat_02[1,tx_el_min_index:tx_el_max_index+1] >= math.radians(rx_el_min))
rx_el_min_index_meerkat_02 = rx_el_min_range_meerkat_02[0][0] + tx_el_min_index;
rx_el_max_range_meerkat_02 = np.where( y_sph_rx_meerkat_02[1,rx_el_min_index_meerkat_02:tx_el_max_index+1] >= math.radians(rx_el_min))# 06/09/17: debugged
rx_el_max_index_meerkat_02 = rx_el_max_range_meerkat_02[0][-1]+rx_el_min_index_meerkat_02;
# --------------------------------------------------------------------------- #
# Bounds for Rx field of regard
time_index_rx = np.zeros([2],dtype=np.int64);
time_index_rx[0] = max(rx_el_min_index,rx_el_min_index_meerkat_01,rx_el_min_index_meerkat_02);
time_index_rx[1] = min(rx_el_max_index,rx_el_max_index_meerkat_01,rx_el_max_index_meerkat_02);
print 'bounds for Rx FoR'
print time_index_rx[0]
print time_index_rx[1]

np.save('main_057_iss_06_time_index_rx.npy',time_index_rx);

span_az = math.degrees(y_sph_tx[2,time_index_rx[1]] - y_sph_tx[2,time_index_rx[0]]);
span_el = math.degrees(y_sph_tx[1,time_index_rx[1]] - y_sph_tx[1,time_index_rx[0]]);
print 'span in el and az. These should span at least the Tx beamwidth of %.3f' %math.degrees(beamwidth_tx)
print span_el
print span_az


print 'plotting results'
# --------------------------------------------------------------------------- #
# Plot bistatic geometry results
f, axarr = plt.subplots(6,sharex=True);
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
f.suptitle(r"\textbf{Look angles at Rx0, Rx1 \& Rx2 to object %s trajectory for %s}" %(norad_id,title_string) ,fontsize=12,y=1.01)


axarr[0].set_title(r'Elevation angle $\theta_{\text{Rx0}}~[\mathrm{^\circ}]$')
axarr[0].set_ylabel(r'$\theta_{\text{Rx}}$');
axarr[0].plot(timevec[tx_el_min_index:tx_el_max_index+1],np.rad2deg(y_sph_rx[1,tx_el_min_index:tx_el_max_index+1]));
#del_el_ticks=30;
#el_ticks_range=np.arange(15,90+del_el_ticks,del_el_ticks,dtype=np.int64)
#axarr[0].set_yticks(el_ticks_range);

axarr[0].axhline(rx_el_min,color='darkgreen',linestyle='dashed');
axarr[0].axhline(rx_el_max,color='darkgreen',linestyle='dashed');
axarr[0].axvspan(timevec[rx_el_min_index],timevec[rx_el_max_index],facecolor='yellow',alpha=0.2);
#del_az_ticks=40;
#az_ticks_range=np.arange(-120,40+del_az_ticks,del_az_ticks,dtype=np.int64)
#axarr[1].set_yticks(az_ticks_range);
axarr[1].set_title(r'Azimuth angle $\psi_{\text{Rx0}}~[\mathrm{^\circ}]$')
axarr[1].set_ylabel(r'$\psi_{\text{Rx}}$');
axarr[1].plot(timevec,np.rad2deg(y_sph_rx[2,:]));

axarr[2].set_title(r'Elevation angle $\theta_{\text{Rx1}}~[\mathrm{^\circ}]$')
axarr[2].set_ylabel(r'$\theta_{\text{Rx}}$');
axarr[2].plot(timevec[tx_el_min_index:tx_el_max_index+1],np.rad2deg(y_sph_rx_meerkat_01[1,tx_el_min_index:tx_el_max_index+1]));
#axarr[2].set_yticks(el_ticks_range);
axarr[2].axhline(rx_el_min,color='darkgreen',linestyle='dashed');
axarr[2].axhline(rx_el_max,color='darkgreen',linestyle='dashed');
axarr[2].axvspan(timevec[rx_el_min_index],timevec[rx_el_max_index],facecolor='yellow',alpha=0.2);
axarr[3].set_title(r'Azimuth angle $\psi_{\text{Rx1}}~[\mathrm{^\circ}]$')
#axarr[3].set_yticks(az_ticks_range);
axarr[3].set_ylabel(r'$\psi_{\text{Rx}}$');
axarr[3].plot(timevec,np.rad2deg(y_sph_rx_meerkat_01[2,:]));

axarr[4].set_title(r'Elevation angle $\theta_{\text{Rx2}}~[\mathrm{^\circ}]$')
axarr[4].set_ylabel(r'$\theta_{\text{Rx}}$');
#axarr[4].set_yticks(el_ticks_range);
axarr[4].plot(timevec[tx_el_min_index:tx_el_max_index+1],np.rad2deg(y_sph_rx_meerkat_02[1,tx_el_min_index:tx_el_max_index+1]));
axarr[4].axhline(rx_el_min,color='darkgreen',linestyle='dashed');
axarr[4].axhline(rx_el_max,color='darkgreen',linestyle='dashed');
axarr[4].axvspan(timevec[rx_el_min_index],timevec[rx_el_max_index],facecolor='yellow',alpha=0.2);
axarr[5].set_title(r'Azimuth angle $\psi_{\text{Rx2}}~[\mathrm{^\circ}]$')
axarr[5].set_ylabel(r'$\psi_{\text{Rx}}$');
#axarr[5].set_yticks(az_ticks_range);
axarr[5].plot(timevec,np.rad2deg(y_sph_rx_meerkat_02[2,:]));

axarr[5].set_xlabel(r'Time $t~[\mathrm{s}]$');
axarr[0].grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
axarr[1].grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
axarr[2].grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
axarr[3].grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
axarr[4].grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
axarr[5].grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')


at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
axarr[3].add_artist(at)
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0:5]], visible=False)
plt.subplots_adjust(hspace=0.6)
f.savefig('main_057_iss_06_rxangles.pdf',bbox_inches='tight',pad_inches=0.05,dpi=40)   
# --------------------------------------------------------------------------- #