# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:40:14 2017

@author: Ashiv Dhondea
"""

import GeometryFunctions as GF
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
from mpl_toolkits.mplot3d import Axes3D

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
		if 'HPBW Rx' in line:
			good_index = line.index('=')
			beamwidth_rx = float(line[good_index+1:-1]);
fp.close();
# --------------------------------------------------------------------------- #
# Bistatic Radar characteristics
beamwidth_rx = math.radians(beamwidth_rx );
# --------------------------------------------------------------------------- #
print 'Loading data'
timevec = np.load('main_057_iss_05_timevec.npy'); # timevector
y_sph_tx = np.load('main_057_iss_05_y_sph_tx.npy'); # spherical measurement vectors in Tx frame

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
norad_id = '25544'
# --------------------------------------------------------------------------- #
y_sph_rx = np.load('main_057_iss_05_y_sph_rx.npy'); # spherical measurement vectors in Rx frame
y_sph_rx_meerkat_01 = np.load('main_057_iss_05_y_sph_rx_meerkat_01.npy'); 
y_sph_rx_meerkat_02 = np.load('main_057_iss_05_y_sph_rx_meerkat_02.npy'); 

tx_beam_indices_best = np.load('main_057_iss_07_tx_beam_indices_best.npy');
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
# Find and compensate for the -pi to pi kink
az_rx_nice = GF.fnSmoothe_AngleSeries(y_sph_rx[2,:],2*math.pi);  

try:
	rx_az_range_up   = np.where(abs(az_rx_nice[tx_bw_time_max:latest_pt+1] - az_rx_nice[tx_bw_time_max]  )<= 0.5*beamwidth_rx );
	rx_az_range_down = np.where(abs(az_rx_nice[earliest_pt:tx_bw_time_max+1] - az_rx_nice[tx_bw_time_max]  )<= 0.5*beamwidth_rx);
except IndexError:
	print 'rx az range exceeds all available data'
# latest point where we can put the az beam's end and earliest point where we can put the az beam's start
rx_bw_az_index_end = rx_az_range_up[0][-1] + tx_bw_time_max; 
rx_bw_az_index_start = rx_az_range_down[0][0] + earliest_pt;
print 'start and end index of rx beam - az component'
print rx_bw_az_index_start 
print rx_bw_az_index_end 

try:
	rx_el_range_up   = np.where(abs(y_sph_rx[1,tx_bw_time_max:latest_pt+1] - y_sph_rx[1,tx_bw_time_max]  )<= 0.5*beamwidth_rx );
	rx_el_range_down = np.where(abs(y_sph_rx[1,earliest_pt:tx_bw_time_max+1] - y_sph_rx[1,tx_bw_time_max]  )<= 0.5*beamwidth_rx);
except IndexError:
	print 'rx el range exceeds all available data'
# latest point where we can put the el beam's end and earliest point where we can put the el beam's start
rx_bw_el_index_end = rx_el_range_up[0][-1] + tx_bw_time_max; 
rx_bw_el_index_start = rx_el_range_down[0][0] + earliest_pt;
print 'start and end index of rx beam - el component'
print rx_bw_el_index_start 
print rx_bw_el_index_end 

rx_bw_index_start = max(rx_bw_az_index_start,rx_bw_el_index_start);
rx_bw_index_end = min(rx_bw_az_index_end,rx_bw_el_index_end);
print 'chosen bounds for rx beam'
print rx_bw_index_start
print rx_bw_index_end

beam_centre_rx = np.degrees(np.array([az_rx_nice[tx_bw_time_max],y_sph_rx[1,tx_bw_time_max]]));

withincircle_rx = np.full([len(timevec)],False,dtype=bool);
for i in range(rx_bw_index_start,rx_bw_index_end+1):
    testpt = np.degrees(np.array([az_rx_nice[i],y_sph_rx[1,i]]));
    withincircle_rx[i] = GF.fnCheck_IsInCircle(beam_centre_rx,0.5*math.degrees(beamwidth_rx),testpt)

early = np.where(withincircle_rx[rx_bw_index_start:rx_bw_index_end+1] == True);
earliest_pt_rx = early[0][0] + rx_bw_index_start;
latest_pt_rx = early[0][-1] + rx_bw_index_start;

print 'earliest and latest pt rx'
print earliest_pt_rx
print latest_pt_rx

print 'rx0 beam indices saved'
rx0_beam_circ_index = np.array([earliest_pt_rx,tx_bw_time_max,latest_pt_rx],dtype=np.int64);
np.save('main_057_iss_09_rx0_beam_circ_index.npy',rx0_beam_circ_index);
# --------------------------------------------------------------------------- #
print 'place Rx1 beam above Rx0'
# Find and compensate for the -pi to pi kink
az_rx1_nice = GF.fnSmoothe_AngleSeries(y_sph_rx_meerkat_01[2,:],2*math.pi);

beam_centre_test = np.degrees(np.array([az_rx1_nice[latest_pt_rx],y_sph_rx_meerkat_01[1,latest_pt_rx]]));

withincircle_rx1 = np.full([len(timevec)],False,dtype=bool);
for i in range(latest_pt_rx,len(timevec)-1):
    testpt = np.degrees(np.array([az_rx1_nice[i],y_sph_rx_meerkat_01[1,i]]));
    withincircle_rx1[i] = GF.fnCheck_IsInCircle(beam_centre_test,0.5*math.degrees(beamwidth_rx),testpt)

early = np.where(withincircle_rx1[latest_pt_rx:len(timevec)-1] == True);
index_for_rx1 = early[0][-1] + latest_pt_rx;
print 'beam centre placement for rx1'
print index_for_rx1

beam_centre_test = np.degrees(np.array([az_rx1_nice[index_for_rx1],y_sph_rx_meerkat_01[1,index_for_rx1]]));

withincircle_rx1 = np.full([len(timevec)],False,dtype=bool);
for i in range(latest_pt_rx,len(timevec)-1):
    testpt = np.degrees(np.array([az_rx1_nice[i],y_sph_rx_meerkat_01[1,i]]));
    withincircle_rx1[i] = GF.fnCheck_IsInCircle(beam_centre_test,0.5*math.degrees(beamwidth_rx),testpt)

early = np.where(withincircle_rx1[latest_pt_rx:] == True);
earliest_pt_rx1 = early[0][0] + latest_pt_rx;
latest_pt_rx1 = early[0][-1] + latest_pt_rx;
print 'beam centre placement for rx1: the second one should be chosen as rx1 beam centre.'
print latest_pt_rx1 
print index_for_rx1
print earliest_pt_rx1

print 'rx1 beam indices saved'
rx1_beam_circ_index = np.array([earliest_pt_rx1,index_for_rx1,latest_pt_rx1],dtype=np.int64);
np.save('main_057_iss_09_rx1_beam_circ_index.npy',rx1_beam_circ_index)
# --------------------------------------------------------------------------- #
print 'place Rx2 beam below Rx0'
# Find and compensate for the -pi to pi kink
az_rx2_nice = GF.fnSmoothe_AngleSeries(y_sph_rx_meerkat_02[2,:],2*math.pi);

beam_centre_test = np.degrees(np.array([az_rx2_nice[earliest_pt_rx],y_sph_rx_meerkat_02[1,earliest_pt_rx]]));

withincircle_rx2 = np.full([len(timevec)],False,dtype=bool);
for i in range(0,earliest_pt_rx):
    testpt = np.degrees(np.array([az_rx2_nice[i],y_sph_rx_meerkat_02[1,i]]));
    withincircle_rx2[i] = GF.fnCheck_IsInCircle(beam_centre_test,0.5*math.degrees(beamwidth_rx),testpt)

early = np.where(withincircle_rx2[0:earliest_pt_rx] == True);
index_for_rx2 = early[0][0] ;
print 'beam centre placement for rx2'
print index_for_rx2

beam_centre_test = np.degrees(np.array([az_rx2_nice[index_for_rx2],y_sph_rx_meerkat_02[1,index_for_rx2]]));

withincircle_rx2 = np.full([len(timevec)],False,dtype=bool);
for i in range(0,earliest_pt_rx+1):
    testpt = np.degrees(np.array([az_rx2_nice[i],y_sph_rx_meerkat_02[1,i]]));
    withincircle_rx2[i] = GF.fnCheck_IsInCircle(beam_centre_test,0.5*math.degrees(beamwidth_rx),testpt)

early = np.where(withincircle_rx2[0:earliest_pt_rx+1] == True);
earliest_pt_rx2 = early[0][0] ;
latest_pt_rx2 = early[0][-1] ;
print 'beam centre placement for rx2: the second one should be chosen as rx2 beam centre.'
print latest_pt_rx2 
print index_for_rx2
print earliest_pt_rx2

print 'rx2 beam indices saved'
rx2_beam_circ_index = np.array([earliest_pt_rx2,index_for_rx2,latest_pt_rx2],dtype=np.int64);
np.save('main_057_iss_09_rx2_beam_circ_index.npy',rx2_beam_circ_index)
# --------------------------------------------------------------------------- #
# Find the epoch of the relevant data points
plt_start_index = tx_beam_index_down 
plt_end_index = tx_beam_index_up+1 

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
title_string = str(start_epoch_test.isoformat())+'Z/'+str(end_epoch_test .isoformat())+'Z';
tx_beam_index_down_epoch = tx_beam_index_down_epoch.replace(tzinfo=None);
tx_beam_index_up_epoch = tx_beam_index_up_epoch.replace(tzinfo=None)
tx_bw_time_max_epoch = tx_bw_time_max_epoch.replace(tzinfo=None)
latest_pt_epoch= latest_pt_epoch.replace(tzinfo=None)

print 'epoch of tx beam points of interest found'
# --------------------------------------------------------------------------- #
## Find timestamps for the rx0, 1, 2 beam stuff
latest_pt_rx_epoch =THF.fnCalculate_DatetimeEpoch(timevec,latest_pt_rx,experiment_timestamps[0]);
earliest_pt_rx_epoch = THF.fnCalculate_DatetimeEpoch(timevec,earliest_pt_rx,experiment_timestamps[0]);
latest_pt_rx_epoch= latest_pt_rx_epoch.replace(tzinfo=None)
earliest_pt_rx_epoch = earliest_pt_rx_epoch.replace(tzinfo=None);

latest_pt_rx1_epoch =  THF.fnCalculate_DatetimeEpoch(timevec,latest_pt_rx1,experiment_timestamps[0]);
earliest_pt_rx1_epoch =  THF.fnCalculate_DatetimeEpoch(timevec,earliest_pt_rx1,experiment_timestamps[0]);
latest_pt_rx1_epoch= latest_pt_rx1_epoch.replace(tzinfo=None)
earliest_pt_rx1_epoch = earliest_pt_rx1_epoch.replace(tzinfo=None);

index_for_rx1_epoch =  THF.fnCalculate_DatetimeEpoch(timevec,index_for_rx1,experiment_timestamps[0]);
index_for_rx1_epoch= index_for_rx1_epoch.replace(tzinfo=None)

index_for_rx2_epoch = THF.fnCalculate_DatetimeEpoch(timevec,index_for_rx2,experiment_timestamps[0]);
index_for_rx2_epoch= index_for_rx2_epoch.replace(tzinfo=None)

latest_pt_rx2_epoch = THF.fnCalculate_DatetimeEpoch(timevec,latest_pt_rx2,experiment_timestamps[0]);
earliest_pt_rx2_epoch = THF.fnCalculate_DatetimeEpoch(timevec,earliest_pt_rx2,experiment_timestamps[0]);
latest_pt_rx2_epoch= latest_pt_rx2_epoch.replace(tzinfo=None)
earliest_pt_rx2_epoch = earliest_pt_rx2_epoch.replace(tzinfo=None);
# --------------------------------------------------------------------------- #
fig = plt.figure(1);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.axis('equal')
fig.suptitle(r"\textbf{Rx0 beam placement for the object %s trajectory during the interval %s}" %(norad_id,title_string),fontsize=10);
plt.plot(np.degrees(y_sph_rx[2,plt_start_index:tx_beam_index_down]),np.degrees(y_sph_rx[1,plt_start_index:tx_beam_index_down]),color='blue',linestyle='dashed')
plt.plot(np.degrees(y_sph_rx[2,tx_beam_index_down+1:earliest_pt_rx]),np.degrees(y_sph_rx[1,tx_beam_index_down+1:earliest_pt_rx]),color='red')
plt.plot(np.degrees(y_sph_rx[2,earliest_pt_rx:latest_pt_rx+1]),np.degrees(y_sph_rx[1,earliest_pt_rx:latest_pt_rx+1]),color='darkgreen')
plt.plot(np.degrees(y_sph_rx[2,latest_pt_rx+1:tx_beam_index_up+1]),np.degrees(y_sph_rx[1,latest_pt_rx+1:tx_beam_index_up+1]),color='red')
plt.plot(np.degrees(y_sph_rx[2,tx_beam_index_up+1:plt_end_index]),np.degrees(y_sph_rx[1,tx_beam_index_up+1:plt_end_index]),color='blue',linestyle='dashed')

plt.scatter(np.degrees(y_sph_rx[2,earliest_pt]),np.degrees(y_sph_rx[1,earliest_pt]),s=50,marker=r"$\square$",facecolors='none', edgecolors='darkgreen',label=r"%s" %str(earliest_pt_epoch.isoformat())+'Z');
plt.scatter(np.degrees(y_sph_rx[2,earliest_pt_rx]),np.degrees(y_sph_rx[1,earliest_pt_rx]),s=50,marker=r"$\diamond$",facecolors='none',edgecolors='navy',label=r"%s" %str(earliest_pt_rx_epoch.isoformat())+'Z');
plt.scatter(np.degrees(y_sph_rx[2,tx_bw_time_max]),np.degrees(y_sph_rx[1,tx_bw_time_max]),s=50,marker=r"$\star$",facecolors='none',edgecolors='darkgreen',label=r"%s" %str(tx_bw_time_max_epoch.isoformat())+'Z');
plt.scatter(np.degrees(y_sph_rx[2,latest_pt_rx]),np.degrees(y_sph_rx[1,latest_pt_rx]),s=50,marker=r"$\boxplus$",facecolors='none', edgecolors='navy',label=r"%s" %str(latest_pt_rx_epoch.isoformat())+'Z');
plt.scatter(np.degrees(y_sph_rx[2,latest_pt]),np.degrees(y_sph_rx[1,latest_pt]),s=50,marker=r"$\otimes$",facecolors='none', edgecolors='darkgreen',label=r"%s" %str(latest_pt_epoch.isoformat())+'Z')
plt.legend(loc=1)
for p in [
    patches.Circle(
        (np.degrees(y_sph_rx[2,tx_bw_time_max]),np.degrees(y_sph_rx[1,tx_bw_time_max])),0.5*math.degrees(beamwidth_rx),
        color = 'gray',
        alpha=0.25
    ),
]:
    ax.add_patch(p)

ax.set_xlabel(r'Azimuth angle $\psi_{\text{Rx}}~[\mathrm{^\circ}]$')
ax.set_ylabel(r'Elevation angle $\theta_{\text{Rx}}~[\mathrm{^\circ}]$'); 

plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_09_rxbeam_circ.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10)
# --------------------------------------------------------------------------- #
print 'cool cool cool'
