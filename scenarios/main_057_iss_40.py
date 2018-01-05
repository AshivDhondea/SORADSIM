# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 21:06:25 2017

based on main_011_iss_26.py

Answering the question of whether a polynomial function is a good fit for the
tracklet.

see notes on pg 32 in blue notebook

@author: Ashiv Dhondea
"""

import AstroFunctions as AstFn
import AstroConstants as AstCnst
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
		if 'HPBW Tx' in line:
			good_index = line.index('=')
			beamwidth_tx = float(line[good_index+1:-1]);

fp.close();
# --------------------------------------------------------------------------- #
print 'Loading data'
timevec = np.load('main_057_iss_05_timevec.npy'); # timevector
x_target = np.load('main_057_iss_05_x_target.npy'); # state vector in SEZ frame
theta_GMST = np.load('main_057_iss_05_theta_GMST.npy'); # GMST angles in rad
y_sph_tx = np.load('main_057_iss_05_y_sph_tx.npy'); # spherical measurement vectors in Tx frame
y_sph_rx = np.load('main_057_iss_05_y_sph_rx.npy'); # spherical measurement vectors in Rx frame
y_sph_rx_meerkat_01 = np.load('main_057_iss_05_y_sph_rx_meerkat_01.npy'); 
y_sph_rx_meerkat_02 = np.load('main_057_iss_05_y_sph_rx_meerkat_02.npy');
tx_beam_indices_best = np.load('main_057_iss_07_tx_beam_indices_best.npy');

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
print 'Loading data'
timevec = np.load('main_057_iss_05_timevec.npy'); # timevector

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
#plot_lim = 6
plt_start_index = earliest_pt #- int(plot_lim/delta_t)
plt_end_index = latest_pt #+ int(plot_lim/delta_t) 
print 'duration = %f' %( (plt_end_index-plt_start_index)*delta_t)

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
## right ascension and declination to target
ra = np.zeros([len(timevec)],dtype=np.float64);
dec = np.zeros([len(timevec)],dtype=np.float64);

for index in range(plt_start_index,plt_end_index+1):
    valladoazim = math.pi - y_sph_rx[2,index];
    RA, DEC = AstFn.fnConvert_AZEL_to_Topo_RADEC(y_sph_rx[1,index],valladoazim,math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),theta_GMST[index]);
    ra[index] =  math.degrees(RA);
    dec[index] =  math.degrees(DEC);

# Find and compensate for the -pi to pi kink 
beam_centre = np.degrees(np.array([y_sph_rx[2,tx_bw_time_max],y_sph_rx[1,tx_bw_time_max]],dtype=np.float64));
numpts=360
circpts = GF.fnCalculate_CircumferencePoints(beam_centre,0.5*math.degrees(beamwidth_rx),numpts)
ra_rx0 = np.zeros([numpts],dtype=np.float64);
dec_rx0 = np.zeros([numpts],dtype=np.float64);
for index in range(numpts):
    valladoazim = math.pi - math.radians(circpts[0,index]);
    RA, DEC = AstFn.fnConvert_AZEL_to_Topo_RADEC(math.radians(circpts[1,index]),valladoazim,math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),theta_GMST[tx_bw_time_max]);
    ra_rx0[index] =  math.degrees(RA);
    dec_rx0[index] =  math.degrees(DEC);
    
beam_centre_01 = np.degrees(np.array([y_sph_rx[2,index_for_rx1 ],y_sph_rx[1,index_for_rx1 ]],dtype=np.float64));
numpts=360
circpts = GF.fnCalculate_CircumferencePoints(beam_centre_01,0.5*math.degrees(beamwidth_rx),numpts)
ra_rx1 = np.zeros([numpts],dtype=np.float64);
dec_rx1 = np.zeros([numpts],dtype=np.float64);
for index in range(numpts):
    valladoazim = math.pi - math.radians(circpts[0,index]);
    RA, DEC = AstFn.fnConvert_AZEL_to_Topo_RADEC(math.radians(circpts[1,index]),valladoazim,math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),theta_GMST[tx_bw_time_max]);
    ra_rx1[index] =  math.degrees(RA);
    dec_rx1[index] =  math.degrees(DEC);

beam_centre_02 = np.degrees(np.array([y_sph_rx[2,index_for_rx2 ],y_sph_rx[1,index_for_rx2 ]],dtype=np.float64));
numpts=360
circpts = GF.fnCalculate_CircumferencePoints(beam_centre_02,0.5*math.degrees(beamwidth_rx),numpts)
ra_rx2 = np.zeros([numpts],dtype=np.float64);
dec_rx2 = np.zeros([numpts],dtype=np.float64);
for index in range(numpts):
    valladoazim = math.pi - math.radians(circpts[0,index]);
    RA, DEC = AstFn.fnConvert_AZEL_to_Topo_RADEC(math.radians(circpts[1,index]),valladoazim,math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),theta_GMST[tx_bw_time_max]);
    ra_rx2[index] =  math.degrees(RA);
    dec_rx2[index] =  math.degrees(DEC);

# --------------------------------------------------------------------------- #
# Normalize everything to match with the paper
beam_centre_main = np.array([ra[tx_bw_time_max],dec[tx_bw_time_max]]);
new_beam_centre_main =  beam_centre_main  - beam_centre_main ;
new_ra =  ra - beam_centre_main[0];
new_dec = dec - beam_centre_main [1];

new_ra_rx0 = ra_rx0 - beam_centre_main[0];
new_dec_rx0 = dec_rx0 - beam_centre_main[1];

new_ra_rx1 = ra_rx1 - beam_centre_main[0];
new_dec_rx1 = dec_rx1 - beam_centre_main[1];

new_ra_rx2 = ra_rx2 - beam_centre_main[0];
new_dec_rx2 = dec_rx2 - beam_centre_main[1];
# --------------------------------------------------------------------------- #
# polynomial fitting to ra and dec data [derived from main_011_iss_23.py]
new_ra_poly_coeffs = np.polyfit(timevec[plt_start_index:plt_end_index+1],new_ra[plt_start_index:plt_end_index+1],2);
new_ra_poly =  np.poly1d(new_ra_poly_coeffs);

new_dec_poly_coeffs = np.polyfit(timevec[plt_start_index:plt_end_index+1],new_dec[plt_start_index:plt_end_index+1],2);
new_dec_poly =  np.poly1d(new_dec_poly_coeffs);

new_ra_poly_coords =  np.zeros(len(timevec),dtype=np.float64);
new_dec_poly_coords =  np.zeros(len(timevec),dtype=np.float64);

for index in range(plt_start_index,plt_end_index+1):
    new_ra_poly_coords[index] = new_ra_poly(timevec[index]);
    new_dec_poly_coords[index] = new_dec_poly(timevec[index]);
    
# error in estimated polynomials
new_ra_error = new_ra_poly_coords - new_ra;
new_dec_error = new_dec_poly_coords - new_dec; 
# --------------------------------------------------------------------------- #
"""
fig = plt.figure(1);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.axis('equal')
fig.suptitle(r"\textbf{Topocentric Right Ascension \& Declination for object %s during the interval %s}" %(norad_id,title_string),fontsize=9);
plt.plot(new_ra[plt_start_index:tx_beam_index_down],new_dec[plt_start_index:tx_beam_index_down],color='blue',linestyle='dashed');
plt.plot(new_ra[tx_beam_index_up+1:plt_end_index],new_dec[tx_beam_index_up+1:plt_end_index],color='blue',linestyle='dashed');
plt.plot(new_ra[tx_beam_index_down:tx_beam_index_up],new_dec[tx_beam_index_down:tx_beam_index_up],color='blue');

plt.plot(new_ra_rx0,new_dec_rx0,color='gray');
plt.plot(new_ra_rx1,new_dec_rx1,color='gray');
plt.plot(new_ra_rx2,new_dec_rx2,color='gray');

plt.annotate(r"\textbf{ %s}" %meerkat_id[0],(new_ra[tx_bw_time_max]+0.1,new_dec[tx_bw_time_max]));
plt.annotate(r"\textbf{ %s}" %meerkat_id[1],(new_ra[index_for_rx1]+0.1,new_dec[index_for_rx1]));
plt.annotate(r"\textbf{ %s}" %meerkat_id[2],(new_ra[index_for_rx2]+0.1,new_dec[index_for_rx2]));

plt.scatter(new_ra[earliest_pt],new_dec[earliest_pt],s=50,marker=r"$\star$",facecolors='none', edgecolors='darkgreen',label=r"%s" %str(earliest_pt_epoch.isoformat()));
plt.scatter(new_ra[earliest_pt_rx],new_dec[earliest_pt_rx],s=50,marker=r"$\bigcirc$",facecolors='none',edgecolors='navy',label=r"%s" %str(earliest_pt_rx_epoch.isoformat()));
plt.scatter(new_ra[tx_bw_time_max],new_dec[tx_bw_time_max],s=50,marker=r"$\oplus$",facecolors='none',edgecolors='darkgreen',label=r"%s" %str(tx_bw_time_max_epoch.isoformat()));
plt.scatter(new_ra[latest_pt_rx],new_dec[latest_pt_rx],s=50,marker=r"$\boxplus$",facecolors='none', edgecolors='navy',label=r"%s" %str(latest_pt_rx_epoch.isoformat()));
plt.scatter(new_ra[latest_pt],new_dec[latest_pt],s=50,marker=r"$\diamond$",facecolors='none', edgecolors='darkgreen',label=r"%s" %str(latest_pt_epoch.isoformat()))

plt.scatter(new_ra[index_for_rx1],new_dec[index_for_rx1],s=50,marker=r"$\circleddash$",facecolors='none', edgecolors='orangered',label=r"%s" %str(index_for_rx1_epoch.isoformat()))
plt.scatter(new_ra[earliest_pt_rx1],new_dec[earliest_pt_rx1],s=50,marker=r"$\square$",facecolors='none', edgecolors='orangered',label=r"%s" %str(earliest_pt_rx1_epoch.isoformat()));
plt.scatter(new_ra[latest_pt_rx1],new_dec[latest_pt_rx1],s=50,marker=r"$\bullet$",facecolors='none', edgecolors='orangered',label=r"%s" %str(latest_pt_rx1_epoch.isoformat()))

plt.scatter(new_ra[index_for_rx2],new_dec[index_for_rx2],s=50,marker=r"$\circledcirc$",facecolors='none', edgecolors='purple',label=r"%s" %str(index_for_rx2_epoch.isoformat()))
plt.scatter(new_ra[earliest_pt_rx2],new_dec[earliest_pt_rx2],s=50,marker=r"$\boxtimes$",facecolors='none', edgecolors='purple',label=r"%s" %str(earliest_pt_rx2_epoch.isoformat()));
plt.scatter(new_ra[latest_pt_rx2],new_dec[latest_pt_rx2],s=50,marker=r"$\times$",facecolors='none', edgecolors='purple',label=r"%s" %str(latest_pt_rx2_epoch.isoformat()))

##
plt.plot(new_ra_poly_coords[plt_start_index:plt_end_index+1],new_dec_poly_coords[plt_start_index:plt_end_index+1],color='red');

ax.set_xlabel(r'Topocentric right ascension $\Delta \alpha_{t}~[\mathrm{^\circ}]$')
ax.set_ylabel(r'Topocentric declination $\Delta \delta_{t}~[\mathrm{^\circ}]$');

plt.legend(loc='center left',title=r"\textbf{Timestamps}",bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True)

plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_40_radec_curvefit.pdf',bbox_inches='tight',pad_inches=0.11,dpi=10);
"""
# --------------------------------------------------------------------------- #
"""
fig = plt.figure(2);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.axis('equal')
skip = 270;
fig.suptitle(r"\textbf{Errors in fitting polynomials to the} $(\Delta \alpha_t,\Delta \delta_t)$ \textbf{profile of object %s during the interval %s}" %(norad_id,title_string) ,fontsize=12);
plt.plot(timevec[plt_start_index:plt_end_index+1:skip+20],new_ra_error[plt_start_index:plt_end_index+1:skip+20],linewidth=0.5,label=r"$\Delta \alpha_t - \Delta \hat{\alpha}_t$");
plt.plot(timevec[plt_start_index:plt_end_index+1:skip],new_dec_error[plt_start_index:plt_end_index+1:skip],linestyle='dashed',linewidth=0.5,label=r"$\Delta \delta_t - \Delta \hat{\delta}_t$");
plt.annotate(r"$\Delta \hat{\alpha}_t(t) =  a_2 t^2 + a_1 t + a_0$",(5,0.0013) );
plt.annotate(r"$\Delta \hat{\delta_t}(t) =  b_2 t^2 + b_1 t + b_0$",(5,0.0011) );
plt.legend(loc=4)
ax.set_ylabel(r'Error in angle $[\mathrm{^\circ}]$')
ax.set_xlabel(r'Time $ t~[\mathrm{s}]$');
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_40_polynom_error.pdf',bbox_inches='tight',pad_inches=0.11,dpi=10)
"""

fig = plt.figure(3);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.axis('equal')
skip = 270;
fig.suptitle(r"\textbf{Errors in fitting polynomials to the TOPORADEC profile of object %s}" %(norad_id) ,fontsize=12);
plt.plot(timevec[plt_start_index:plt_end_index+1:skip+300],new_ra_error[plt_start_index:plt_end_index+1:skip+300],linewidth=0.5,label=r"$\Delta \alpha_t - \Delta \hat{\alpha}_t$");
plt.plot(timevec[plt_start_index:plt_end_index+1:skip],new_dec_error[plt_start_index:plt_end_index+1:skip],linestyle='dashed',linewidth=0.5,label=r"$\Delta \delta_t - \Delta \hat{\delta}_t$");
plt.annotate(r"$\Delta \hat{\alpha}_t(t) =  a_2 t^2 + a_1 t + a_0$",(5,0.0013) );
plt.annotate(r"$\Delta \hat{\delta_t}(t) =  b_2 t^2 + b_1 t + b_0$",(5,0.0011) );
plt.legend(loc=4)
ax.set_ylabel(r'Error in angle $[\mathrm{^\circ}]$')
ax.set_xlabel(r'Time $ t~[\mathrm{s}]$');
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_40_polynom_error1.pdf',bbox_inches='tight',pad_inches=0.11,dpi=10)