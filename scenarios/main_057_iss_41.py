# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 22:42:05 2017

based on main_011_iss_30.py

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
		if 'centre_frequency' in line:
			good_index = line.index('=')
			centre_frequency = float(line[good_index+1:-1]); 
		if 'HPBW Rx' in line:
			good_index = line.index('=')
			beamwidth_rx = float(line[good_index+1:-1]);
		if 'HPBW Tx' in line:
			good_index = line.index('=')
			beamwidth_tx = float(line[good_index+1:-1]);
		if 'Gain Rx' in line:
			good_index = line.index('=')
			gain_rx_dB = float(line[good_index+1:-1]);
		if 'Gain Tx' in line:
			good_index = line.index('=')
			gain_tx_dB = float(line[good_index+1:-1]);
		if 'HPBW Tx' in line:
			good_index = line.index('=')
			beamwidth_tx = float(line[good_index+1:-1]);
		if 'bandwidth' in line:
			good_index = line.index('=')
			bandwidth = float(line[good_index+1:-1]); 
		if 'system temperature' in line:
			good_index = line.index('=')
			meerkat_sys_temp = float(line[good_index+1:-1]); 
		if 'transmitted power' in line:
			good_index = line.index('=')
			P_Tx = float(line[good_index+1:-1]); 
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
beamwidth_tx = math.radians(beamwidth_tx);

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
#plot_lim = #6
plt_start_index = earliest_pt#tx_beam_index_down - int(plot_lim/delta_t)
plt_end_index = latest_pt#tx_beam_index_up+1 + int(plot_lim/delta_t) 

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
earliest_pt_rx2_epoch = earliest_pt_rx2_epoch.replace(tzinfo=None)
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
# Find the true values of the coefficients in the polynomial in eqn 8/
# This block of code comes from main_011_iss_26.py

# polynomial fitting to ra and dec data [derived from main_011_iss_23.py]
new_ra_poly_coeffs = np.polyfit(timevec[earliest_pt_rx2:latest_pt_rx1+1],new_ra[earliest_pt_rx2:latest_pt_rx1+1],2);
new_ra_poly =  np.poly1d(new_ra_poly_coeffs);

new_dec_poly_coeffs = np.polyfit(timevec[earliest_pt_rx2:latest_pt_rx1+1],new_dec[earliest_pt_rx2:latest_pt_rx1+1],2);
new_dec_poly =  np.poly1d(new_dec_poly_coeffs);

new_ra_poly_coords =  np.zeros(len(timevec),dtype=np.float64);
new_dec_poly_coords =  np.zeros(len(timevec),dtype=np.float64);

for index in range(plt_start_index,plt_end_index+1):
    new_ra_poly_coords[index] = new_ra_poly(timevec[index]);
    new_dec_poly_coords[index] = new_dec_poly(timevec[index]);
    
# --------------------------------------------------------------------------- #
print 'loading SNR data'
snr_rx0 = np.load('main_057_iss_12_snr_rx0.npy');
snr_rx1 = np.load('main_057_iss_12_snr_rx1.npy');
snr_rx2 = np.load('main_057_iss_12_snr_rx2.npy');

snr_rx0_dB = RS.fn_Power_to_dB(snr_rx0);
snr_rx1_dB = RS.fn_Power_to_dB(snr_rx1);
snr_rx2_dB = RS.fn_Power_to_dB(snr_rx2);

timevec_cpi_list_2 = np.load('main_057_iss_12_timevec_cpi_list_2.npy');
timevec_cpi_list_1 = np.load('main_057_iss_12_timevec_cpi_list_1.npy');
timevec_cpi_list_0 = np.load('main_057_iss_12_timevec_cpi_list_0.npy');

timevec_cpi_2 = np.load('main_057_iss_12_timevec_cpi_2.npy');
timevec_cpi_1 = np.load('main_057_iss_12_timevec_cpi_1.npy');
timevec_cpi_0 = np.load('main_057_iss_12_timevec_cpi_0.npy');

snr_coh_2 = np.load('main_057_iss_12_snr_coh_2.npy');
snr_coh_1 = np.load('main_057_iss_12_snr_coh_1.npy');
snr_coh_0 = np.load('main_057_iss_12_snr_coh_0.npy');

snr_coh_2_dB = RS.fn_Power_to_dB(snr_coh_2);
snr_coh_1_dB = RS.fn_Power_to_dB(snr_coh_1);
snr_coh_0_dB = RS.fn_Power_to_dB(snr_coh_0);
# --------------------------------------------------------------------------- #

def fnCalculate_italradec_Norm_Weights(snr_coh):
    """
    Finding the weights for the curve fitting in S1-2 on pg 6 of
    "PERFORMANCE ASSESSMENT OF THE MULTIBEAM RADAR
    SENSOR BIRALES FOR SPACE SURVEILLANCE AND TRACKING", Di Lizia, Mazari et al.
    
    Created: 12 July 2017
    """
    index_max_snr = np.argmax(snr_coh);
    max_snr = max(snr_coh);
    return index_max_snr, snr_coh/max_snr

def fnCalculate_italradec_Curve_Fit(timevector,norm_snr_coh,ra):
    """
    Finding the set of polynomial coefficients which appear in the polynomials
    in equation 8 in 
     "PERFORMANCE ASSESSMENT OF THE MULTIBEAM RADAR
    SENSOR BIRALES FOR SPACE SURVEILLANCE AND TRACKING", Di Lizia, Mazari et al.
    
    Created: 12 July 2017
    Edited: 
    13/07/17: edited further.
    23/07/17: edited further in main_011_iss_30.py
    """    
    numpts = np.shape(timevector)[0];
    dy = 3; # a degree 2 polynomial
    y = ra#*np.ones([numpts],dtype=np.float64);
    RynInv = np.diag(norm_snr_coh);
    Tmat = np.zeros([numpts,dy],dtype=np.float64);
    for i in range(numpts):
        Tmat_Row = np.polynomial.polynomial.polyvander(timevector[i],dy-1)
        Tmat[i,:] = np.fliplr(Tmat_Row);
    TmatT = Tmat.T;
    S_hat = np.linalg.pinv(np.dot(TmatT,np.dot(RynInv,Tmat)));
    poly_coeff = np.dot(S_hat,np.dot(TmatT,np.dot(RynInv,y)));
    return poly_coeff,Tmat,S_hat
# --------------------------------------------------------------------------- #
index_max_snr_2,norm_snr_coh_2 = fnCalculate_italradec_Norm_Weights(snr_coh_2);
index_max_snr_1,norm_snr_coh_1 = fnCalculate_italradec_Norm_Weights(snr_coh_1);
index_max_snr_0,norm_snr_coh_0 = fnCalculate_italradec_Norm_Weights(snr_coh_0);

# be careful with the indices in the following arrays, they have to be mapped back
# to the indices for each dish's snr time series.
three_indices_overall =  np.array([index_max_snr_2,index_max_snr_0,index_max_snr_1],dtype=np.int64)
timevec_overall = np.array([timevec_cpi_2[three_indices_overall[0]],timevec_cpi_0[three_indices_overall[1]],timevec_cpi_1[three_indices_overall[2]]],dtype=np.float64)
snr_coh_overall = np.array([snr_coh_2[three_indices_overall[0]],snr_coh_0[three_indices_overall[1]],snr_coh_1[three_indices_overall[2]]]);

index_sorted_overall = np.argsort(snr_coh_overall);
index_max_snr_overall = index_sorted_overall[-1];
norm_snr_coh_overall = snr_coh_overall/snr_coh_overall[index_max_snr_overall]; # weights for curve fit

#new_ra_overall = np.array([new_ra[index_for_rx2],new_ra[index_for_rx0],new_ra[index_for_rx1]],dtype=np.float64);
#new_dec_overall = np.array([new_dec[index_for_rx2],new_dec[index_for_rx0],new_dec[index_for_rx1]],dtype=np.float64);

new_ra_overall = np.array([new_ra[earliest_pt],new_ra[index_for_rx0],new_ra[latest_pt]],dtype=np.float64);
new_dec_overall = np.array([new_dec[earliest_pt],new_dec[index_for_rx0],new_dec[latest_pt]],dtype=np.float64);
# these are the correct indices to use (see pg 36 in notebook)

polycoeff_ra,Tmat,S_hat_ra = fnCalculate_italradec_Curve_Fit(timevec_overall,norm_snr_coh_overall,new_ra_overall);
polycoeff_dec,Tmat,S_hat_dec = fnCalculate_italradec_Curve_Fit(timevec_overall,norm_snr_coh_overall,new_dec_overall);

polycoeff_ra_poly =  np.poly1d(polycoeff_ra);
polycoeff_dec_poly =  np.poly1d(polycoeff_dec);

polycoeff_ra_poly_coords =  np.zeros(len(timevec),dtype=np.float64);
polycoeff_dec_poly_coords =  np.zeros(len(timevec),dtype=np.float64);

for index in range(plt_start_index,plt_end_index+1):
    polycoeff_ra_poly_coords[index] = polycoeff_ra_poly(timevec[index]);
    polycoeff_dec_poly_coords[index] = polycoeff_dec_poly(timevec[index]);
# --------------------------------------------------------------------------- #
# error in estimated polynomials
new_ra_error = new_ra_poly_coords - new_ra;
new_dec_error = new_dec_poly_coords - new_dec; 

# error in estimated polynomials
polycoeff_ra_error = polycoeff_ra_poly_coords - new_ra;
polycoeff_dec_error = polycoeff_dec_poly_coords - new_dec; 

polycoeff_max_error = max(np.absolute(polycoeff_ra_error[earliest_pt:latest_pt+1]).max(),np.absolute(polycoeff_dec_error[earliest_pt:latest_pt+1]).max());

fig = plt.figure(2);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.axis('equal')
fig.suptitle(r"\textbf{Polynomial fit for the topocentric right ascension \& declination profile for object 25544 during the interval %s}" %title_string ,fontsize=10);
skip = 50;
plt.plot(timevec[earliest_pt:latest_pt+1:skip],polycoeff_ra_error[earliest_pt:latest_pt+1:skip],label=r"$\Delta \alpha_t - \Delta \hat{\alpha}_t$");
plt.plot(timevec[earliest_pt:latest_pt+1:skip],polycoeff_dec_error[earliest_pt:latest_pt+1:skip],linestyle='dashed',label=r"$\Delta \delta_t - \Delta \hat{\delta}_t$");


#plt.annotate(r"$\Delta \hat{\alpha}_t(t) =  a_2 t^2 + a_1 t + a_0$",(27,1.) );
#plt.annotate(r"$\Delta \hat{\delta_t}(t) =  b_2 t^2 + b_1 t + b_0$",(27,0.9) );
plt.legend(loc='best')
ax.set_ylabel(r'Error in angle $[\mathrm{^\circ}]$')
ax.set_xlabel(r'Time $ t~[\mathrm{s}]$');
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_41_polynom_error1.pdf',bbox_inches='tight',pad_inches=0.11,dpi=10)

fig = plt.figure(3);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.axis('equal')
fig.suptitle(r"\textbf{Polynomial fit for the TOPORADEC profile of object 25544}" ,fontsize=12);
skip = 50;
plt.plot(timevec[earliest_pt:latest_pt+1:skip],polycoeff_ra_error[earliest_pt:latest_pt+1:skip],label=r"$\Delta \alpha_t - \Delta \hat{\alpha}_t$");
plt.plot(timevec[earliest_pt:latest_pt+1:skip],polycoeff_dec_error[earliest_pt:latest_pt+1:skip],linestyle='dashed',label=r"$\Delta \delta_t - \Delta \hat{\delta}_t$");


#plt.annotate(r"$\Delta \hat{\alpha}_t(t) =  a_2 t^2 + a_1 t + a_0$",(27,1.) );
#plt.annotate(r"$\Delta \hat{\delta_t}(t) =  b_2 t^2 + b_1 t + b_0$",(27,0.9) );
plt.legend(loc='best')
ax.set_ylabel(r'Error in angle $[\mathrm{^\circ}]$')
ax.set_xlabel(r'Time $ t~[\mathrm{s}]$');
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_41_polynom_error01.pdf',bbox_inches='tight',pad_inches=0.11,dpi=10)