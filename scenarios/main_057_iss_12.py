# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 00:15:03 2017

SNR calculations

Heavily based on the paper
"PERFORMANCE ASSESSMENT OF THE MULTIBEAM RADAR
SENSOR BIRALES FOR SPACE SURVEILLANCE AND TRACKING", Di Lizia, Mazari et al.


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
        if 'PRF' in line:
            good_index = line.index('=')
            pulse_repetition_frequency = float(line[good_index+1:-1]); 
            pulse_repetition_frequency = int(pulse_repetition_frequency)
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
#experiment_timestamps[-1] = experiment_timestamps[-1].replace(tzinfo=None)
#title_string1 = str(experiment_timestamps[0].isoformat())+'Z/'+str(experiment_timestamps[-1].isoformat())+'Z';
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
#print 'Loading data'
#timevec = np.load('main_057_iss_05_timevec.npy'); # timevector
#
## discretization step length/PRF
#delta_t = timevec[2]-timevec[1];
## time stamps
#experiment_timestamps = [None]*len(timevec)
#index=0;
#with open('main_057_iss_05_experiment_timestamps.txt') as fp:
#    for line in fp:
#        modified_timestring = line[:-8];
#        experiment_timestamps[index] = aniso8601.parse_datetime(modified_timestring);
#        index+=1;
#fp.close();
norad_id = '25544'
# --------------------------------------------------------------------------- #
#y_sph_rx = np.load('main_057_iss_05_y_sph_rx.npy'); # spherical measurement vectors in Rx frame
#y_sph_rx_meerkat_01 = np.load('main_057_iss_05_y_sph_rx_meerkat_01.npy'); 
#y_sph_rx_meerkat_02 = np.load('main_057_iss_05_y_sph_rx_meerkat_02.npy'); 
#theta_GMST = np.load('main_057_iss_05_theta_GMST.npy');
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
plot_lim = 3
plt_start_index = tx_beam_index_down - int(plot_lim/delta_t)
plt_end_index = tx_beam_index_up+1 + int(plot_lim/delta_t) 

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
# Finding beamwidth in RA and DEC for each receiver
beam_a = 0.5*(ra_rx0.max() - ra_rx0.min());
beam_b = 0.5*(dec_rx0.max() - dec_rx0.min());

beamwidth_ra_0 = beam_a*2.;
beamwidth_dec_0 = beam_b*2.;

beam_a = 0.5*(ra_rx1.max() - ra_rx1.min());
beam_b = 0.5*(dec_rx1.max() - dec_rx1.min());

beamwidth_ra_1 = beam_a*2.;
beamwidth_dec_1 = beam_b*2.;

beam_a = 0.5*(ra_rx2.max() - ra_rx2.min());
beam_b = 0.5*(dec_rx2.max() - dec_rx2.min());

beamwidth_ra_2 = beam_a*2.;
beamwidth_dec_2 = beam_b*2.;
# --------------------------------------------------------------------------- #

gain_tx = RS.fn_dB_to_Power(gain_tx_dB);
gain_rx = RS.fn_dB_to_Power(gain_rx_dB); # uncompensated gain
RCS =401.801#RS.fn_dB_to_Power(40); # [m^2] this is the monostatic RCS of the ISS 

P_Rx0 = np.ones([len(timevec)],dtype=np.float64);
P_Rx1 = np.ones([len(timevec)],dtype=np.float64);
P_Rx2 = np.ones([len(timevec)],dtype=np.float64);

snr_rx0 = np.ones([len(timevec)],dtype=np.float64);
snr_rx1 = np.ones([len(timevec)],dtype=np.float64);
snr_rx2 = np.ones([len(timevec)],dtype=np.float64);

for i in range(earliest_pt,latest_pt+1): 
    rho_Tx = 1e3*y_sph_tx[0,i]; # [m]
    rho_Rx = 1e3*y_sph_rx[0,i]; # [m]
    
    delta_ra = new_ra[i] - new_ra[index_for_rx0];
    delta_dec = new_dec[i] - new_dec[index_for_rx0];
    gain_rx_radec_dB = RS.fnCalculate_Beamshape_Loss(beamwidth_ra_0,beamwidth_dec_0,delta_ra,delta_dec,gain_rx_dB);
    gain_rx_radec0 = RS.fn_dB_to_Power(gain_rx_radec_dB)        
    P_Rx0[i] = RS.fnCalculate_ReceivedPower(P_Tx,gain_tx,gain_rx_radec0,rho_Rx,rho_Tx,wavelength,RCS);

    rho_Rx = 1e3*y_sph_rx_meerkat_01[0,i];
    delta_ra = new_ra[i] - new_ra[index_for_rx1];
    delta_dec = new_dec[i] - new_dec[index_for_rx1];
    gain_rx_radec_dB = RS.fnCalculate_Beamshape_Loss(beamwidth_ra_1,beamwidth_dec_1,delta_ra,delta_dec,gain_rx_dB);
    gain_rx_radec1 = RS.fn_dB_to_Power(gain_rx_radec_dB)    
    P_Rx1[i] = RS.fnCalculate_ReceivedPower(P_Tx,gain_tx,gain_rx_radec1,rho_Rx,rho_Tx,wavelength,RCS); # need all distances in metres

    rho_Rx = 1e3*y_sph_rx_meerkat_02[0,i];
    delta_ra = new_ra[i] - new_ra[index_for_rx2];
    delta_dec = new_dec[i] - new_dec[index_for_rx2];
    gain_rx_radec_dB = RS.fnCalculate_Beamshape_Loss(beamwidth_ra_2,beamwidth_dec_2,delta_ra,delta_dec,gain_rx_dB);
    gain_rx_radec1 = RS.fn_dB_to_Power(gain_rx_radec_dB)    
    P_Rx2[i] = RS.fnCalculate_ReceivedPower(P_Tx,gain_tx,gain_rx_radec1,rho_Rx,rho_Tx,wavelength,RCS);

    snr_rx0[i] = RS.fnCalculate_ReceivedSNR(P_Rx0[i],meerkat_sys_temp,bandwidth);
    snr_rx1[i] = RS.fnCalculate_ReceivedSNR(P_Rx1[i],meerkat_sys_temp,bandwidth);
    snr_rx2[i] = RS.fnCalculate_ReceivedSNR(P_Rx2[i],meerkat_sys_temp,bandwidth);
    
P_Rx0_dB = RS.fn_Power_to_dB(P_Rx0)
P_Rx1_dB = RS.fn_Power_to_dB(P_Rx1)
P_Rx2_dB = RS.fn_Power_to_dB(P_Rx2)

snr_rx0_dB = RS.fn_Power_to_dB(snr_rx0);
snr_rx1_dB = RS.fn_Power_to_dB(snr_rx1);
snr_rx2_dB = RS.fn_Power_to_dB(snr_rx2);

np.save('main_057_iss_12_snr_rx0.npy',snr_rx0);
np.save('main_057_iss_12_snr_rx1.npy',snr_rx1);
np.save('main_057_iss_12_snr_rx2.npy',snr_rx2);
# --------------------------------------------------------------------------- #
# These indices come from pg 31 in my blue notebook.
timevec_index = np.arange(0,len(timevec),dtype=np.int64);
t_cpi = 0.1; # [s]
count = int(t_cpi/delta_t); # number of data points in a CPI

timevec_cpi_list_2 = timevec_index[earliest_pt:latest_pt_rx2:count];
timevec_cpi_2 = np.arange(timevec[timevec_cpi_list_2[0]],timevec[timevec_cpi_list_2[-1]+1],t_cpi,dtype=np.float64);
snr_coh_2 = np.ones([len(timevec_cpi_list_2)],dtype=np.float64);

for i in range(1,len(timevec_cpi_list_2)):
    snr_coh_2[i]= np.sum(snr_rx2[timevec_cpi_list_2[i-1]:timevec_cpi_list_2[i]])

timevec_cpi_list_1 = timevec_index[earliest_pt_rx1:latest_pt:count];
timevec_cpi_1 = np.arange(timevec[timevec_cpi_list_1[0]],timevec[timevec_cpi_list_1[-1]+1],t_cpi,dtype=np.float64);
snr_coh_1 = np.ones([len(timevec_cpi_list_1)],dtype=np.float64);

for i in range(1,len(timevec_cpi_list_1)):
    snr_coh_1[i]= np.sum(snr_rx1[timevec_cpi_list_1[i-1]:timevec_cpi_list_1[i]])

timevec_cpi_list_0 = timevec_index[latest_pt_rx2:earliest_pt_rx1:count];
timevec_cpi_0 = np.arange(timevec[timevec_cpi_list_0[0]],timevec[timevec_cpi_list_0[-1]+1],t_cpi,dtype=np.float64);
snr_coh_0 = np.ones([len(timevec_cpi_list_0)],dtype=np.float64);

for i in range(1,len(timevec_cpi_list_0)):
    snr_coh_0[i]= np.sum(snr_rx0[timevec_cpi_list_0[i-1]:timevec_cpi_list_0[i]])
    
snr_coh_2_dB = RS.fn_Power_to_dB(snr_coh_2);
snr_coh_1_dB = RS.fn_Power_to_dB(snr_coh_1);
snr_coh_0_dB = RS.fn_Power_to_dB(snr_coh_0);
# --------------------------------------------------------------------------- #

print 'writing data to file'
np.save('main_057_iss_12_timevec_cpi_list_2.npy',timevec_cpi_list_2);
np.save('main_057_iss_12_timevec_cpi_list_1.npy',timevec_cpi_list_1);
np.save('main_057_iss_12_timevec_cpi_list_0.npy',timevec_cpi_list_0);

np.save('main_057_iss_12_timevec_cpi_2.npy',timevec_cpi_2);
np.save('main_057_iss_12_timevec_cpi_1.npy',timevec_cpi_1);
np.save('main_057_iss_12_timevec_cpi_0.npy',timevec_cpi_0);

np.save('main_057_iss_12_snr_coh_2.npy',snr_coh_2);
np.save('main_057_iss_12_snr_coh_1.npy',snr_coh_1);
np.save('main_057_iss_12_snr_coh_0.npy',snr_coh_0);

max_snr_coh_dB = max(np.max(snr_coh_2_dB),np.max(snr_coh_1_dB),np.max(snr_coh_0_dB))
print max_snr_coh_dB
print RCS
# --------------------------------------------------------------------------- #

fig = plt.figure(1);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.axis('equal')
fig.suptitle(r"\textbf{SNR at MeerKAT for object %s during the interval %s}" %(norad_id,title_string) ,fontsize=12);

plt.plot(timevec[timevec_cpi_list_2[0]:timevec_cpi_list_2[-1]],snr_rx2_dB[timevec_cpi_list_2[0]:timevec_cpi_list_2[-1]],color='purple',linestyle='dashed',label=r"\textbf{ %s (single pulse)}" %meerkat_id[2])
plt.plot(timevec[timevec_cpi_list_1[0]:timevec_cpi_list_1[-1]],snr_rx1_dB[timevec_cpi_list_1[0]:timevec_cpi_list_1[-1]],color='orangered',linestyle='dashed',label=r"\textbf{ %s (single pulse)}" %meerkat_id[1])
plt.plot(timevec[timevec_cpi_list_0[0]:timevec_cpi_list_0[-1]],snr_rx0_dB[timevec_cpi_list_0[0]:timevec_cpi_list_0[-1]],color='navy',linestyle='dashed',label=r"\textbf{ %s (single pulse)}" %meerkat_id[0])


plt.plot(timevec_cpi_2[1:],snr_coh_2_dB[1:],color='purple',label=r"\textbf{ %s (coherent integration)}" %meerkat_id[2]);
plt.plot(timevec_cpi_1[1:],snr_coh_1_dB[1:],color='orangered',label=r"\textbf{ %s (coherent integration)}" %meerkat_id[1]);
plt.plot(timevec_cpi_0[1:],snr_coh_0_dB[1:],color='navy',label=r"\textbf{ %s (coherent integration)}" %meerkat_id[0]);
plt.xlim(timevec[earliest_pt],timevec[latest_pt]);
plt.legend(loc='best')
ax.set_ylabel(r'$\text{SNR}~[\mathrm{dB}]$')
ax.set_xlabel(r'Time $ t~[\mathrm{s}]$');
at = AnchoredText(r"$\text{PRF} = %d ~\mathrm{kHz}$ \& $\text{CPI} = %f~\mathrm{s}$" %(pulse_repetition_frequency/1000,t_cpi),prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
ax.add_artist(at)
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_12_snr_singlepulse_coh.pdf',bbox_inches='tight',pad_inches=0.11,dpi=10)
# --------------------------------------------------------------------------- #
