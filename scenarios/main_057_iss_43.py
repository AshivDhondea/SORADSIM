# -*- coding: utf-8 -*-
"""
Created on 02 November 2017

@author: Ashiv Dhondea
"""

import AstroConstants as AstCnst
import AstroFunctions as AstFn
import BistaticAndDoppler as BD
import DynamicsFunctions as DynFn
#import GNF as GNF
import GeometryFunctions as GF
import MathsFunctions as MathsFn
import Num_Integ as Integ
import TimeHandlingFunctions as THF
#import UnbiasedConvertedMeasurements as UCM
import StatsFunctions as StatsFn

import math
import numpy as np

import datetime as dt
import pytz
import aniso8601

import pandas as pd # for loading MeerKAT dishes' latlon

# Importing what's needed for nice plots.
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params)
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText


# ------------------------------------------------------------------------- #
print 'Loading MeerKAT positions'
dframe = pd.read_excel("MeerKAT64v36.wgs84.64x4_edited.xlsx",sheetname="Sheet1")
dframe = dframe.reset_index()

meerkat_id = dframe['ID'][0:64]
meerkat_lat = dframe['Lat'][0:64]
meerkat_lon = dframe['Lon'][0:64]
# --------------------------------------------------------------------------- #
with open('main_meerkat_radar_parameters_doreen.txt') as fp:
    for line in fp:
        if 'PRF' in line:
            good_index = line.index('=')
            pulse_repetition_frequency = float(line[good_index+1:-1]); 
            pulse_repetition_frequency = int(pulse_repetition_frequency)
        if 'centre_frequency' in line:
            good_index = line.index('=')
            centre_frequency = float(line[good_index+1:-1]); 
fp.close();

# Doppler related stuff
speed_light = AstCnst.c;#[km/s]
wavelength = speed_light/centre_frequency; # [km]
pulse_repetition_time = 1/pulse_repetition_frequency; # [s]
# --------------------------------------------------------------------------- #
print 'Loading data'
timevec = np.load('main_057_iss_05_timevec.npy'); # timevector
x_state_sgp4 = np.load('main_057_iss_05_x_state_sgp4.npy'); # state vector in ECI frame

dx = np.shape(x_state_sgp4)[0];
# --------------------------------------------------------------------------- #
# Location of MeerKAT
lat_meerkat_00 = float(meerkat_lat[0]);
lon_meerkat_00 =  float(meerkat_lon[0]);
altitude_meerkat = 1.038; # [km]

# Location of Denel Bredasdorp
lat_denel = -34.6; # [deg]
lon_denel = 20.316666666666666; # [deg]
altitude_denel = 0.018;#[km]

# Location of observer [Cape Town, ZA]
lat_station = -33.9249; # [deg]
lon_station =  18.4241; # [deg]
altitude_station = 0; # [m]

Tx_ecef = AstFn.fnRadarSite(math.radians(lat_denel),math.radians(lon_denel),altitude_denel);
Rx0_ecef = AstFn.fnRadarSite(math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),altitude_meerkat);

Tx_pos = AstFn.fnRAZEL_Cartesian(math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),altitude_meerkat,Tx_ecef,0.);
Rx0_pos = np.zeros([3],dtype=np.float64);
# --------------------------------------------------------------------------- #
print 'Loading data'
timevec_old = np.load('main_057_iss_05_timevec.npy'); # timevector

# time stamps
experiment_timestamps = [None]*len(timevec_old)
index=0;
with open('main_057_iss_05_experiment_timestamps.txt') as fp:
    for line in fp:
        modified_timestring = line[:-8];
        experiment_timestamps[index] = aniso8601.parse_datetime(modified_timestring);
        index+=1;
fp.close();
norad_id = '25544'
# --------------------------------------------------------------------------- #
tx_beam_indices_best = np.load('main_057_iss_07_tx_beam_indices_best.npy');
# --------------------------------------------------------------------------- #
# sort out a few variables
tx_bw_time_max = tx_beam_indices_best[1];
tx_beam_index_down = tx_beam_indices_best[0];
tx_beam_index_up = tx_beam_indices_best[2];
# --------------------------------------------------------------------------- #
print 'finding relevant epochs'
## Find the epoch of the relevant data points
#tx_beam_index_down_epoch = THF.fnCalculate_DatetimeEpoch(timevec_old,tx_beam_index_down,experiment_timestamps[0]);
#tx_beam_index_up_epoch= THF.fnCalculate_DatetimeEpoch(timevec_old,tx_beam_index_up,experiment_timestamps[0]);
#tx_bw_time_max_epoch = THF.fnCalculate_DatetimeEpoch(timevec_old,tx_bw_time_max,experiment_timestamps[0]);
#
#tx_beam_index_down_epoch = tx_beam_index_down_epoch.replace(tzinfo=None);
#tx_beam_index_up_epoch = tx_beam_index_up_epoch.replace(tzinfo=None)
#tx_bw_time_max_epoch = tx_bw_time_max_epoch.replace(tzinfo=None)

tx_beam_circ_index = np.load('main_057_iss_08_tx_beam_circ_index.npy');
earliest_pt = tx_beam_circ_index[0];
tx_bw_time_max = tx_beam_circ_index[1];
latest_pt = tx_beam_circ_index[2];
# --------------------------------------------------------------------------- #
earliest_pt_epoch= THF.fnCalculate_DatetimeEpoch(timevec,earliest_pt,experiment_timestamps[0]);
latest_pt_epoch = THF.fnCalculate_DatetimeEpoch(timevec,latest_pt,experiment_timestamps[0]);

earliest_pt_epoch = earliest_pt_epoch.replace(tzinfo=None)
latest_pt_epoch= latest_pt_epoch.replace(tzinfo=None)

title_string = str(earliest_pt_epoch.isoformat())+'Z/'+str(latest_pt_epoch.isoformat())+'Z';

# --------------------------------------------------------------------------- #
print 'creating required arrays'
t_cpi = 0.1; # [s] length of one coherent processing interval

# Find the time elapsed between the start and end of the observation window
simulation_duration_dt_obj = latest_pt_epoch - earliest_pt_epoch;
simulation_duration_secs = simulation_duration_dt_obj.total_seconds();

print simulation_duration_secs

delta_t = t_cpi; 
# Declare time and state vector variables.
print 'Propagation time step = %f' %delta_t, '[s]'
duration = simulation_duration_secs-delta_t; #[s]
print 'Duration of simulation = %f' %duration, '[s]'
timevec = np.arange(0,duration,delta_t,dtype=np.float64);
print timevec[-1]
print len(timevec)
x_state = np.zeros([6,len(timevec)],dtype=np.float64);
y_bd  = np.zeros([2,len(timevec)],dtype=np.float64);

theta_GMST = np.zeros([len(timevec)],dtype=np.float64); 
# timestamps for this experiment
experiment_timestamps = [None]*len(timevec);

initialtraj = x_state_sgp4[:,tx_beam_index_down];
x_state[:,0] = initialtraj;
# --------------------------------------------------------------------------- #
for index in range(1,len(timevec)):
    # Calculate ECI state vector
    x_state[:,index] = Integ.fnRK4_vector(DynFn.fnKepler_J2, delta_t, x_state[:,index-1],timevec[index]);

# --------------------------------------------------------------------------- #
for index in range(len(timevec)):
    # Find the current timestamp as a datetime object
    tle_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec, index,earliest_pt_epoch)            
    current_time_iso = tle_epoch_test.isoformat() + 'Z';
    experiment_timestamps[index] =current_time_iso;
       
    # From the epoch, find the GMST angle.               
    # Rotate ECI position vector by GMST angle to get ECEF position
    theta_GMST[index] = GF.fnZeroTo2Pi(math.radians(THF.fn_Convert_Datetime_to_GMST(tle_epoch_test)));
    
    y_bd[:,index] =BD.fnCalculate_Bistatic_RangeAndDoppler_OD_fast(x_state[:,index],Rx0_ecef,Tx_ecef,theta_GMST[index],wavelength);
    

kindex = np.arange(0,len(timevec),1,dtype=np.int64)
# ------------------------------------------------------------------------- #
f, axarr = plt.subplots(2,sharex=True);
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
plt.rc('font',family='helvetica');
f.suptitle(r"\textbf{Bistatic range \& Doppler shift for object %s trajectory for %s}" %(norad_id,title_string),fontsize=12)

axarr[0].plot(kindex ,y_bd[0,:])
axarr[0].set_ylabel(r'$\rho_{\text{b}}~[\mathrm{km}]$');
axarr[0].set_title(r'Bistatic range')

axarr[1].plot(kindex ,y_bd[1,:])
axarr[1].set_title(r'Bistatic Doppler shift')
axarr[1].set_ylabel(r'$f_{\text{b,d}}~[\mathrm{Hz}]$');

axarr[1].set_xlabel(r'$k$');

axarr[0].grid(True)
axarr[1].grid(True)
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0:1]], visible=False)
f.savefig('main_057_iss_43_biradopp.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10)  