# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 18:58:15 2017

@author: Ashiv Dhondea
"""
import AstroConstants as AstCnst
import AstroFunctions as AstFn
import BistaticAndDoppler as BD
import TimeHandlingFunctions as THF

import numpy as np
import math

import datetime as dt
import pytz
import aniso8601

# Importing what's needed for nice plots.
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params)
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

import pandas as pd # for loading MeerKAT dishes' latlon
# --------------------------------------------------------------------------- #
print 'Loading MeerKAT positions'
dframe = pd.read_excel("MeerKAT64v36.wgs84.64x4_edited.xlsx",sheetname="Sheet1")
dframe = dframe.reset_index()
meerkat_id = dframe['ID'][0:64]
meerkat_lat = dframe['Lat'][0:64]
meerkat_lon = dframe['Lon'][0:64]
# --------------------------------------------------------------------------- #
## Bistatic Radar characteristics
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

# Convert lat,lon,altitude to ecef, then to sez. create observations in sez. 
# also convert track to sez before creating observations
# reminder that the sez frame has its origin at the receiver.

Tx_ecef = AstFn.fnRadarSite(math.radians(lat_denel),math.radians(lon_denel),altitude_denel);
Rx0_ecef = AstFn.fnRadarSite(math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),altitude_meerkat);

Tx_pos = AstFn.fnRAZEL_Cartesian(math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),altitude_meerkat,Tx_ecef,0.);
Rx0_pos = np.zeros([3],dtype=np.float64);
# --------------------------------------------------------------------------- #
# Doppler related stuff
speed_light = AstCnst.c; # [km/s]

# the nominal radar prf
with open('main_meerkat_radar_parameters_doreen.txt') as fp:
    for line in fp:
        if 'PRF' in line:
            good_index = line.index('=')
            pulse_repetition_frequency = float(line[good_index+1:-1]); 
        if 'centre_frequency' in line:
            good_index = line.index('=')
            centre_frequency = float(line[good_index+1:-1]); 
        if 'bandwidth' in line:
            good_index = line.index('=')
            bandwidth= float(line[good_index+1:-1]);  

fp.close();

wavelength = speed_light/centre_frequency; # [km]
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

x_target = np.load('main_057_iss_05_x_target.npy');
norad_id = '25544'

t_cpi = 0.1; # [s]
cpi = int(t_cpi/delta_t); # number of data points in a CPI

tx_beam_indices_best = np.load('main_057_iss_07_tx_beam_indices_best.npy');
# --------------------------------------------------------------------------- #
# sort out a few variables
#tx_bw_time_max = tx_beam_indices_best[1];
#tx_beam_index_down = tx_beam_indices_best[0];
#tx_beam_index_up = tx_beam_indices_best[2];


tx_beam_circ_index = np.load('main_057_iss_08_tx_beam_circ_index.npy');
earliest_pt = tx_beam_circ_index[0];
tx_bw_time_max = tx_beam_circ_index[1];
latest_pt = tx_beam_circ_index[2];
# --------------------------------------------------------------------------- #
## Calculate bistatic angle to target as well as bistatic range resolution
bistatic_angle = np.zeros([len(timevec)],dtype=np.float64);
range_resolution = np.zeros([len(timevec)],dtype=np.float64);
doppler_resolution = np.zeros([len(timevec)],dtype=np.float64);

for index in range(earliest_pt,latest_pt+1):
#    print 'check'
    bistatic_angle[index] = BD.fnCalculate_Bistatic_Angle(x_target[0:3,index],Rx0_pos,Tx_pos);
    range_resolution[index] = BD.fnCalculate_Bistatic_Range_Resolution(speed_light,bistatic_angle[index],bandwidth);
    doppler_resolution[index] = BD.fnCalculate_Bistatic_Doppler_Resolution(wavelength,t_cpi,bistatic_angle[index])

print 'Saving data'
np.save('main_057_iss_19_bistatic_angle.npy',bistatic_angle); # [rad]
np.save('main_057_iss_19_range_resolution.npy',range_resolution); # [km]
np.save('main_057_iss_19_doppler_resolution.npy',doppler_resolution); # [km/s]
kd =(-1./wavelength);
# --------------------------------------------------------------------------- #
earliest_pt_epoch= THF.fnCalculate_DatetimeEpoch(timevec,earliest_pt,experiment_timestamps[0]);
latest_pt_epoch = THF.fnCalculate_DatetimeEpoch(timevec,latest_pt,experiment_timestamps[0]);

earliest_pt_epoch = earliest_pt_epoch.replace(tzinfo=None)
latest_pt_epoch= latest_pt_epoch.replace(tzinfo=None)

title_string = str(earliest_pt_epoch.isoformat())+'Z/'+str(latest_pt_epoch.isoformat())+'Z';

# --------------------------------------------------------------------------- #
## Plot bistatic angle as function of time
f, axarr = plt.subplots(3,sharex=True);
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
f.suptitle(r"\textbf{Bistatic angle and range \& Doppler resolution for object %s trajectory for %s}" %(norad_id,title_string),fontsize=12,y=1.01)
axarr[0].plot(timevec[earliest_pt:latest_pt+1],bistatic_angle[earliest_pt:latest_pt+1]);
axarr[0].set_title(r'Bistatic angle $\beta~[\mathrm{rad}]$')
axarr[0].set_ylabel(r'$\beta$');

#axarr[0].axvspan(timevec[plot_start_index],timevec[plot_end_index],facecolor='gray',alpha=0.3);
axarr[1].set_title(r'Range resolution $\Delta \rho_{\text{b}}~[\mathrm{km}]$')
axarr[1].plot(timevec[earliest_pt:latest_pt+1],range_resolution[earliest_pt:latest_pt+1]);
axarr[1].set_ylabel(r'$\Delta \rho_{\text{b}}$');

#axarr[1].axvspan(timevec[plot_start_index],timevec[plot_end_index],facecolor='gray',alpha=0.3);
axarr[2].set_title(r'Doppler resolution $\Delta f_{\text{b, d}} ~[\mathrm{Hz}]$')
axarr[2].plot(timevec[earliest_pt:latest_pt+1],-kd*doppler_resolution[earliest_pt:latest_pt+1]);
axarr[2].set_ylabel(r'$\Delta f_{\text{b, d}}$');

#axarr[2].axvspan(timevec[plot_start_index],timevec[plot_end_index],facecolor='gray',alpha=0.3);
axarr[2].set_xlabel(r'Time $t~[\mathrm{s}]$');
axarr[0].grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
axarr[1].grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
axarr[2].grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$ \& $T_{\text{CPI}} = %f ~\mathrm{s}$" %(delta_t,t_cpi),prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
axarr[2].add_artist(at)
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0:2]], visible=False)
plt.subplots_adjust(hspace=0.4)
f.savefig('main_057_iss_19_bistaticangle.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10) 

# ------------------------------------------------------------------------- #
print 'Writing to file'
fname = 'main_057_iss_19_resolution.txt'
f = open(fname,'w');
max_range_resolution = range_resolution.max();
doppler_resolution_abs = np.absolute(kd*doppler_resolution);
max_doppler_resolution = doppler_resolution_abs.max();
f.write('worst range resolution = '+str(max_range_resolution)+' km \n');
f.write('worst Doppler resolution = '+str(max_doppler_resolution)+' Hz \n');

f.close();
print 'cool cool cool'