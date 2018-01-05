# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 23:09:51 2017
Range measurement error: eqn 8.6 on page 168 in Curry: Radar performance modelling
Doppler measurement error: eqn 8.13 on page 172

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
tx_bw_time_max = tx_beam_indices_best[1];
tx_beam_index_down = tx_beam_indices_best[0];
tx_beam_index_up = tx_beam_indices_best[2];
# --------------------------------------------------------------------------- #
print 'Load SNR data'
timevec_cpi_list_2 = np.load('main_057_iss_12_timevec_cpi_list_2.npy');
timevec_cpi_list_1 = np.load('main_057_iss_12_timevec_cpi_list_1.npy');
timevec_cpi_list_0 = np.load('main_057_iss_12_timevec_cpi_list_0.npy');

timevec_cpi_2 = np.load('main_057_iss_12_timevec_cpi_2.npy');
timevec_cpi_1 = np.load('main_057_iss_12_timevec_cpi_1.npy');
timevec_cpi_0 = np.load('main_057_iss_12_timevec_cpi_0.npy');

snr_coh_2 = np.load('main_057_iss_12_snr_coh_2.npy');
snr_coh_1 = np.load('main_057_iss_12_snr_coh_1.npy');
snr_coh_0 = np.load('main_057_iss_12_snr_coh_0.npy');

# --------------------------------------------------------------------------- #
print 'Load resolution data'
bistatic_angle = np.load('main_057_iss_19_bistatic_angle.npy'); # [rad]
range_resolution = np.load('main_057_iss_19_range_resolution.npy'); # [km]
doppler_resolution = np.load('main_057_iss_19_doppler_resolution.npy'); # [km/s]
kd =(-1./wavelength);
# --------------------------------------------------------------------------- #
sd_range_error_2 = np.zeros_like(snr_coh_2);
sd_range_error_1 = np.zeros_like(snr_coh_1);
sd_range_error_0 = np.zeros_like(snr_coh_0);

sd_doppler_error_2 = np.zeros_like(snr_coh_2);
sd_doppler_error_1 = np.zeros_like(snr_coh_1);
sd_doppler_error_0 = np.zeros_like(snr_coh_0);

for i in range(1,len(timevec_cpi_list_2)):
    sd_range_error_2[i] = range_resolution[timevec_cpi_list_2[i]]/(math.sqrt(2*snr_coh_2[i]));
    sd_doppler_error_2[i] = -kd*doppler_resolution[timevec_cpi_list_2[i]]/(math.sqrt(2*snr_coh_2[i]));

for i in range(1,len(timevec_cpi_list_0)):
    sd_range_error_0[i] = range_resolution[timevec_cpi_list_0[i]]/(math.sqrt(2*snr_coh_0[i]));
    sd_doppler_error_0[i] = -kd*doppler_resolution[timevec_cpi_list_0[i]]/(math.sqrt(2*snr_coh_0[i]));
    
for i in range(1,len(timevec_cpi_list_1)):
    sd_range_error_1[i] = range_resolution[timevec_cpi_list_1[i]]/(math.sqrt(2*snr_coh_1[i]));
    sd_doppler_error_1[i] = -kd*doppler_resolution[timevec_cpi_list_1[i]]/(math.sqrt(2*snr_coh_1[i]));
    
sd_doppler_error_max = max(sd_doppler_error_2[1:].max(),sd_doppler_error_1[1:].max(),sd_doppler_error_0[1:].max());
sd_range_error_max = max(sd_range_error_2[1:].max(),sd_range_error_1[1:].max(),sd_range_error_0[1:].max());
cov_rangedopp = np.diag([sd_range_error_max**2,sd_doppler_error_max**2])     
np.save('main_057_iss_42_cov_rangedopp.npy',cov_rangedopp);
# --------------------------------------------------------------------------- #
print 'finding relevant epochs'
# Find the epoch of the relevant data points
plt_start_index = tx_beam_index_down
plt_end_index = tx_beam_index_up

start_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec,plt_start_index,experiment_timestamps[0]);
end_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec,plt_end_index,experiment_timestamps[0]);
tx_beam_index_down_epoch = THF.fnCalculate_DatetimeEpoch(timevec,tx_beam_index_down,experiment_timestamps[0]);
tx_beam_index_up_epoch= THF.fnCalculate_DatetimeEpoch(timevec,tx_beam_index_up,experiment_timestamps[0]);
tx_bw_time_max_epoch = THF.fnCalculate_DatetimeEpoch(timevec,tx_bw_time_max,experiment_timestamps[0]);

end_epoch_test = end_epoch_test.replace(tzinfo=None);
start_epoch_test = start_epoch_test.replace(tzinfo=None)
title_string = str(start_epoch_test.isoformat())+'Z/'+str(end_epoch_test .isoformat())+'Z';
tx_beam_index_down_epoch = tx_beam_index_down_epoch.replace(tzinfo=None);
tx_beam_index_up_epoch = tx_beam_index_up_epoch.replace(tzinfo=None)
tx_bw_time_max_epoch = tx_bw_time_max_epoch.replace(tzinfo=None)

# --------------------------------------------------------------------------- #

fig = plt.figure(1);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.axis('equal')
fig.suptitle(r"\textbf{Bistatic range measurement error at Rx for object %s during the interval %s}" %(norad_id,title_string) ,fontsize=12);

plt.plot(timevec_cpi_2[1:],sd_range_error_2[1:],color='purple',label=r"\textbf{%s}" %meerkat_id[2])
plt.plot(timevec_cpi_1[1:],sd_range_error_1[1:],color='orangered',label=r"\textbf{%s}" %meerkat_id[1])
plt.plot(timevec_cpi_0[1:],sd_range_error_0[1:],color='navy',label=r"\textbf{%s}" %meerkat_id[0])

#plt.xlim(timevec[earliest_pt],timevec[latest_pt]);
plt.legend(loc='best')
ax.set_ylabel(r'$\sigma_{\rho_{\text{b}}}~[\mathrm{km}]$')
ax.set_xlabel(r'Time $ t~[\mathrm{s}]$');
at = AnchoredText(r"$\text{PRF} = %d ~\mathrm{kHz}$ \& $T_\text{CPI} = %f~\mathrm{s}$" %(pulse_repetition_frequency/1000,t_cpi),prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
ax.add_artist(at)
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_42_sd_range.pdf',bbox_inches='tight',pad_inches=0.11,dpi=10)    
#fig.savefig('main_057_iss_42_sd_range.png',bbox_inches='tight',pad_inches=0.11)    
    

fig = plt.figure(2);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.axis('equal')
fig.suptitle(r"\textbf{Bistatic Doppler measurement error at Rx for object %s during the interval %s}" %(norad_id,title_string) ,fontsize=12);

plt.plot(timevec_cpi_2[1:],sd_doppler_error_2[1:],color='purple',label=r"\textbf{%s}" %meerkat_id[2])
plt.plot(timevec_cpi_1[1:],sd_doppler_error_1[1:],color='orangered',label=r"\textbf{%s}" %meerkat_id[1])
plt.plot(timevec_cpi_0[1:],sd_doppler_error_0[1:],color='navy',label=r"\textbf{%s}" %meerkat_id[0])

#plt.xlim(timevec[earliest_pt],timevec[latest_pt]);
plt.legend(loc='best')
ax.set_ylabel(r'$\sigma_{f_{\text{b,d}}}~[\mathrm{Hz}]$')
ax.set_xlabel(r'Time $ t~[\mathrm{s}]$');
at = AnchoredText(r"$\text{PRF} = %d ~\mathrm{kHz}$ \& $T_\text{CPI} = %f~\mathrm{s}$" %(pulse_repetition_frequency/1000,t_cpi),prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
ax.add_artist(at)
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_42_sd_doppler.pdf',bbox_inches='tight',pad_inches=0.11,dpi=10) 
#fig.savefig('main_057_iss_42_sd_doppler.png',bbox_inches='tight',pad_inches=0.11)
# ------------------------------------------------------------------------- #
print 'Writing to file'
fname = 'main_057_iss_42_stddev.txt'
f = open(fname,'w');

f.write('worst range measurement error = '+str(sd_range_error_max)+' km \n');
f.write('worst Doppler measurement error = '+str(sd_doppler_error_max)+' Hz \n');

f.close();
print 'cool cool cool'