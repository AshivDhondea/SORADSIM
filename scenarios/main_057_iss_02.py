# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:18:04 2017

@author: Ashiv Dhondea
"""

import AstroFunctions as AstFn
import GeometryFunctions as GF
import TimeHandlingFunctions as THF
import UnbiasedConvertedMeasurements as UCM

import math
import numpy as np

# Libraries needed for time keeping and formatting
import datetime as dt
import pytz
import aniso8601

# Module for SGP4 orbit propagation
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

import pandas as pd # for loading MeerKAT dishes' latlon

# Importing what's needed for nice plots.
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params)
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
# --------------------------------------------------------------------------- #
print 'Loading MeerKAT positions'
dframe = pd.read_excel("MeerKAT64v36.wgs84.64x4_edited.xlsx",sheetname="Sheet1")
dframe = dframe.reset_index()
meerkat_id = dframe['ID'][0:64]
meerkat_lat = dframe['Lat'][0:64]
meerkat_lon = dframe['Lon'][0:64]
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
# Create datetime objects for the start and end of the visibility interval
start_vis_epoch_low_res = aniso8601.parse_datetime(start_timestring[:-1]);
start_vis_epoch_low_res = start_vis_epoch_low_res.replace(tzinfo=None);

end_vis_epoch_low_res = aniso8601.parse_datetime(end_timestring[:-1]);
end_vis_epoch_low_res= end_vis_epoch_low_res.replace(tzinfo=None);
"""
Note that these were obtained at a coarse resolution with a timestep of 1[s].
This script will run at 1e-3[s], so we need to be cautious about using these as
bounds.
"""
delta_t = 1e-3; #[s]
delta_t_original = 1; # [s]
epsilon_time_res = 0.1*delta_t_original/delta_t;#[s], assume 10% error in time
start_vis_epoch = start_vis_epoch_low_res - dt.timedelta(seconds=epsilon_time_res);
end_vis_epoch = end_vis_epoch_low_res + dt.timedelta(seconds=epsilon_time_res);
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
Rx1_ecef = AstFn.fnRadarSite(math.radians(lat_meerkat_01),math.radians(lon_meerkat_01),altitude_meerkat);
Rx2_ecef = AstFn.fnRadarSite(math.radians(lat_meerkat_02),math.radians(lon_meerkat_02),altitude_meerkat);

Tx_pos = AstFn.fnRAZEL_Cartesian(math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),altitude_meerkat,Tx_ecef,0.);
Rx1_pos = AstFn.fnRAZEL_Cartesian(math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),altitude_meerkat,Rx1_ecef,0.);
Rx2_pos = AstFn.fnRAZEL_Cartesian(math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),altitude_meerkat,Rx2_ecef,0.);
Rx0_pos = np.zeros([3],dtype=np.float64);
# --------------------------------------------------------------------------- #
# ISS (ZARYA)                        
tle_line1 = '1 25544U 98067A   17253.93837963  .00001150  00000-0  24585-4 0  9991';
tle_line2 = '2 25544  51.6444 330.8522 0003796 258.3764  78.6882 15.54163465 75088';

so_name = 'ISS (ZARYA)' 
# Read TLE to extract Keplerians and epoch. 
a,e,i,BigOmega,omega,E,nu,epoch  = AstFn.fnTLEtoKeps(tle_line1,tle_line2);

# Create satellite object
satellite_obj = twoline2rv(tle_line1, tle_line2, wgs84);

line1 = (tle_line1);
line2 = (tle_line2);

# Figure out the TLE epoch 
year,dayy, hrs, mins, secs, millisecs = THF.fn_Calculate_Epoch_Time(epoch);
todays_date =THF.fn_epoch_date(year,dayy);
print "TLE epoch date is", todays_date
print "UTC time = ",hrs,"h",mins,"min",secs+millisecs,"s"
timestamp_tle_epoch = dt.datetime(year=todays_date.year,month=todays_date.month,day=todays_date.day,hour=hrs,minute=mins,second=secs,microsecond=int(millisecs),tzinfo= None);
# --------------------------------------------------------------------------- #
# Find the time elapsed between the start and end of the observation window
simulation_duration_dt_obj = end_vis_epoch- start_vis_epoch;
simulation_duration_secs = simulation_duration_dt_obj.total_seconds();

# Declare time and state vector variables.
print 'Propagation time step = %f' %delta_t, '[s]'
duration = simulation_duration_secs; #[s]
print 'Duration of simulation = %f' %duration, '[s]'
timevec = np.arange(0,duration+delta_t,delta_t,dtype=np.float64);
x_state_sgp4 = np.zeros([6,len(timevec)],dtype=np.float64);
R_SEZ = np.zeros([3,len(timevec)],dtype=np.float64);
V_SEZ = np.zeros([3,len(timevec)],dtype=np.float64);
x_target = np.zeros([6,len(timevec)],dtype=np.float64); 
#  spherical measurements from the Tx and Rx
y_sph_tx = np.zeros([3,len(timevec)],dtype=np.float64);
y_sph_rx = np.zeros([3,len(timevec)],dtype=np.float64);
y_sph_rx_meerkat_01 = np.zeros([3,len(timevec)],dtype=np.float64);
y_sph_rx_meerkat_02 = np.zeros([3,len(timevec)],dtype=np.float64);
theta_GMST = np.zeros([len(timevec)],dtype=np.float64); 
xecef_sgp4 = np.zeros([3,len(timevec)],dtype=np.float64);
# Declare variables to store latitude and longitude values of the ground track
lat_sgp4 = np.zeros([len(timevec)],dtype=np.float64);
lon_sgp4 = np.zeros([len(timevec)],dtype=np.float64);
# timestamps for this experiment
experiment_timestamps = [None]*len(timevec);

for index in range(len(timevec)):
    # Find the current timestamp as a datetime object
    tle_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec, index, start_vis_epoch)            
    current_time_iso = tle_epoch_test.isoformat() + 'Z';
    experiment_timestamps[index] =current_time_iso; 
    # SGP4 propagation
    satpos,satvel = satellite_obj.propagate(tle_epoch_test.year,tle_epoch_test.month,tle_epoch_test.day,tle_epoch_test.hour,tle_epoch_test.minute,tle_epoch_test.second+(1e-6)*tle_epoch_test.microsecond);
    x_state_sgp4[0:3,index] = np.asarray(satpos);
    x_state_sgp4[3:6,index] = np.asarray(satvel);
    
    # From the epoch, find the GMST angle.               
    # Rotate ECI position vector by GMST angle to get ECEF position
    theta_GMST[index] = GF.fnZeroTo2Pi(math.radians(THF.fn_Convert_Datetime_to_GMST(tle_epoch_test)));
    xecef_sgp4[:,index] = AstFn.fnECItoECEF(x_state_sgp4[0:3,index],theta_GMST[index]);
    lat_sgp4[index],lon_sgp4[index] = AstFn.fnCarts_to_LatLon(xecef_sgp4[:,index]);

    # We find the position and velocity vector for the target in the local frame.
    # We then create the measurement vector consisting of range and look angles to the target.
    R_ECI = x_state_sgp4[0:3,index]; V_ECI = x_state_sgp4[3:6,index];  
    R_SEZ[:,index] = AstFn.fnRAZEL_Cartesian(math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),altitude_meerkat,R_ECI,theta_GMST[index]); 
    R_ECEF = AstFn.fnECItoECEF(R_ECI,theta_GMST[index]);
    V_SEZ[:,index] = AstFn.fnVel_ECI_to_SEZ(V_ECI,R_ECEF,math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),theta_GMST[index]);
    x_target[:,index] = np.hstack((R_SEZ[:,index],V_SEZ[:,index])); # state vector in SEZ frame
    # Calculate range and angles for system modelling.
    y_sph_rx[:,index] = UCM.fnCalculate_Spherical(R_SEZ[:,index]); # slant-range and look angles wrt to Rx
    pos_in_tx_frame = R_SEZ[:,index] - Tx_pos; # correction for Tx centred coordinate system
    y_sph_tx[:,index] = UCM.fnCalculate_Spherical(pos_in_tx_frame); # slant-range and look angles wrt to Tx
    pos_in_rx1_frame = R_SEZ[:,index] - Rx1_pos; 
    y_sph_rx_meerkat_01[:,index] = UCM.fnCalculate_Spherical(pos_in_rx1_frame); 
    pos_in_rx2_frame = R_SEZ[:,index] - Rx2_pos; 
    y_sph_rx_meerkat_02[:,index] = UCM.fnCalculate_Spherical(pos_in_rx2_frame); 

# --------------------------------------------------------------------------- #
# Find indices for all data points whose elevation angle exceeds 10 deg
tx_el_min = math.radians(10.)

try:
	tx_el_min_range = np.where( y_sph_tx[1,:] >= tx_el_min)
	tx_el_min_index = tx_el_min_range[0][0];
except IndexError:
    print 'cannot place tx el min'
    tx_el_min_index = 0;
	
try:
	tx_el_max_range = np.where( y_sph_tx[1,tx_el_min_index:] >= tx_el_min)
	tx_el_max_index = tx_el_max_range[0][-1]+tx_el_min_index;
except IndexError:
    print 'cannot place tx el max'
    tx_el_max_index = len(timevec)-1;
# --------------------------------------------------------------------------- #
 ### Bounds
time_index = np.zeros([2],dtype=np.int64);
time_index[0] = tx_el_min_index
time_index[1] = tx_el_max_index
print 'bounds for Tx FoR'
print time_index[0]
print time_index[1]

np.save('main_057_iss_02_time_index.npy',time_index);
# --------------------------------------------------------------------------- #
ground_station='Tx';
experiment_timestamps_start = THF.fnRead_Experiment_Timestamps(experiment_timestamps,0);
start_vis_epoch = THF.fnCalculate_DatetimeEpoch(timevec,time_index[0],experiment_timestamps_start);
end_vis_epoch = THF.fnCalculate_DatetimeEpoch(timevec,time_index[1],experiment_timestamps_start);

start_vis_epoch = start_vis_epoch.replace(tzinfo=None);
end_vis_epoch = end_vis_epoch.replace(tzinfo=None);

experiment_timestamps_start = THF.fnRead_Experiment_Timestamps(experiment_timestamps,0);
start_plot_epoch = THF.fnCalculate_DatetimeEpoch(timevec,0,experiment_timestamps_start);
end_plot_epoch = THF.fnCalculate_DatetimeEpoch(timevec,len(timevec)-1,experiment_timestamps_start);

start_plot_epoch = start_plot_epoch.replace(tzinfo=None);
end_plot_epoch = end_plot_epoch.replace(tzinfo=None);
title_string = str(start_plot_epoch.isoformat())+'Z/'+str(end_plot_epoch.isoformat())+'Z';
# --------------------------------------------------------------------------- #
"""
fig = plt.figure(1);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig.suptitle(r"\textbf{Elevation angle to %s from %s over %s}" %(so_name,ground_station,title_string),fontsize=12);
plt.plot(timevec,np.rad2deg(y_sph_tx[1,:]))
plt.axvspan(timevec[time_index[0]],timevec[time_index[1]],facecolor='green',alpha=0.2);
plt.scatter(timevec[time_index[0]],math.degrees(y_sph_tx[1,time_index[0]]),s=50,marker=r"$\Box$",facecolors='none', edgecolors='crimson',label=r"%s"  %str(start_vis_epoch.isoformat()+'Z'));
plt.scatter(timevec[time_index[1]],math.degrees(y_sph_tx[1,time_index[1]]),s=50,marker=r"$\circledcirc$",facecolors='none', edgecolors='purple',label=r"%s" %str(end_vis_epoch.isoformat()+'Z'));
plt.legend(loc='center left',title=r"\textbf{Timestamps}",bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True)
ax.set_ylabel(r"Elevation angle $\theta~[\mathrm{^\circ}]$")
ax.set_xlabel(r'Time $t~[\mathrm{s}]$');
at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
ax.add_artist(at)
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_02_el.pdf',bbox_inches='tight',pad_inches=0.11,dpi=10)
# --------------------------------------------------------------------------- #    
print 'Saving results'
np.save('main_057_iss_02_timevec.npy',timevec); # timevector
np.save('main_057_iss_02_x_target.npy',x_target); # state vector in SEZ frame
np.save('main_057_iss_02_lat_sgp4.npy',lat_sgp4); 
np.save('main_057_iss_02_lon_sgp4.npy',lon_sgp4); 
np.save('main_057_iss_02_theta_GMST.npy',theta_GMST); # GMST angles in rad
np.save('main_057_iss_02_y_sph_rx.npy',y_sph_rx); # spherical measurement vectors in Rx frame
np.save('main_057_iss_02_y_sph_tx.npy',y_sph_tx); # spherical measurement vectors in Tx frame
np.save('main_057_iss_02_y_sph_rx_meerkat_01.npy',y_sph_rx_meerkat_01); # spherical measurement vectors in Rx frame
np.save('main_057_iss_02_y_sph_rx_meerkat_02.npy',y_sph_rx_meerkat_02);

fname = 'main_057_iss_02_experiment_timestamps.txt';
f = open(fname, 'w') # Create data file;
for index in range (len(experiment_timestamps)):
    f.write(str(experiment_timestamps[index]));
    f.write('\n');
f.close();
print 'num of data points in timestamps'
print index+1
# --------------------------------------------------------------------------- #
print 'cool cool cool'
"""
print 'Writing to file'
fname = 'main_057_iss_02_elmax.txt'
f = open(fname,'w');
pass_duration = (time_index[1] - time_index[0])*delta_t;
f.write('pass duration in Tx FoR ='+str(pass_duration)+' s \n');
el_max = math.degrees(y_sph_tx[1,:].max());
f.write('maximum elevation angle ='+str(el_max)+' deg \n');
f.close();
print 'cool cool cool'
