# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:14:57 2017

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
start_vis_epoch = aniso8601.parse_datetime(start_timestring[:-1]);
start_vis_epoch = start_vis_epoch.replace(tzinfo=None);

end_vis_epoch = aniso8601.parse_datetime(end_timestring[:-1]);
end_vis_epoch = end_vis_epoch.replace(tzinfo=None);
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

lat_station = lat_denel; # [deg]
lon_station =  lon_denel; # [deg]
altitude_station = 129.e-3; # [km]
# --------------------------------------------------------------------------- #
## ISS (ZARYA)            
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

# Find the time elapsed between the start and end of the observation window
simulation_duration_dt_obj = end_vis_epoch- start_vis_epoch;
simulation_duration_secs = simulation_duration_dt_obj.total_seconds();

# Declare time and state vector variables.
delta_t = 1e-3; #[s]
print 'Propagation time step = %f' %delta_t, '[s]'
duration = simulation_duration_secs; #[s]
print 'Duration of simulation = %f' %duration, '[s]'
timevec = np.arange(0,duration+delta_t,delta_t,dtype=np.float64);

x_state_sgp4 = np.zeros([6,len(timevec)],dtype=np.float64);
R_SEZ = np.zeros([3,len(timevec)],dtype=np.float64);
V_SEZ = np.zeros([3,len(timevec)],dtype=np.float64);
x_target = np.zeros([6,len(timevec)],dtype=np.float64); 
#  spherical measurements from the Rx
y_sph_rx = np.zeros([3,len(timevec)],dtype=np.float64);
theta_GMST = np.zeros([len(timevec)],dtype=np.float64); 
xecef_sgp4 = np.zeros([3,len(timevec)],dtype=np.float64);
# Declare variables to store latitude and longitude values of the ground track
lat_sgp4 = np.zeros([len(timevec)],dtype=np.float64);
lon_sgp4 = np.zeros([len(timevec)],dtype=np.float64);

# timestamps for this experiment
experiment_timestamps = [None]*len(timevec);


for index in range(0,len(timevec)):
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
    R_SEZ[:,index] = AstFn.fnRAZEL_Cartesian(math.radians(lat_station),math.radians(lon_station),altitude_station,R_ECI,theta_GMST[index]); 
    R_ECEF = AstFn.fnECItoECEF(R_ECI,theta_GMST[index]);
    V_SEZ[:,index] = AstFn.fnVel_ECI_to_SEZ(V_ECI,R_ECEF,math.radians(lat_station),math.radians(lon_station),theta_GMST[index]);
    x_target[:,index] = np.hstack((R_SEZ[:,index],V_SEZ[:,index])); # state vector in SEZ frame
    # Calculate range and angles for system modelling.
    y_sph_rx[:,index] = UCM.fnCalculate_Spherical(R_SEZ[:,index]); # slant-range and look angles wrt to Rx

# --------------------------------------------------------------------------- #    
ground_station='Tx';

experiment_timestamps_start = THF.fnRead_Experiment_Timestamps(experiment_timestamps,0);
start_plot_epoch = THF.fnCalculate_DatetimeEpoch(timevec,0,experiment_timestamps_start);
end_plot_epoch = THF.fnCalculate_DatetimeEpoch(timevec,len(timevec)-1,experiment_timestamps_start);

start_plot_epoch = start_plot_epoch.replace(tzinfo=None);
end_plot_epoch = end_plot_epoch.replace(tzinfo=None);
title_string = str(start_plot_epoch.isoformat())+'Z/'+str(end_plot_epoch.isoformat())+'Z';
# --------------------------------------------------------------------------- # 
fig = plt.figure(1);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig.suptitle(r"\textbf{Elevation angle to %s from %s over %s}" %(so_name,ground_station,title_string),fontsize=12);
plt.plot(timevec,np.rad2deg(y_sph_rx[1,:]))
#at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
#at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
#ax.add_artist(at)
ax.set_ylabel(r"Elevation angle $\theta~[\mathrm{^\circ}]$")
ax.set_xlabel(r'Time $t~[\mathrm{s}]$');
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_01_el.pdf',bbox_inches='tight',pad_inches=0.11,dpi=40)


