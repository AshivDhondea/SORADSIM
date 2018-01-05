# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 13:58:47 2017

ISS (ZARYA)             
1 25544U 98067A   17253.93837963  .00001150  00000-0  24585-4 0  9991
2 25544  51.6444 330.8522 0003796 258.3764  78.6882 15.54163465 75088

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

# Importing what's needed for nice plots.
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params)
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

# Module for SGP4 orbit propagation
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

# --------------------------------------------------------------------------- #
# Location of Denel Bredasdorp
lat_denel = -34.6;
lon_denel = 20.316666666666666;

lat_meerkat_00 = -30.71292524;
lon_meerkat_00 =  21.44380306;

lat_station = lat_denel; # [deg]
lon_station =  lon_denel; # [deg]
altitude_station = 0.018; # [km]
# More specifically, the ground station has been placed at Menzies' building.
ground_station='Tx'

## ISS (ZARYA)       
tle_line1 = '1 25544U 98067A   17253.93837963  .00001150  00000-0  24585-4 0  9991';
tle_line2 = '2 25544  51.6444 330.8522 0003796 258.3764  78.6882 15.54163465 75088';

so_name = 'ISS (ZARYA)'           

# Read TLE to extract Keplerians and epoch. 
a,e,i,BigOmega,omega,E,nu,epoch  = AstFn.fnTLEtoKeps(tle_line1,tle_line2);
# Find Kepler period
T = AstFn.fnKeplerOrbitalPeriod(a);

# Create satellite object
satellite_obj = twoline2rv(tle_line1, tle_line2, wgs84);

line1 = (tle_line1);
line2 = (tle_line2);

# Figure out the TLE epoch 
year,dayy, hrs, mins, secs, millisecs = THF.fn_Calculate_Epoch_Time(epoch);
todays_date = THF.fn_epoch_date(year,dayy);
print "TLE epoch date is", todays_date
print "UTC time = ",hrs,"h",mins,"min",secs+millisecs,"s"
timestamp_tle_epoch = dt.datetime(year=todays_date.year,month=todays_date.month,day=todays_date.day,hour=hrs,minute=mins,second=secs,microsecond=int(millisecs),tzinfo= pytz.utc);

# --------------------------------------------------------------------------- #
test_observation_epoch= dt.datetime(year=todays_date.year,month=todays_date.month,day=todays_date.day+1,hour=10,minute=30,second=secs,microsecond=int(millisecs),tzinfo= pytz.utc);

simulation_duration_dt_obj = test_observation_epoch - timestamp_tle_epoch;
simulation_duration_secs = simulation_duration_dt_obj.total_seconds();

# Declare time and state vector variables.
delta_t = 1; #[s]
# --------------------------------------------------------------------------- #
# Find indices for all data points whose elevation angle exceeds 10 deg
min_el = math.radians(10.) 
    
import SatelliteVisibility as SatVis
start_vis_region,end_vis_region,timevec,y_sph_rx,experiment_timestamps,lat_sgp4,lon_sgp4 = SatVis.fnCheck_satellite_visibility(satellite_obj,lat_station,lon_station,altitude_station,timestamp_tle_epoch,delta_t,simulation_duration_secs,min_el);    

if start_vis_region == 0 & end_vis_region == 0:
    print 'The satellite will not be within LoS during the given period of time'
else:
    print 'The satellite will be within LoS within the given period of time'

experiment_timestamps_start = THF.fnRead_Experiment_Timestamps(experiment_timestamps,0);

start_vis_epoch = THF.fnCalculate_DatetimeEpoch(timevec,start_vis_region,experiment_timestamps_start);
end_vis_epoch = THF.fnCalculate_DatetimeEpoch(timevec,end_vis_region,experiment_timestamps_start);

# --------------------------------------------------------------------------- #
## Plot elevation angle alone; add in the timestamps and window duration
start_plot_index = start_vis_region-60;
end_plot_index = end_vis_region+60

start_plot_epoch = THF.fnCalculate_DatetimeEpoch(timevec,start_plot_index,experiment_timestamps_start);
end_plot_epoch = THF.fnCalculate_DatetimeEpoch(timevec,end_plot_index,experiment_timestamps_start);

start_plot_epoch = start_plot_epoch.replace(tzinfo=None);
end_plot_epoch = end_plot_epoch.replace(tzinfo=None);
start_vis_epoch = start_vis_epoch.replace(tzinfo=None);
end_vis_epoch = end_vis_epoch.replace(tzinfo=None);

title_string_obsv = str(start_plot_epoch.isoformat())+'Z/'+str(end_plot_epoch.isoformat())+'Z';

fig = plt.figure(1);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.axis('equal')
fig.suptitle(r"\textbf{Elevation angle to %s from %s over %s}" %(so_name,ground_station,title_string_obsv),fontsize=12);
plt.plot(timevec[start_plot_index:end_plot_index],np.rad2deg(y_sph_rx[1,start_plot_index:end_plot_index]))
plt.axvspan(timevec[start_vis_region],timevec[end_vis_region],facecolor='green',alpha=0.2);
#plt.annotate(r"%s" %str(start_vis_epoch.isoformat()+'Z'),(timevec[start_vis_region],math.degrees(y_sph_rx[1,start_vis_region])));
#plt.annotate(r"%s" %str(end_vis_epoch.isoformat())+'Z',(timevec[end_vis_region],math.degrees(y_sph_rx[1,end_vis_region])));


plt.scatter(timevec[start_vis_region],math.degrees(y_sph_rx[1,start_vis_region]),s=50,marker=r"$\Box$",facecolors='none', edgecolors='crimson',label=r"%s"  %str(start_vis_epoch.isoformat()+'Z'));
plt.scatter(timevec[end_vis_region],math.degrees(y_sph_rx[1,end_vis_region]),s=50,marker=r"$\circledcirc$",facecolors='none', edgecolors='purple',label=r"%s" %str(end_vis_epoch.isoformat()+'Z'));
plt.legend(loc='center left',title=r"\textbf{Timestamps}",bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True)

ax.set_ylabel(r"Elevation angle $\theta~[\mathrm{^\circ}]$")
ax.set_xlabel(r'Time $t~[\mathrm{s}$]');
at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
ax.add_artist(at)
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_00_el.pdf',bbox_inches='tight',pad_inches=0.11,dpi=10)

# --------------------------------------------------------------------------- #
experiment_timestamps_start = THF.fnRead_Experiment_Timestamps(experiment_timestamps,0);
start_simulation_epoch = THF.fnCalculate_DatetimeEpoch(timevec,0,experiment_timestamps_start);
end_simulation_epoch = THF.fnCalculate_DatetimeEpoch(timevec,len(timevec)-1,experiment_timestamps_start);

start_simulation_epoch = start_simulation_epoch.replace(tzinfo=None);
end_simulation_epoch = end_simulation_epoch.replace(tzinfo=None);
timestamp_tle_epoch = timestamp_tle_epoch.replace(tzinfo=None);

title_string = str(start_simulation_epoch.isoformat())+'Z/'+str(end_simulation_epoch.isoformat())+'Z';

f, axarr = plt.subplots(3,sharex=True);
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
plt.rc('font',family='helvetica');
f.suptitle(r"\textbf{Radar observation vectors from %s to %s over %s}" %(ground_station,so_name,title_string),fontsize=12)
axarr[0].plot(timevec[0:-1:100],y_sph_rx[0,0:-1:100])
axarr[0].set_ylabel(r'$\rho$');
axarr[0].set_title(r'Slant-range $\rho [\mathrm{km}]$')
axarr[1].plot(timevec[0:-1:100],np.degrees(y_sph_rx[1,0:-1:100]))
axarr[1].axvspan(timevec[start_vis_region],timevec[end_vis_region],facecolor='green',alpha=0.2);
axarr[1].set_title(r'Elevation angle $\theta~[\mathrm{^\circ}]$')
axarr[1].set_ylabel(r'$\theta$');

axarr[2].plot(timevec[0:-1:100],np.degrees(y_sph_rx[2,0:-1:100]))
axarr[2].set_title(r'Azimuth angle $\psi~[\mathrm{^\circ}]$')
axarr[2].set_ylabel(r'$ \psi$');
axarr[2].set_xlabel(r'Time $t~[\mathrm{s}$]');

axarr[0].grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
axarr[1].grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
axarr[2].grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')

at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
axarr[2].add_artist(at)
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0:2]], visible=False)
plt.subplots_adjust(hspace=0.4)
f.savefig('main_057_iss_00_radarvec.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10)

# --------------------------------------------------------------------------- #
coastline_data= np.loadtxt('Coastline.txt',skiprows=1)
w, h = plt.figaspect(0.5)
fig = plt.figure(figsize=(w,h))
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
plt.rc('font',family='helvetica');
params = {'legend.fontsize': 8,
    'legend.handlelength': 2}
plt.rcParams.update(params)

fig.suptitle(r"\textbf{%s ground track over the interval %s}" %(so_name,title_string),fontsize=12)
plt.plot(coastline_data[:,0],coastline_data[:,1],'g');
ax.set_xlabel(r'Longitude $[\mathrm{^\circ}]$',fontsize=12)
ax.set_ylabel(r'Latitude $[\mathrm{^\circ}]$',fontsize=12)

del_yticks=10;
yticks_range=np.arange(-90,90+del_yticks,del_yticks,dtype=np.int64)
del_xticks=30;
xticks_range=np.arange(-180,180+del_xticks,del_xticks,dtype=np.int64)

plt.xlim(xticks_range[0],xticks_range[-1]);
plt.ylim(yticks_range[0],yticks_range[-1]);
plt.yticks(yticks_range);
plt.xticks(xticks_range);
plt.scatter(math.degrees(lon_sgp4[0]),math.degrees(lat_sgp4[0]),s=80,marker=r"$\oplus$",facecolors='none', edgecolors='darkorange',label=timestamp_tle_epoch.isoformat() + 'Z');

plt.plot(np.rad2deg(lon_sgp4[1:start_vis_region:2]),np.rad2deg(lat_sgp4[1:start_vis_region:2]),'b.',markersize=1);

plt.annotate(r'%s' %str(start_vis_epoch.isoformat()+'Z'), xy=(math.degrees(lon_sgp4[start_vis_region]),math.degrees(lat_sgp4[start_vis_region])),  xycoords='data',
            xytext=(math.degrees(lon_sgp4[start_vis_region])-75,math.degrees(lat_sgp4[start_vis_region])-20),
            arrowprops=dict(facecolor='black',shrink=0.05,width=0.1,headwidth=2))

plt.plot(np.rad2deg(lon_sgp4[start_vis_region:end_vis_region+1]),np.rad2deg(lat_sgp4[start_vis_region:end_vis_region+1]),color='crimson',linewidth=2);

plt.scatter(math.degrees(lon_sgp4[-1]),math.degrees(lat_sgp4[-1]),s=80,marker=r"$\Box$",facecolors='none', edgecolors='crimson',label=str(end_simulation_epoch.isoformat())+'Z');

plt.annotate(r'%s' %str(end_vis_epoch.isoformat())+'Z', xy=(math.degrees(lon_sgp4[end_vis_region]),math.degrees(lat_sgp4[end_vis_region])),  xycoords='data',
            xytext=(math.degrees(lon_sgp4[end_vis_region])-55,math.degrees(lat_sgp4[end_vis_region])-30),
            arrowprops=dict(facecolor='black', shrink=0.05,width=0.1,headwidth=2)
            )

plt.plot(np.rad2deg(lon_sgp4[end_vis_region+1:-1:2]),np.rad2deg(lat_sgp4[end_vis_region+1:-1:2]),'b.',markersize=1);

ax.grid(True);
plt.plot(lon_station,lat_station,marker='.',color='gray'); # station lat lon
ax.annotate(r'\textbf{Tx}', (16, -40));
plt.plot(lon_meerkat_00,lat_meerkat_00,marker='.',color='gray'); # rx lat lon
ax.annotate(r'\textbf{Rx}', (19, -29));

at = AnchoredText("AshivD",prop=dict(size=5), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)

plt.legend(loc='lower left',title=r"\textbf{Start \& end times}");
ax.get_legend().get_title().set_fontsize('10')

fig.savefig('main_057_iss_00_groundtrack.pdf',format='pdf',bbox_inches='tight',pad_inches=0.01,dpi=10);

# --------------------------------------------------------------------------- #
print 'Writing to file'
fname = 'main_057_iss_00_visibility.txt'
f = open(fname,'w');
visibility_window = str(start_vis_epoch.isoformat())+'Z/'+str(end_vis_epoch.isoformat())+'Z';
f.write('visibility interval in Tx FoR ='+visibility_window+'\n');


f.close();
print 'cool cool cool'



