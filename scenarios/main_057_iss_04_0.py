# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 15:31:18 2017

@author: Ashiv Dhondea
"""
import math
import numpy as np
import GeometryFunctions as GF

# Libraries needed for time keeping and formatting
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

# beamwidth of transmitter and receiver
beamwidth_tx = math.radians(beamwidth_tx); 
beamwidth_rx = math.radians(beamwidth_rx);
# --------------------------------------------------------------------------- #
print 'Loading data'
timevec = np.load('main_057_iss_02_timevec.npy'); # timevector
# discretization step length/PRF
delta_t = timevec[2]-timevec[1];
y_sph_tx = np.load('main_057_iss_02_y_sph_tx.npy'); # spherical measurement vectors in Tx frame
# --------------------------------------------------------------------------- #
# time stamps
experiment_timestamps = [None]*len(timevec)
index=0;
with open('main_057_iss_02_experiment_timestamps.txt') as fp:
    for line in fp:
        modified_timestring = line[:-1];
        experiment_timestamps[index] = aniso8601.parse_datetime(modified_timestring);
        index+=1;
fp.close();

norad_id = '25544'
# --------------------------------------------------------------------------- #  
az_tx_nice = GF.fnSmoothe_AngleSeries(y_sph_tx[2,:],2*math.pi);  
time_index_rx = np.load('main_057_iss_03_time_index_rx.npy');
## Bounds
print 'bounds for Rx FoR'
print time_index_rx[0]
print time_index_rx[1]
beam_centre_candidates = np.zeros([2,len(timevec)],dtype=np.float64);

print 'finding the beam centre candidates '
for i in range(time_index_rx[0],time_index_rx[1]+1):
    beam_centre_candidates[:,i] = np.array([az_tx_nice[i],y_sph_tx[1,i]],dtype=np.float64);
#np.save('main_051_iss_04_beam_centre_candidates.npy',beam_centre_candidates);

# --------------------------------------------------------------------------- #
tx_bw_el_range = np.where(abs(beam_centre_candidates[1,time_index_rx[0]:] - beam_centre_candidates[1,time_index_rx[0]]) >= 0.5*beamwidth_tx);
tx_bw_el_index_start = tx_bw_el_range[0][0] + time_index_rx[0]; # earliest point where we can put the el beam's centre's start
tx_bw_az_range = np.where(abs(beam_centre_candidates[0,time_index_rx[0]:] - beam_centre_candidates[0,time_index_rx[0]]) >= 0.5*beamwidth_tx);
tx_bw_az_index_start = tx_bw_az_range[0][0] + time_index_rx[0]; # earliest point where we can put the az beam's start
tx_bw_start_index = max(tx_bw_el_index_start,tx_bw_az_index_start); # earliest point where we can put the beam's start 
print 'tx bw start index'
print tx_bw_start_index
tx_bw_el_range_up = np.where(abs(beam_centre_candidates[1,tx_bw_start_index:time_index_rx[1]] -beam_centre_candidates[1,time_index_rx[1]]) >= 0.5*beamwidth_tx );
tx_bw_el_index_end = tx_bw_el_range_up[0][-1] + tx_bw_start_index;  # latest point where we can put the el beam's end
tx_bw_az_range_up = np.where(abs(beam_centre_candidates[0,tx_bw_start_index:time_index_rx[1]] - beam_centre_candidates[0,time_index_rx[1]]) >= 0.5*beamwidth_tx );
tx_bw_az_index_end = tx_bw_az_range_up[0][-1] + tx_bw_start_index; # latest point where we can put the az beam's end
tx_bw_end_index = min(tx_bw_el_index_end,tx_bw_az_index_end); # latest point where we can put the beam's end 
print 'tx bw end index'
print tx_bw_end_index
tx_bw_bounds = np.array([tx_bw_start_index,tx_bw_end_index],dtype=np.int64);
#np.save('main_051_iss_04_tx_bw_bounds.npy',tx_bw_bounds);
# --------------------------------------------------------------------------- #
el_tx_argmax = np.argmax(y_sph_tx[1,time_index_rx[0]:time_index_rx[1]+1]) + time_index_rx[0];
el_tx_max = y_sph_tx[1,el_tx_argmax]
el_tx_nice = np.zeros([len(timevec)],dtype=np.float64);
el_tx_nice[0:el_tx_argmax+1] = y_sph_tx[1,0:el_tx_argmax+1] ;
for i in range(el_tx_argmax+1,time_index_rx[1]+1+1):
    el_tx_nice[i] = abs(y_sph_tx[1,i]-el_tx_max) + el_tx_max;


tx_az_bins_up = np.zeros([len(timevec)],dtype=np.int64);
tx_az_bins_down = np.zeros([len(timevec)],dtype=np.int64);
tx_el_bins_up = np.zeros([len(timevec)],dtype=np.int64);
tx_el_bins_down = np.zeros([len(timevec)],dtype=np.int64);
tx_bins_length = np.zeros([len(timevec)],dtype=np.int64);
# Save the limits of the beam (i.e. indices of all data points within beam)
tx_beam_indices = np.zeros([2,len(timevec)],dtype=np.int64);

print 'finding the dwell time for the beam centres which happen before el tx argmax'
for i in range(tx_bw_start_index,el_tx_argmax):
    tx_az_range_up   = np.where( abs(az_tx_nice[i:el_tx_argmax] - az_tx_nice[i] )<= 0.5*beamwidth_tx );
    tx_az_range_down = np.where( abs(az_tx_nice[time_index_rx[0]:i] - az_tx_nice[i] )<= 0.5*beamwidth_tx);
    tx_az_bins_up[i] = tx_az_range_up[0][-1] + i;
    tx_az_bins_down[i] = tx_az_range_down[0][0]+time_index_rx[0];
    
    tx_el_range_up = np.where( abs(el_tx_nice[i:el_tx_argmax] - el_tx_nice[i] )<= 0.5*beamwidth_tx );
    tx_el_range_down = np.where(abs(el_tx_nice[time_index_rx[0]:i] - el_tx_nice[i] )<= 0.5*beamwidth_tx);
    tx_el_bins_up[i] = tx_el_range_up[0][-1] + i; 
    tx_el_bins_down[i] = tx_el_range_down[0][0] + time_index_rx[0];
    
    tx_beam_indices[0,i] = max(tx_el_bins_down[i],tx_az_bins_down[i])
    tx_beam_indices[1,i] = min(tx_el_bins_up[i],tx_az_bins_up[i]);
    
    tx_bins_length[i] = tx_beam_indices[1,i] - tx_beam_indices[0,i];

print 'finding the dwell time for the beam centres which happen after el tx argmax'
for i in range(el_tx_argmax+1,tx_bw_end_index+1):
    tx_az_range_up   = np.where( abs(az_tx_nice[i:time_index_rx[1]+1] - az_tx_nice[i] )<= 0.5*beamwidth_tx );
    tx_az_range_down = np.where( abs(az_tx_nice[el_tx_argmax:i] - az_tx_nice[i] )<= 0.5*beamwidth_tx);
    tx_az_bins_up[i] = tx_az_range_up[0][-1] + i;
    tx_az_bins_down[i] = tx_az_range_down[0][0]+el_tx_argmax;
    
    tx_el_range_up = np.where( abs(el_tx_nice[i:time_index_rx[1]+1] - el_tx_nice[i] )<= 0.5*beamwidth_tx );
    tx_el_range_down = np.where(abs(el_tx_nice[el_tx_argmax:i] - el_tx_nice[i] )<= 0.5*beamwidth_tx);
    tx_el_bins_up[i] = tx_el_range_up[0][-1] + i; 
    tx_el_bins_down[i] = tx_el_range_down[0][0]+el_tx_argmax;
    
    tx_beam_indices[0,i] = max(tx_el_bins_down[i],tx_az_bins_down[i])
    tx_beam_indices[1,i] = min(tx_el_bins_up[i],tx_az_bins_up[i]);
    
    tx_bins_length[i] = tx_beam_indices[1,i] - tx_beam_indices[0,i];
    
#np.save('main_057_iss_04_tx_bins_length.npy',tx_bins_length);
#np.save('main_057_iss_04_tx_beam_indices.npy',tx_beam_indices); 

# --------------------------------------------------------------------------- #
print 'find the best positioning of the tx beam'
# Find index of maximum dwell time
tx_bw_time_max = np.argmax(tx_bins_length);
print 'index of maximum dwell time is'
print tx_bw_time_max
print 'max dwell time'
print tx_bins_length[tx_bw_time_max]*delta_t
# Find indices of all data points within this interval
tx_beam_index_down = tx_beam_indices[0,tx_bw_time_max];
tx_beam_index_up = tx_beam_indices[1,tx_bw_time_max];
print 'lower and upper index of beam chosen'
print tx_beam_index_down
print tx_beam_index_up
print 'these span in terms of degrees (these shouldnt exceed the tx beamwidth)'
print np.degrees(y_sph_tx[1,tx_beam_index_up] - y_sph_tx[1,tx_beam_index_down] ) 
print np.degrees(y_sph_tx[2,tx_beam_index_up] - y_sph_tx[2,tx_beam_index_down]) 
print 'index of el bins up'
print tx_el_bins_up[i]
print 'index of el bins down'
print tx_el_bins_down[i]
print 'index of az bins up'
print tx_az_bins_up[i]
print 'index of az bins down'
print tx_az_bins_down[i]

tx_beam_indices_best = np.array([tx_beam_index_down,tx_bw_time_max,tx_beam_index_up],dtype=np.int64);
#np.save('main_057_iss_04_tx_beam_indices_best.npy',tx_beam_indices_best);

print 'checking the angles'
print np.degrees(y_sph_tx[1,tx_beam_index_up] - y_sph_tx[1,tx_beam_index_down] ) 
print np.degrees(y_sph_tx[2,tx_beam_index_up] - y_sph_tx[2,tx_beam_index_down]) 
# --------------------------------------------------------------------------- #
ground_station='Tx';
start_epoch = experiment_timestamps[time_index_rx[0]];
start_epoch = start_epoch.replace(tzinfo=None);
end_epoch = experiment_timestamps[time_index_rx[1]];
end_epoch = end_epoch.replace(tzinfo=None);

tx_beam_index_down_epoch = experiment_timestamps[tx_beam_index_down]
tx_beam_index_down_epoch = tx_beam_index_down_epoch.replace(tzinfo=None);

tx_bw_time_max_epoch = experiment_timestamps[tx_bw_time_max];
tx_bw_time_max_epoch = tx_bw_time_max_epoch.replace(tzinfo=None);

tx_beam_index_up_epoch = experiment_timestamps[tx_beam_index_up];
tx_beam_index_up_epoch =tx_beam_index_up_epoch .replace(tzinfo=None);

title_string = str(start_epoch.isoformat())+'Z/'+str(end_epoch.isoformat()+'Z');

##
fig = plt.figure(1);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig.suptitle(r"\textbf{Elevation angle to object %s from %s over %s}" %(norad_id,ground_station,title_string),fontsize=12);
plt.plot(timevec[time_index_rx[0]:time_index_rx[1]+1],np.degrees(y_sph_tx[1,time_index_rx[0]:time_index_rx[1]+1]))

plt.scatter(timevec[tx_beam_index_down],math.degrees(y_sph_tx[1,tx_beam_index_down]),s=50,marker=r"$\Box$",facecolors='none', edgecolors='crimson',label=r"%s"  %str(tx_beam_index_down_epoch.isoformat()+'Z'));
plt.scatter(timevec[tx_bw_time_max],math.degrees(y_sph_tx[1,tx_bw_time_max]),s=50,marker=r"$\oplus$",facecolors='none', edgecolors='purple',label=r"%s" %str(tx_bw_time_max_epoch.isoformat()+'Z'));
plt.scatter(timevec[tx_beam_index_up],math.degrees(y_sph_tx[1,tx_beam_index_up]),s=50,marker=r"$\circledcirc$",facecolors='none', edgecolors='darkgreen',label=r"%s" %str(tx_beam_index_up_epoch.isoformat()+'Z'));

plt.legend(loc='center left',title=r"\textbf{Timestamps}",bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True)

ax.set_ylabel(r"Elevation angle $\theta~[\mathrm{^\circ}]$")
ax.set_xlabel(r'Time $t~[\mathrm{s}]$');
at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
ax.add_artist(at)
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_04_0_el.pdf',bbox_inches='tight',pad_inches=0.11,dpi=10)

# ------------------------------------------------------------------------- #
kindex = np.arange(0,len(timevec),1,dtype=np.int64);


fig = plt.figure(2);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
avoid_singularity = 1550;
fig.suptitle(r"\textbf{Dwell-time duration in Tx beam for object %s over %s}" %(norad_id,title_string),fontsize=12);
plt.plot(kindex[tx_bw_start_index:el_tx_argmax-avoid_singularity], tx_bins_length[tx_bw_start_index:el_tx_argmax-avoid_singularity]*delta_t ,'b');

#plt.text(250000,5,r"\begin{tabular}{|c | c |}  \hline  $k_{\text{max}}$ & %d \\ \hline $\max \{ T_{i,Tx} \}$ & $%.3f~\mathrm{s}$ \\ \hline $\theta \lbrack k_{\text{max}}\rbrack$ & $.3f ^\circ$ \\ \hline  $\psi \lbrack k_{\text{max}}\rbrack$ & $.3f ^\circ$ \\ \hline \end{tabular}" %(tx_bw_time_max,tx_bins_length[tx_bw_time_max]*delta_t,math.degrees(y_sph_tx[1,tx_bw_time_max]),math.degrees(y_sph_tx[2,tx_bw_time_max])),size=12)

plt.text(250000,5,r"\begin{tabular}{|c | c |} \hline $k_{\text{max}}$ & $%d$ \\ \hline $\max \{ T_{i,Tx} \}$ & $%.3f~\mathrm{s}$ \\ \hline $\theta \lbrack k_{\text{max}}\rbrack$ & $%.3f ^\circ$ \\ \hline  $\psi \lbrack k_{\text{max}}\rbrack$ & $%.3f ^\circ$ \\ \hline \end{tabular}" %(tx_bw_time_max,tx_bins_length[tx_bw_time_max]*delta_t,math.degrees(y_sph_tx[1,tx_bw_time_max]),math.degrees(y_sph_tx[2,tx_bw_time_max])),size=12)


plt.plot(kindex[el_tx_argmax+avoid_singularity:tx_bw_end_index], tx_bins_length[el_tx_argmax+avoid_singularity:tx_bw_end_index]*delta_t ,'b');
plt.scatter(kindex[tx_bw_time_max],tx_bins_length[tx_bw_time_max]*delta_t,s=100,marker=r"$\square$",facecolors='none', edgecolors='red',label=r"%s" %str(tx_bw_time_max_epoch.isoformat())+'Z')
plt.legend(loc='best')
plt.xlim(kindex[0],kindex[-1]);
ax.set_xlabel(r'$k$')
ax.set_ylabel(r'Illumination Time $ T_{i,Tx}~[\mathrm{s}]$'); 
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
ax.add_artist(at)
fig.savefig('main_057_iss_04_0_dwelltime_tx0.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10)


beam_centre = np.degrees(np.array([y_sph_tx[2,tx_bw_time_max],y_sph_tx[1,tx_bw_time_max]],dtype=np.float64));
beam_xx = np.array([beam_centre[0] - 0.5*math.degrees(beamwidth_tx),beam_centre[0] + 0.5*math.degrees(beamwidth_tx)]);
beam_yy = np.array([beam_centre[1] - 0.5*math.degrees(beamwidth_tx),beam_centre[1] + 0.5*math.degrees(beamwidth_tx)]);
xx = np.array([beam_xx[0],beam_xx[1]],dtype=np.float64)
yy_below = np.array([beam_yy[0],beam_yy[0]],dtype=np.float64);
yy_above = np.array([beam_yy[1],beam_yy[1]],dtype=np.float64);
label_down = np.arange(np.degrees(y_sph_tx[2,tx_bw_time_max]), xx[1] + 0.1 +  0.05 , 0.05);
label_down_y = yy_below[1]*np.ones(len(label_down),dtype=np.float64);
label_up_y = yy_above[0]*np.ones(len(label_down),dtype=np.float64);



fig = plt.figure(3);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.axis('equal')
fig.suptitle(r"\textbf{Tx beam placement for the object %s trajectory during the interval %s}" %(norad_id,title_string),fontsize=12);
plt.plot(np.degrees(y_sph_tx[2,time_index_rx[0]:tx_bw_end_index+1]),np.degrees(y_sph_tx[1,time_index_rx[0]:tx_bw_end_index+1]))

ax.fill_between(xx,yy_below,yy_above,facecolor='gray',alpha=0.42);
ax.set_xlabel(r'Azimuth angle $\psi_{\text{Tx}}~[\mathrm{^\circ}]$')
ax.set_ylabel(r'Elevation angle $ \theta_{\text{Tx}}~[\mathrm{^\circ}]$'); 

plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
#plt.minorticks_on()
at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
ax.add_artist(at)

plt.legend(loc='center left',title=r"\textbf{Timestamps}",bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True)

fig.savefig('main_057_iss_04_0_txbeam_square.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10)

# ------------------------------------------------------------------------- #

plt_index = time_index_rx[0]
plt_index_epoch = experiment_timestamps[plt_index];
plt_index_epoch = plt_index_epoch.replace(tzinfo=None);

plt_index_end = tx_bw_time_max + 12000
plt_index_epoch_end = experiment_timestamps[plt_index_end];
plt_index_epoch_end = plt_index_epoch_end.replace(tzinfo=None);



title_string1 = str(plt_index_epoch.isoformat())+'Z/'+str(plt_index_epoch_end.isoformat()+'Z');

fig = plt.figure(4);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.axis('equal')
fig.suptitle(r"\textbf{Tx beam placement for the object %s trajectory during the interval %s}" %(norad_id,title_string1),fontsize=12);
plt.plot(np.degrees(y_sph_tx[2,plt_index:plt_index_end+1]),np.degrees(y_sph_tx[1,plt_index:plt_index_end+1]))

plt.scatter(math.degrees(y_sph_tx[2,tx_beam_index_down]),math.degrees(y_sph_tx[1,tx_beam_index_down]),s=100,marker=r"$\Box$",facecolors='none', edgecolors='crimson',label=r"%s"  %str(tx_beam_index_down_epoch.isoformat())+'Z')

plt.scatter(math.degrees(y_sph_tx[2,tx_bw_time_max]),math.degrees(y_sph_tx[1,tx_bw_time_max]),s=100,marker=r"$\oplus$",facecolors='none', edgecolors='purple',label=r"%s" %str(tx_bw_time_max_epoch.isoformat())+'Z')

plt.scatter(math.degrees(y_sph_tx[2,tx_beam_index_up]),math.degrees(y_sph_tx[1,tx_beam_index_up]),s=100,marker=r"$\circledcirc$",facecolors='none', edgecolors='darkgreen',label=r"%s" %str(tx_beam_index_up_epoch.isoformat()+'Z'));

ax.fill_between(xx,yy_below,yy_above,facecolor='gray',alpha=0.36);
ax.set_xlabel(r'Azimuth angle $\psi_{\text{Tx}}~[\mathrm{^\circ}]$')
ax.set_ylabel(r'Elevation angle $ \theta_{\text{Tx}}~[\mathrm{^\circ}]$'); 

plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
#plt.minorticks_on()
at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
ax.add_artist(at)

plt.legend(loc='center left',title=r"\textbf{Timestamps}",bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True)

fig.savefig('main_057_iss_04_0_txbeam_square_end.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10)


# ------------------------------------------------------------------------- # 
#fname = 'main_057_iss_04_dwelltime.txt';
#f = open(fname, 'w') # Create data file;
#f.write(str(delta_t))
#f.write('\n');
#f.write(str(tx_beam_index_down))
#f.write('\n');
#f.write(str(tx_beam_index_up));
#f.write('\n');
#f.write(str(tx_bins_length[tx_bw_time_max]*delta_t));
#f.write('\n');
#f.write('tx beam index down ='+str(experiment_timestamps[tx_beam_index_down]));
#f.write('\n');
#f.write('tx bw time max ='+str(experiment_timestamps[tx_bw_time_max]));
#f.write('\n');
#f.write('tx beam index up='+str(experiment_timestamps[tx_beam_index_up]))
#f.write('\n');
#
#f.write('note that these correspond to the square beam')
#f.close();
# --------------------------------------------------------------------------- #
print 'cool cool cool'
