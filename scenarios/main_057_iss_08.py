# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:16:19 2017

@author: Ashiv Dhondea
"""

import AstroFunctions as AstFn
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
from mpl_toolkits.mplot3d import Axes3D
# --------------------------------------------------------------------------- #
with open('main_meerkat_radar_parameters_doreen.txt') as fp:
    for line in fp:
		if 'HPBW Tx' in line:
			good_index = line.index('=')
			beamwidth_tx = float(line[good_index+1:-1]);
fp.close();
# --------------------------------------------------------------------------- #
print 'Loading data'
timevec = np.load('main_057_iss_05_timevec.npy'); # timevector
y_sph_tx = np.load('main_057_iss_05_y_sph_tx.npy'); # spherical measurement vectors in Tx frame

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
# Bistatic Radar characteristics
# beamwidth of transmitter and receiver
beamwidth_tx = math.radians(beamwidth_tx );
# --------------------------------------------------------------------------- # 
#az_tx_nice = GF.fnSmoothe_AngleSeries(y_sph_tx[2,:],2*math.pi);  
time_index_rx = np.load('main_057_iss_06_time_index_rx.npy');
beam_centre_candidates = np.load('main_057_iss_07_beam_centre_candidates.npy');
tx_bw_bounds = np.load('main_057_iss_07_tx_bw_bounds.npy');
tx_bins_length = np.load('main_057_iss_07_tx_bins_length.npy');
tx_beam_indices = np.load('main_057_iss_07_tx_beam_indices.npy');
tx_beam_indices_best = np.load('main_057_iss_07_tx_beam_indices_best.npy');
overall_bound = np.load('main_057_iss_07_overall_bound.npy');
# --------------------------------------------------------------------------- #
# sort out a few variables
tx_bw_time_max = tx_beam_indices_best[1];
tx_beam_index_down = tx_beam_indices_best[0];
tx_beam_index_up = tx_beam_indices_best[2];
tx_bw_start_index = tx_bw_bounds[0];
tx_bw_end_index = tx_bw_bounds[1];

overall_bound_lower = overall_bound[0]
overall_bound_upper = overall_bound[1]

print 'tx beam index down, max and up'
print tx_beam_index_down
print tx_bw_time_max
print tx_beam_index_up

beam_centre = np.degrees(beam_centre_candidates[:,tx_bw_time_max]);

withincircle = np.empty([len(timevec)],dtype=bool);
for i in range(tx_beam_index_down,tx_beam_index_up+1):
    testpt = np.degrees(beam_centre_candidates[:,i]);
    withincircle[i] = GF.fnCheck_IsInCircle(beam_centre,0.5*math.degrees(beamwidth_tx),testpt);
early = np.where(withincircle[tx_beam_index_down:tx_beam_index_up] == True);
earliest_pt = early[0][0] + tx_beam_index_down;
latest_pt = early[0][-1] + tx_beam_index_down

print 'earliest and latest points for Tx beam'
print earliest_pt
print latest_pt

print 'actual dwell time'
print (latest_pt - earliest_pt)*delta_t

#el_tx_argmax = np.argmax(y_sph_tx[1,time_index_rx[0]:time_index_rx[1]+1]) + time_index_rx[0];
tx_beam_circ_index = np.array([earliest_pt,tx_bw_time_max,latest_pt],dtype=np.int64);
np.save('main_057_iss_08_tx_beam_circ_index.npy',tx_beam_circ_index)

title_string1 = str(experiment_timestamps[tx_bw_start_index].isoformat())+'/'+str(experiment_timestamps[tx_bw_end_index].isoformat());
# --------------------------------------------------------------------------- #
# Find the epoch of the relevant data points
plot_lim = 1.0;
plt_start_index = tx_beam_index_down - int(plot_lim/delta_t)
plt_end_index = tx_beam_index_up+1 + int(plot_lim/delta_t)

start_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec,plt_start_index,experiment_timestamps[0]);
end_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec,plt_end_index,experiment_timestamps[0]);
tx_beam_index_down_epoch = THF.fnCalculate_DatetimeEpoch(timevec,tx_beam_index_down,experiment_timestamps[0]);
tx_beam_index_up_epoch = THF.fnCalculate_DatetimeEpoch(timevec,tx_beam_index_up,experiment_timestamps[0]);
tx_bw_time_max_epoch = THF.fnCalculate_DatetimeEpoch(timevec,tx_bw_time_max,experiment_timestamps[0]);

end_epoch_test = end_epoch_test .replace(tzinfo=None);
start_epoch_test = start_epoch_test .replace(tzinfo=None)
title_string = str(start_epoch_test.isoformat())+'Z/'+str(end_epoch_test.isoformat())+'Z';

tx_beam_index_down_epoch = tx_beam_index_down_epoch.replace(tzinfo=None);
tx_beam_index_up_epoch = tx_beam_index_up_epoch.replace(tzinfo=None)
tx_bw_time_max_epoch = tx_bw_time_max_epoch.replace(tzinfo=None)

earliest_pt_epoch = THF.fnCalculate_DatetimeEpoch(timevec,earliest_pt,experiment_timestamps[0]);
latest_pt_epoch = THF.fnCalculate_DatetimeEpoch(timevec,latest_pt,experiment_timestamps[0]);

earliest_pt_epoch= earliest_pt_epoch.replace(tzinfo=None)
latest_pt_epoch= latest_pt_epoch.replace(tzinfo=None)
# --------------------------------------------------------------------------- #
beam_centre = np.degrees(np.array([y_sph_tx[2,tx_bw_time_max],y_sph_tx[1,tx_bw_time_max]],dtype=np.float64));
beam_xx = np.array([beam_centre[0] - 0.5*math.degrees(beamwidth_tx),beam_centre[0] + 0.5*math.degrees(beamwidth_tx)]);
beam_yy = np.array([beam_centre[1] - 0.5*math.degrees(beamwidth_tx),beam_centre[1] + 0.5*math.degrees(beamwidth_tx)]);
xx = np.array([beam_xx[0],beam_xx[1]],dtype=np.float64)
yy_below = np.array([beam_yy[0],beam_yy[0]],dtype=np.float64);
yy_above = np.array([beam_yy[1],beam_yy[1]],dtype=np.float64);
label_down = np.arange(np.degrees(y_sph_tx[2,tx_bw_time_max]), xx[1] + 0.1 +  0.05 , 0.05);
label_down_y = yy_below[1]*np.ones(len(label_down),dtype=np.float64);
label_up_y = yy_above[0]*np.ones(len(label_down),dtype=np.float64);


xx_below = np.array([beam_xx[0],beam_xx[0]],dtype=np.float64);
xx_above = np.array([beam_xx[1],beam_xx[1]],dtype=np.float64);
label_up = np.arange(beam_yy[0]-0.05,np.degrees(y_sph_tx[1,tx_bw_time_max])+0.04, 0.04);
label_up_x = xx_below[1]*np.ones(len(label_up),dtype=np.float64);
label_down_x = xx_above[0]*np.ones(len(label_up),dtype=np.float64);


numpts=360
circpts = GF.fnCalculate_CircumferencePoints(beam_centre,0.5*math.degrees(beamwidth_tx),numpts)

# --------------------------------------------------------------------------- #
#time_index = np.load('main_057_iss_05_time_index.npy')
ground_station='Tx';
#fig = plt.figure(6);
#ax = fig.gca()
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#fig.suptitle(r"\textbf{Elevation angle to object %s from %s }" %(norad_id,ground_station),fontsize=12);
#plt.plot(timevec[time_index_rx[0]:time_index_rx[1]+1],np.rad2deg(y_sph_tx[1,time_index_rx[0]:time_index_rx[1]+1]))
#plt.axvspan(timevec[overall_bound[0]],timevec[overall_bound[1]],facecolor='green',alpha=0.2);
#plt.scatter(timevec[tx_beam_index_down],math.degrees(y_sph_tx[1,tx_beam_index_down]),s=50,marker=r"$\Box$",facecolors='none', edgecolors='crimson',label=r"%s"  %str(tx_beam_index_down_epoch.isoformat()+'Z'));
#plt.scatter(timevec[tx_bw_time_max],math.degrees(y_sph_tx[1,tx_bw_time_max]),s=50,marker=r"$\oplus$",facecolors='none', edgecolors='purple',label=r"%s" %str(tx_bw_time_max_epoch.isoformat()+'Z'));
#plt.scatter(timevec[tx_beam_index_up],math.degrees(y_sph_tx[1,tx_beam_index_up]),s=50,marker=r"$\circledcirc$",facecolors='none', edgecolors='darkgreen',label=r"%s" %str(tx_beam_index_up_epoch.isoformat()+'Z'));
#
#plt.legend(loc='center left',title=r"\textbf{Timestamps}",bbox_to_anchor=(1, 0.5),
#          fancybox=True, shadow=True)
#
#ax.set_ylabel(r"Elevation angle $\theta~[\mathrm{^\circ}]$")
#ax.set_xlabel(r'Time $t~[\mathrm{s}]$');
#at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
#at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
#ax.add_artist(at)
#plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
#fig.savefig('main_057_iss_08_el.pdf',bbox_inches='tight',pad_inches=0.11,dpi=20)

print math.degrees(beamwidth_tx)
diff = math.degrees(y_sph_tx[1,tx_beam_index_up] - y_sph_tx[1,tx_beam_index_down])
print 'diff'
print diff
diff_up = math.degrees(y_sph_tx[1,tx_beam_index_up] - y_sph_tx[1,tx_bw_time_max])
print 'diff up'
print diff_up
diff_down = math.degrees(y_sph_tx[1,tx_bw_time_max] - y_sph_tx[1,tx_beam_index_down])
print 'diff down'
print diff_down

diff_az = math.degrees(y_sph_tx[2,tx_beam_index_up] - y_sph_tx[2,tx_beam_index_down])
print 'diff az'
print diff_az
diff_up_az = math.degrees(y_sph_tx[2,tx_beam_index_up] - y_sph_tx[2,tx_bw_time_max])
print 'diff up az'
print diff_up_az
diff_down_az = math.degrees(y_sph_tx[2,tx_bw_time_max] - y_sph_tx[2,tx_beam_index_down])
print 'diff down az'
print diff_down_az

# --------------------------------------------------------------------------- #

fig = plt.figure(1);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.axis('equal')
fig.suptitle(r"\textbf{Tx beam placement for object %s trajectory during %s}" %(norad_id,title_string),fontsize=12);
plt.plot(np.degrees(y_sph_tx[2,plt_start_index :tx_beam_index_down]),np.degrees(y_sph_tx[1,plt_start_index :tx_beam_index_down]),color='blue',linestyle='dashed')
plt.plot(np.degrees(y_sph_tx[2,tx_beam_index_up+1:plt_end_index]),np.degrees(y_sph_tx[1,tx_beam_index_up+1:plt_end_index]),color='blue',linestyle='dashed')
plt.plot(np.degrees(y_sph_tx[2,tx_beam_index_down:tx_beam_index_up]),np.degrees(y_sph_tx[1,tx_beam_index_down:tx_beam_index_up]),color='blue')
plt.scatter(np.degrees(y_sph_tx[2,tx_beam_index_down]),np.degrees(y_sph_tx[1,tx_beam_index_down]),s=50,marker=r"$\diamond$",facecolors='none',edgecolors='red',label=r"%s" %str(tx_beam_index_down_epoch.isoformat())+'Z');
plt.scatter(np.degrees(y_sph_tx[2,tx_bw_time_max]),np.degrees(y_sph_tx[1,tx_bw_time_max]),s=50,marker=r"$\otimes$",facecolors='none',edgecolors='green',label=r"%s" %str(tx_bw_time_max_epoch.isoformat())+'Z')
plt.scatter(np.degrees(y_sph_tx[2,tx_beam_index_up]),np.degrees(y_sph_tx[1,tx_beam_index_up]),s=50,marker=r"$\boxplus$",facecolors='none', edgecolors='magenta',label=r"%s" %str(tx_beam_index_up_epoch.isoformat())+'Z')

ax.fill_between(xx,yy_below,yy_above,facecolor='gray',alpha=0.2);

#plt.annotate(
#    '', xy=(xx[1]+0.1, yy_below[1] ), xycoords='data',
#    xytext=(xx[1]+0.1,yy_above[0]), textcoords='data',
#    arrowprops={'arrowstyle': '<->'})
#
#plt.text(xx[1]+0.13,yy_below[1]+math.degrees(beamwidth_tx*0.5), r"$\Theta_{3~\mathrm{dB}}^{\text{Tx}} =%.3f \mathrm{^\circ}$" %math.degrees(beamwidth_tx))

plt.annotate(
    '', xy=(xx[0],yy_below[0]-0.05), xycoords='data',
    xytext=(xx[1],yy_below[0]-0.05), textcoords='data',
    arrowprops={'arrowstyle': '<->'})
plt.text(xx[0]+0.5*(xx[1]-xx[0]), yy_below[0]-0.16,r"$\Theta_{3~\mathrm{dB}}^{\text{Tx}} =%.3f \mathrm{^\circ}$" %math.degrees(beamwidth_tx))


plt.plot(label_down,label_down_y,color='darkslategray',linestyle='dotted')
plt.plot(label_down,label_up_y,color='darkslategray',linestyle='dotted')
ax.set_xlabel(r'Azimuth angle $\psi_{\text{Tx}}~[\mathrm{^\circ}]$')
ax.set_ylabel(r'Elevation angle $ \theta_{\text{Tx}}~[\mathrm{^\circ}]$'); 

plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
#plt.minorticks_on()
at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
ax.add_artist(at)

plt.legend(loc='center left',title=r"\textbf{Timestamps}",bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True)

fig.savefig('main_057_iss_08_txbeam_square0.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10)

# --------------------#

fig = plt.figure(2);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.axis('equal')
fig.suptitle(r"\textbf{Tx beam placement for object %s trajectory during %s}" %(norad_id,title_string),fontsize=12);
plt.plot(np.degrees(y_sph_tx[2,plt_start_index:tx_beam_index_down]),np.degrees(y_sph_tx[1,plt_start_index:tx_beam_index_down]),color='blue',linestyle='dashed')
plt.plot(np.degrees(y_sph_tx[2,tx_beam_index_up+1:plt_end_index]),np.degrees(y_sph_tx[1,tx_beam_index_up+1:plt_end_index]),color='blue',linestyle='dashed')
plt.plot(np.degrees(y_sph_tx[2,tx_beam_index_down:tx_beam_index_up]),np.degrees(y_sph_tx[1,tx_beam_index_down:tx_beam_index_up]),color='blue')
plt.scatter(np.degrees(y_sph_tx[2,tx_beam_index_down]),np.degrees(y_sph_tx[1,tx_beam_index_down]),s=50,marker=r"$\diamond$",facecolors='none',edgecolors='red',label=r"%s" %str(tx_beam_index_down_epoch.isoformat())+'Z');
plt.scatter(np.degrees(y_sph_tx[2,tx_bw_time_max]),np.degrees(y_sph_tx[1,tx_bw_time_max]),s=50,marker=r"$\otimes$",facecolors='none',edgecolors='green',label=r"%s" %str(tx_bw_time_max_epoch.isoformat())+'Z')
plt.scatter(np.degrees(y_sph_tx[2,tx_beam_index_up]),np.degrees(y_sph_tx[1,tx_beam_index_up]),s=50,marker=r"$\boxplus$",facecolors='none', edgecolors='magenta',label=r"%s" %str(tx_beam_index_up_epoch.isoformat())+'Z')

plt.plot(circpts[0,0:-1:2],circpts[1,0:-1:2],color='teal')

ax.fill_between(xx,yy_below,yy_above,facecolor='gray',alpha=0.2);

#plt.annotate(
#    '', xy=(xx[1]+0.1, yy_below[1] ), xycoords='data',
#    xytext=(xx[1]+0.1,yy_above[0]), textcoords='data',
#    arrowprops={'arrowstyle': '<->'})
#plt.text(xx[1]+0.15,yy_below[1]+math.degrees(beamwidth_tx*0.5), r"$\Theta_{3~\mathrm{dB}}^{\text{Tx}} =%.3f \mathrm{^\circ}$" %math.degrees(beamwidth_tx))

plt.annotate(
    '', xy=(xx[0],yy_below[0]-0.05), xycoords='data',
    xytext=(xx[1],yy_below[0]-0.05), textcoords='data',
    arrowprops={'arrowstyle': '<->'})
plt.text(xx[0]+0.5*(xx[1]-xx[0]), yy_below[0]-0.16,r"$\Theta_{3~\mathrm{dB}}^{\text{Tx}} =%.3f \mathrm{^\circ}$" %math.degrees(beamwidth_tx))

plt.plot(label_up_x,label_up,color='darkslategray',linestyle='dotted')
plt.plot(label_down_x,label_up,color='darkslategray',linestyle='dotted')

ax.set_xlabel(r'Azimuth angle $\psi_{\text{Tx}}~[\mathrm{^\circ}]$')
ax.set_ylabel(r'Elevation angle $ \theta_{\text{Tx}}~[\mathrm{^\circ}]$'); 

plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
#plt.minorticks_on()
at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
ax.add_artist(at);
plt.legend(loc='center left',title=r"\textbf{Timestamps}",bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True)
          
plt.ylim(11.3,13.1)
fig.savefig('main_057_iss_08_txbeam_squarecirc0.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10)

# ---------------------------- #
fig = plt.figure(3);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.axis('equal')
fig.suptitle(r"\textbf{Tx beam placement for object %s trajectory during %s}" %(norad_id,title_string),fontsize=10);
plt.plot(np.degrees(y_sph_tx[2,plt_start_index:tx_beam_index_down+100:100]),np.degrees(y_sph_tx[1,plt_start_index :tx_beam_index_down+100:100]),color='blue',linestyle='dashed')
plt.plot(np.degrees(y_sph_tx[2,tx_beam_index_up+1:plt_end_index+100:100]),np.degrees(y_sph_tx[1,tx_beam_index_up+1:plt_end_index+100:100]),color='blue',linestyle='dashed')
plt.plot(np.degrees(y_sph_tx[2,tx_beam_index_down:tx_beam_index_up+100:100]),np.degrees(y_sph_tx[1,tx_beam_index_down:tx_beam_index_up+100:100]),color='blue')

plt.scatter(np.degrees(y_sph_tx[2,earliest_pt]),np.degrees(y_sph_tx[1,earliest_pt]),s=50,marker=r"$\square$",facecolors='none', edgecolors='red',label=r"%s" %str(earliest_pt_epoch.isoformat())+'Z')
plt.scatter(np.degrees(y_sph_tx[2,tx_bw_time_max]),np.degrees(y_sph_tx[1,tx_bw_time_max]),s=50,marker=r"$\otimes$",facecolors='none',edgecolors='green',label=r"%s" %str(tx_bw_time_max_epoch.isoformat())+'Z')
plt.scatter(np.degrees(y_sph_tx[2,latest_pt]),np.degrees(y_sph_tx[1,latest_pt]),s=50,marker=r"$\circledast$",facecolors='none', edgecolors='magenta',label=r"%s" %str(latest_pt_epoch.isoformat())+'Z')

for p in [
    patches.Circle(
        (np.degrees(y_sph_tx[2,tx_bw_time_max]),np.degrees(y_sph_tx[1,tx_bw_time_max])),0.5*math.degrees(beamwidth_tx),
        color = 'gray',
        alpha=0.3
    ),
]:
    ax.add_patch(p)

#plt.annotate(
#    '', xy=(xx[1]+0.1, yy_below[1] ), xycoords='data',
#    xytext=(xx[1]+0.1,yy_above[0]), textcoords='data',
#    arrowprops={'arrowstyle': '<->'})
#plt.text(xx[1]+0.15,yy_below[1]+math.degrees(beamwidth_tx*0.5), r"$\Theta_{3~\mathrm{dB}}^{\text{Tx}} =%.3f \mathrm{^\circ}$" %math.degrees(beamwidth_tx))

plt.annotate(
    '', xy=(xx[0],yy_below[0]-0.05), xycoords='data',
    xytext=(xx[1],yy_below[0]-0.05), textcoords='data',
    arrowprops={'arrowstyle': '<->'})
plt.text(xx[0]+0.5*(xx[1]-xx[0]), yy_below[0]-0.16,r"$\Theta_{3~\mathrm{dB}}^{\text{Tx}} =%.3f \mathrm{^\circ}$" %math.degrees(beamwidth_tx))

plt.plot(label_up_x,label_up,color='darkslategray',linestyle='dotted')
plt.plot(label_down_x,label_up,color='darkslategray',linestyle='dotted')

plt.text(178.5,12.7,r"\begin{tabular}{|c | c |} \hline $k_{\text{max}}$ & $%d$ \\ \hline $\max \{ T_{i,Tx} \}$ & $%.6f~\mathrm{s}$ \\ \hline $\theta_\text{\gls{Tx}} \lbrack k_{\text{max}}\rbrack$ & $%.6f ^\circ$ \\ \hline  $\psi_\text{\gls{Tx}} \lbrack k_{\text{max}}\rbrack$ & $%.6f ^\circ$ \\ \hline \end{tabular}" %(tx_bw_time_max,(latest_pt - earliest_pt)*delta_t,math.degrees(y_sph_tx[1,tx_bw_time_max]),math.degrees(y_sph_tx[2,tx_bw_time_max])),size=12)


ax.set_xlabel(r'Azimuth angle $\psi_{\text{Tx}}~[\mathrm{^\circ}]$')
ax.set_ylabel(r'Elevation angle $ \theta_{\text{Tx}}~[\mathrm{^\circ}]$'); 

plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
ax.add_artist(at)

plt.legend(loc='center left',title=r"\textbf{Timestamps}",bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True)

plt.ylim(11.3,13.1)
plt.xlim(176.7,179.5)
fig.savefig('main_057_iss_08_txbeam_circ0.pdf',bbox_inches='tight',pad_inches=0.09,dpi=10)
# ------------------------------------------------------------------------------------------------------------------- #
"""
fig = plt.figure(4);
ax = fig.gca()
plt.rc('text', usetex=True)
plt.rc('font', family='serif');

fig.suptitle(r"\textbf{Dwell-time duration in Tx beam for the object %s trajectory on %s}" %(norad_id,title_string1),fontsize=12);
plt.plot(timevec[tx_bw_start_index:tx_bw_end_index],tx_bins_length[tx_bw_start_index:tx_bw_end_index]*delta_t);
plt.scatter(timevec[tx_bw_time_max],tx_bins_length[tx_bw_time_max]*delta_t,s=100,marker=r"$\square$",facecolors='none', edgecolors='red',label=r"%s" %str(tx_bw_time_max_epoch.isoformat())+'Z')
plt.legend(loc='best')
plt.xlim(timevec[tx_bw_start_index],timevec[tx_bw_end_index]);
ax.set_xlabel(r'Time $t~[\mathrm{s}]$')
ax.set_ylabel(r'Illumination Time $ T_{i,Tx}~[\mathrm{s}]$'); 
at = AnchoredText(r"$\Delta_t = %f ~\mathrm{s}$" %delta_t,prop=dict(size=6), frameon=True,loc=4)
at.patch.set_boxstyle("round,pad=0.05,rounding_size=0.2")
ax.add_artist(at)
plt.ylim(0,11)
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
fig.savefig('main_057_iss_08_dwelltime_tx.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10)
"""
# ----------------------------------------------------------------------------------------------------------------------- #
print 'cool cool cool'
