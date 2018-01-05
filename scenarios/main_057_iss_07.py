# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:14:43 2017

Edited:
12.09.17: debugged

@author: Ashiv Dhondea
"""

import math
import numpy as np
import GeometryFunctions as GF

# --------------------------------------------------------------------------- #
with open('main_meerkat_radar_parameters_doreen.txt') as fp:
    for line in fp:
		if 'HPBW Tx' in line:
			good_index = line.index('=')
			beamwidth_tx = float(line[good_index+1:-1]);
fp.close();

# beamwidth of transmitter and receiver
beamwidth_tx = math.radians(beamwidth_tx); 
# ------------------------------------------------------------------------------------ #
print 'Loading data at high sampling rate'
timevec = np.load('main_057_iss_05_timevec.npy'); # timevector
# discretization step length/PRF
delta_t = timevec[2]-timevec[1];

time_index = np.load('main_057_iss_05_time_index.npy');
tx_el_min_index = time_index[0];
tx_el_max_index = time_index[1];
print 'bounds for Tx FoR'
print tx_el_min_index
print tx_el_max_index

time_index_rx = np.load('main_057_iss_06_time_index_rx.npy');
print 'bounds for Rx FoR'
print time_index_rx[0]
print time_index_rx[1]

overall_bound_lower = max(tx_el_min_index,time_index_rx[0]);
overall_bound_upper = min(tx_el_max_index,time_index_rx[1]);
print 'overall bounds'
print overall_bound_lower
print overall_bound_upper

overall_bound = np.array([overall_bound_lower,overall_bound_upper],dtype=np.int64);
np.save('main_057_iss_07_overall_bound.npy',overall_bound);
# --------------------------------------------------------------------------- #
print 'Loading data'
y_sph_tx = np.load('main_057_iss_05_y_sph_tx.npy'); # spherical measurement vectors in Tx frame
# --------------------------------------------------------------------------- # 
# Find the points which can possibly be assigned as beam centres
az_tx_nice = GF.fnSmoothe_AngleSeries(y_sph_tx[2,:],2*math.pi);
beam_centre_candidates = np.zeros([2,len(timevec)],dtype=np.float64);

print 'finding the beam centre candidates '
for i in range(overall_bound_lower,overall_bound_upper+1):
    beam_centre_candidates[:,i] = np.array([az_tx_nice[i],y_sph_tx[1,i]],dtype=np.float64);
np.save('main_057_iss_07_beam_centre_candidates.npy',beam_centre_candidates);
# --------------------------------------------------------------------------- #
span_el = math.degrees(beam_centre_candidates[1,overall_bound_upper] - beam_centre_candidates[1,overall_bound_lower]);
span_az = math.degrees(beam_centre_candidates[0,overall_bound_upper] - beam_centre_candidates[0,overall_bound_lower]);
print 'span in el and az. These should span at least the Tx beamwidth of %.3f' %math.degrees(beamwidth_tx)
print span_el
print span_az

tx_bw_el_range = np.where(abs(beam_centre_candidates[1,overall_bound_lower:] - beam_centre_candidates[1,overall_bound_lower]) >= 0.5*beamwidth_tx);
tx_bw_el_index_start = tx_bw_el_range[0][0] + overall_bound_lower; # earliest point where we can put the el beam's centre's start
tx_bw_az_range = np.where(abs(beam_centre_candidates[0,overall_bound_lower:] - beam_centre_candidates[0,overall_bound_lower]) >= 0.5*beamwidth_tx);
tx_bw_az_index_start = tx_bw_az_range[0][0] + overall_bound_lower; # earliest point where we can put the az beam's start
tx_bw_start_index_check = max(tx_bw_el_index_start,tx_bw_az_index_start); # earliest point where we can put the beam's start 
print 'tx bw start index'
print tx_bw_start_index_check
tx_bw_el_range_up = np.where(abs(beam_centre_candidates[1,overall_bound_lower:overall_bound_upper+1] -beam_centre_candidates[1,overall_bound_upper]) >= 0.5*beamwidth_tx );
tx_bw_el_index_end = tx_bw_el_range_up[0][-1] + overall_bound_lower;  # latest point where we can put the el beam's end
tx_bw_az_range_up = np.where(abs(beam_centre_candidates[0,overall_bound_lower:overall_bound_upper+1] - beam_centre_candidates[0,overall_bound_upper]) >= 0.5*beamwidth_tx );
tx_bw_az_index_end = tx_bw_az_range_up[0][-1] + overall_bound_lower; # latest point where we can put the az beam's end
tx_bw_end_index_check = min(tx_bw_el_index_end,tx_bw_az_index_end); # latest point where we can put the beam's end 
print 'tx bw end index'
print tx_bw_end_index_check

tx_bw_start_index = min(tx_bw_start_index_check,tx_bw_end_index_check);
tx_bw_end_index = max(tx_bw_start_index_check,tx_bw_end_index_check);

tx_bw_bounds = np.array([tx_bw_start_index,tx_bw_end_index],dtype=np.int64);

np.save('main_057_iss_07_tx_bw_bounds.npy',tx_bw_bounds);

# --------------------------------------------------------------------------- #
el_tx_argmax = np.argmax(y_sph_tx[1,tx_bw_start_index:tx_bw_end_index+1]) + tx_bw_start_index;
print 'el tx argmax'
print el_tx_argmax

tx_az_bins_up = np.zeros([len(timevec)],dtype=np.int64);
tx_az_bins_down = np.zeros([len(timevec)],dtype=np.int64);
tx_el_bins_up = np.zeros([len(timevec)],dtype=np.int64);
tx_el_bins_down = np.zeros([len(timevec)],dtype=np.int64);
tx_bins_length = np.zeros([len(timevec)],dtype=np.int64);
# Save the limits of the beam (i.e. indices of all data points within beam)
tx_beam_indices = np.zeros([2,len(timevec)],dtype=np.int64);

print 'finding the dwell time for the beam centres which happen before el tx argmax'
for i in range(tx_bw_start_index,tx_bw_end_index+1): # debugged 12.09.17
    tx_az_range_up   = np.where( abs(az_tx_nice[i:overall_bound_upper+1] - az_tx_nice[i] )<= 0.5*beamwidth_tx );
    tx_az_range_down = np.where( abs(az_tx_nice[overall_bound_lower:i] - az_tx_nice[i] )<= 0.5*beamwidth_tx);
    tx_az_bins_up[i] = tx_az_range_up[0][-1] + i;
    tx_az_bins_down[i] = tx_az_range_down[0][0]+overall_bound_lower;
    
    tx_el_range_up = np.where( abs(y_sph_tx[1,i:overall_bound_upper+1] - y_sph_tx[1,i] )<= 0.5*beamwidth_tx );
    tx_el_range_down = np.where(abs(y_sph_tx[1,overall_bound_lower:i] - y_sph_tx[1,i] )<= 0.5*beamwidth_tx);
    tx_el_bins_up[i] = tx_el_range_up[0][-1] + i; 
    tx_el_bins_down[i] = tx_el_range_down[0][0] + overall_bound_lower;
    
    tx_beam_indices[0,i] = max(tx_el_bins_down[i],tx_az_bins_down[i])
    tx_beam_indices[1,i] = min(tx_el_bins_up[i],tx_az_bins_up[i]);
    
    tx_bins_length[i] = tx_beam_indices[1,i] - tx_beam_indices[0,i];

np.save('main_057_iss_07_tx_bins_length.npy',tx_bins_length);
np.save('main_057_iss_07_tx_beam_indices.npy',tx_beam_indices); 
# --------------------------------------------------------------------------- #
print 'find the best positioning of the tx beam'
# Find index of maximum dwell time
tx_bw_time_max = np.argmax(tx_bins_length);
print 'index of maximum dwell time is'
print tx_bw_time_max
# Find indices of all data points within this interval
tx_beam_index_down = tx_beam_indices[0,tx_bw_time_max];
tx_beam_index_up = tx_beam_indices[1,tx_bw_time_max];
print 'lower and upper index of beam chosen'
print tx_beam_index_down
print tx_beam_index_up
print 'these span in terms of degrees (these shouldnt exceed the tx beamwidth)'
print np.degrees(y_sph_tx[1,tx_beam_index_up] - y_sph_tx[1,tx_beam_index_down] ) 
print np.degrees(y_sph_tx[2,tx_beam_index_up] - y_sph_tx[2,tx_beam_index_down]) 
#print 'index of el bins up'
#print tx_el_bins_up[i]
#print 'index of el bins down'
#print tx_el_bins_down[i]
#print 'index of az bins up'
#print tx_az_bins_up[i]
#print 'index of az bins down'
#print tx_az_bins_down[i]

tx_beam_indices_best = np.array([tx_beam_index_down,tx_bw_time_max,tx_beam_index_up],dtype=np.int64);
np.save('main_057_iss_07_tx_beam_indices_best.npy',tx_beam_indices_best);

print 'checking the angles'
print np.degrees(y_sph_tx[1,tx_beam_index_up] - y_sph_tx[1,tx_beam_index_down] ) 
print np.degrees(y_sph_tx[2,tx_beam_index_up] - y_sph_tx[2,tx_beam_index_down]) 
# --------------------------------------------------------------------------- # 
fname = 'main_057_iss_07_dwelltime.txt';
f = open(fname, 'w') # Create data file;
f.write(str(delta_t))
f.write('\n');
f.write(str(tx_beam_index_down))
f.write('\n');
f.write(str(tx_beam_index_up));
f.write('\n');
f.write(str(tx_bins_length[tx_bw_time_max]*delta_t));
f.write('\n');
f.write('note that these correspond to the square beam')
f.close();
# --------------------------------------------------------------------------- #
