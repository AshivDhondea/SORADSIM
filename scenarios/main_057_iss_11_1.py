# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 23:40:47 2017
edited: 24 nov 2017
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
        if 'HPBW Tx' in line:
            good_index = line.index('=')
            beamwidth_tx = float(line[good_index+1:-1]);
        if 'HPBW Rx' in line:
            good_index = line.index('=')
            beamwidth_rx = float(line[good_index+1:-1]);
fp.close();
# --------------------------------------------------------------------------- #
# Bistatic Radar characteristics
# beamwidth of transmitter and receiver
beamwidth_rx = math.radians(beamwidth_rx);
beamwidth_tx = math.radians(beamwidth_tx);

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

x_target_02 = np.load('main_057_iss_02_x_target.npy');
timevec_02 = np.load('main_057_iss_02_timevec.npy');
experiment_timestamps_02 = [None]*len(timevec_02)
index=0;
with open('main_057_iss_02_experiment_timestamps.txt') as fp:
    for line in fp:
        modified_timestring = line[:-1];
        experiment_timestamps_02[index] = aniso8601.parse_datetime(modified_timestring);
        index+=1;
fp.close();
# --------------------------------------------------------------------------- #
y_sph_rx = np.load('main_057_iss_05_y_sph_rx.npy'); # spherical measurement vectors in Rx frame
y_sph_rx_meerkat_01 = np.load('main_057_iss_05_y_sph_rx_meerkat_01.npy'); 
y_sph_rx_meerkat_02 = np.load('main_057_iss_05_y_sph_rx_meerkat_02.npy'); 
theta_GMST = np.load('main_057_iss_05_theta_GMST.npy');
y_sph_tx = np.load('main_057_iss_05_y_sph_tx.npy'); 
x_target = np.load('main_057_iss_05_x_target.npy');
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
#plot_lim = 4
plt_start_index = 0#tx_beam_index_down - int(plot_lim/delta_t)
plt_end_index = 120000#len(timevec_02)-1#tx_beam_index_up+1 + int(6/delta_t) 

start_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec_02,plt_start_index,experiment_timestamps_02[0]);
end_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec_02,plt_end_index,experiment_timestamps_02[0]);

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
Tx_ecef = AstFn.fnRadarSite(math.radians(lat_denel),math.radians(lon_denel),altitude_denel);
Rx1_ecef = AstFn.fnRadarSite(math.radians(lat_meerkat_01),math.radians(lon_meerkat_01),altitude_meerkat);
Rx2_ecef = AstFn.fnRadarSite(math.radians(lat_meerkat_02),math.radians(lon_meerkat_02),altitude_meerkat);
Rx3_ecef = AstFn.fnRadarSite(math.radians(lat_meerkat_03),math.radians(lon_meerkat_03),altitude_meerkat);

Tx_pos = AstFn.fnRAZEL_Cartesian(math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),altitude_meerkat,Tx_ecef,0.);
Rx1_pos = AstFn.fnRAZEL_Cartesian(math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),altitude_meerkat,Rx1_ecef,0.);
Rx2_pos = AstFn.fnRAZEL_Cartesian(math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),altitude_meerkat,Rx2_ecef,0.);
Rx3_pos = AstFn.fnRAZEL_Cartesian(math.radians(lat_meerkat_00),math.radians(lon_meerkat_00),altitude_meerkat,Rx3_ecef,0.);
Rx0_pos = np.zeros([3],dtype=np.float64);
# --------------------------------------------------------------------------- #
print 'plot in 3D'
# ---- radar geometry visualization in 3D ----------------------------------- #
# Radar boresight
el_tx_beam_centre= y_sph_tx[1,tx_bw_time_max];
az_tx_beam_centre =  y_sph_tx[2,tx_bw_time_max];

el_rx_beam_centre= y_sph_rx[1,tx_bw_time_max];
az_rx_beam_centre =  y_sph_rx[2,tx_bw_time_max];

el_rx1_beam_centre= y_sph_rx_meerkat_01[1,tx_bw_time_max];
az_rx1_beam_centre =  y_sph_rx_meerkat_01[2,tx_bw_time_max];
# -- draw beams ------------------------------------------------------------- #
del_z_geo = 16;
z_geo_max = 1.6*max(y_sph_tx[0,:].max(),y_sph_rx[0,:].max(),y_sph_rx_meerkat_01[0,:].max());

z_geo = np.arange(0,z_geo_max+del_z_geo,del_z_geo)
del_theta_geo = 0.2;
theta_geo = np.arange(0, 2*math.pi + del_theta_geo,del_theta_geo)

beam_rx_geo = np.zeros([3,np.shape(z_geo)[0],np.shape(theta_geo)[0]],dtype=np.float64) 
beam_tx_geo = np.zeros([3,np.shape(z_geo)[0],np.shape(theta_geo)[0]],dtype=np.float64) 
beam_rx1_geo = np.zeros([3,np.shape(z_geo)[0],np.shape(theta_geo)[0]],dtype=np.float64)
beam_rx2_geo = np.zeros([3,np.shape(z_geo)[0],np.shape(theta_geo)[0]],dtype=np.float64)

for i_z in range(len(z_geo)):
    for i_theta in range(len(theta_geo)):
        beam_rx_geo[0,i_z,i_theta] = z_geo[i_z]*math.tan(beamwidth_rx)*math.cos(theta_geo[i_theta]);
        beam_rx_geo[1,i_z,i_theta] = z_geo[i_z]*math.tan(beamwidth_rx)*math.sin(theta_geo[i_theta]);
        beam_rx_geo[2,i_z,i_theta] = z_geo[i_z];
        beam_rx_geo[:,i_z,i_theta] = np.dot(np.dot(GF.fnRotate3(1.*math.pi - az_rx_beam_centre),GF.fnRotate2(0.5*math.pi-el_rx_beam_centre)),beam_rx_geo[:,i_z,i_theta]);
        
        beam_tx_geo[0,i_z,i_theta] = z_geo[i_z]*math.tan(beamwidth_tx)*math.cos(theta_geo[i_theta]);
        beam_tx_geo[1,i_z,i_theta] = z_geo[i_z]*math.tan(beamwidth_tx)*math.sin(theta_geo[i_theta]);
        beam_tx_geo[2,i_z,i_theta] = z_geo[i_z];
        beam_tx_geo[:,i_z,i_theta] = np.dot(np.dot(GF.fnRotate3(1.*math.pi - az_tx_beam_centre),GF.fnRotate2(0.5*math.pi-el_tx_beam_centre)),beam_tx_geo[:,i_z,i_theta]) + Tx_pos;
        
        beam_rx1_geo[0,i_z,i_theta] = z_geo[i_z]*math.tan(beamwidth_rx)*math.cos(theta_geo[i_theta]);
        beam_rx1_geo[1,i_z,i_theta] = z_geo[i_z]*math.tan(beamwidth_rx)*math.sin(theta_geo[i_theta]);
        beam_rx1_geo[2,i_z,i_theta] = z_geo[i_z];
        beam_rx1_geo[:,i_z,i_theta] = np.dot(np.dot(GF.fnRotate3(1.*math.pi - az_rx1_beam_centre),GF.fnRotate2(0.5*math.pi-el_rx1_beam_centre)),beam_rx1_geo[:,i_z,i_theta]) + Rx1_pos;

        beam_rx2_geo[0,i_z,i_theta] = z_geo[i_z]*math.tan(beamwidth_rx)*math.cos(theta_geo[i_theta]);
        beam_rx2_geo[1,i_z,i_theta] = z_geo[i_z]*math.tan(beamwidth_rx)*math.sin(theta_geo[i_theta]);
        beam_rx2_geo[2,i_z,i_theta] = z_geo[i_z];
        beam_rx2_geo[:,i_z,i_theta] = np.dot(np.dot(GF.fnRotate3(1.*math.pi - az_rx1_beam_centre),GF.fnRotate2(0.5*math.pi-el_rx1_beam_centre)),beam_rx2_geo[:,i_z,i_theta]) + Rx2_pos;

# --  bistatic plane -------------------------------------------------------- # 
plane_normal_vec,plane_d = GF.fnCalculate_Coords_Plane3D(Rx0_pos,x_target[0:3,tx_bw_time_max],Tx_pos);
txrx_vec =Tx_pos-Rx0_pos;
baseline = np.linalg.norm(txrx_vec);
print baseline
normalized_vec = plane_normal_vec/np.linalg.norm(plane_normal_vec);
txrx_vec_normalized = txrx_vec/baseline;
# see testplanetransformation2.py in current_build17
closest_point = GF.fnCalculate_ClosestPointOnLine(Rx0_pos,Tx_pos,x_target[0:3,tx_bw_time_max]);
ortho_vec_in_plane = (x_target[0:3,tx_bw_time_max] - closest_point);
ortho_vec_in_plane_normalized = ortho_vec_in_plane/np.linalg.norm(ortho_vec_in_plane);

mu_b = np.arange(0.,1.+0.1,0.1);
baseline_coords_ortho = np.zeros([3,len(mu_b)],dtype=np.float64);
for i in range(len(mu_b)):
    baseline_coords_ortho[:,i] = GF.fnCalculate_Coords_Line3D(ortho_vec_in_plane,closest_point,mu_b[i]); 

mu_b_closest = (closest_point[1]-Rx0_pos[1])/txrx_vec[1];
mu_b_closest_del =0.1;
mu_b_closest_min = min(mu_b_closest,0);
mu_b_closest_max = max(mu_b_closest,1.+mu_b_closest_del)
mu_b_c= np.arange(mu_b_closest_min,mu_b_closest_max,mu_b_closest_del);
baseline_coords = np.zeros([3,len(mu_b_c)],dtype=np.float64);
for i in range(len(mu_b_c)):
    baseline_coords[:,i] = GF.fnCalculate_Coords_Line3D(txrx_vec,Rx0_pos,mu_b_c[i]);

mu_rx = np.arange(0.,1.0+0.2,0.2);
rho_rx =  np.zeros([3,len(mu_rx)],dtype=np.float64);
rho_rx_vec = x_target[0:3,tx_bw_time_max] - Rx0_pos;
for i in range(len(mu_rx)):
    rho_rx[:,i] = GF.fnCalculate_Coords_Line3D(rho_rx_vec,x_target[0:3,tx_bw_time_max],-mu_rx[i]);

mu_tx = np.arange(0.,1.0+0.2,0.2);
rho_tx =  np.zeros([3,len(mu_tx)],dtype=np.float64);
rho_tx_vec = x_target[0:3,tx_bw_time_max] - Tx_pos;
for i in range(len(mu_rx)):
    rho_tx[:,i] = GF.fnCalculate_Coords_Line3D(rho_tx_vec,x_target[0:3,tx_bw_time_max],-mu_tx[i]);

mu_y = np.arange(-0.2,1.+0.2,0.01);
yaxis = np.zeros([3,len(mu_y)],dtype=np.float64);
yaxis_end = np.zeros([3,len(mu_y)],dtype=np.float64);
for i in range(len(mu_y)):
    yaxis[:,i] = GF.fnCalculate_Coords_Line3D(ortho_vec_in_plane,Rx0_pos,mu_y[i]);
    yaxis_end[:,i] = GF.fnCalculate_Coords_Line3D(ortho_vec_in_plane,closest_point,mu_y[i]);

xaxis = np.zeros_like(baseline_coords);
xaxis_end = np.zeros_like(baseline_coords);
for i in range(0,np.shape(baseline_coords)[1]):
    xaxis[:,i] = GF.fnCalculate_Coords_Line3D(txrx_vec,yaxis[:,0],mu_b_c[i]);
    xaxis_end[:,i] = GF.fnCalculate_Coords_Line3D(txrx_vec,yaxis[:,-1],mu_b_c[i]);
    
mu_rx1 = np.arange(0.,1.0+0.2,0.2);
rho_rx1 =  np.zeros([3,len(mu_rx1)],dtype=np.float64);
rho_rx1_vec = x_target[0:3,tx_bw_time_max] - Rx1_pos;
for i in range(len(mu_rx1)):
    rho_rx1[:,i] = GF.fnCalculate_Coords_Line3D(rho_rx1_vec,x_target[0:3,tx_bw_time_max],-mu_rx1[i]);

mu_rx2 = np.arange(0.,1.0+0.2,0.2);
rho_rx2 =  np.zeros([3,len(mu_rx2)],dtype=np.float64);
rho_rx2_vec = x_target[0:3,tx_bw_time_max] - Rx2_pos;
for i in range(len(mu_rx2)):
    rho_rx2[:,i] = GF.fnCalculate_Coords_Line3D(rho_rx2_vec,x_target[0:3,tx_bw_time_max],-mu_rx2[i]);
   
# --------------------------------------------------------------------------- #
new_x,new_y,new_s =  GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,x_target[0:3,tx_bw_time_max])
new_tgt = np.array([new_x,new_y],dtype=np.float64);

new_x,new_y,new_s =  GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,Tx_pos)
new_tx = np.array([new_x,new_y],dtype=np.float64);
print new_tx

new_x,new_y,new_s =  GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,Rx0_pos)
new_rx = np.array([new_x,new_y],dtype=np.float64);
print new_rx

print 'new baseline'
print np.linalg.norm(new_tx-new_rx);

new_x,new_y,new_s =  GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,Rx1_pos)
new_rx1 = np.array([new_x,new_y],dtype=np.float64);
print new_rx1

new_x,new_y,new_s =  GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,Rx2_pos)
new_rx2 = np.array([new_x,new_y],dtype=np.float64);
print new_rx2

new_x,new_y,new_s =  GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,closest_point)
new_closest_point = np.array([new_x,new_y],dtype=np.float64);

new_xaxis = np.zeros([2,np.shape(baseline_coords)[1]],dtype=np.float64);
new_xaxis_end = np.zeros([2,np.shape(baseline_coords)[1]],dtype=np.float64);
for i in range(0,np.shape(baseline_coords)[1]):
    new_x,new_y,new_s =  GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,xaxis[:,i])
    new_xaxis[:,i] = np.array([new_x,new_y],dtype=np.float64);
    new_x,new_y,new_s =  GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,xaxis_end[:,i])
    new_xaxis_end[:,i] = np.array([new_x,new_y],dtype=np.float64);
    
new_baseline_coords_ortho = np.zeros([2,np.shape(baseline_coords_ortho)[1]],dtype=np.float64);
new_baseline_coords = np.zeros([2,np.shape(baseline_coords)[1]],dtype=np.float64);
for i in range(0,np.shape(baseline_coords_ortho)[1]):
    new_x,new_y,new_s =  GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,baseline_coords_ortho[:,i] )
    new_baseline_coords_ortho[:,i] =  np.array([new_x,new_y],dtype=np.float64);

for i in range(0,np.shape(baseline_coords)[1]): 
    new_x,new_y,new_s =  GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,baseline_coords[:,i] )
    new_baseline_coords[:,i] =  np.array([new_x,new_y],dtype=np.float64);

new_rho_rx = np.zeros([2,len(mu_rx)],dtype=np.float64);
for i in range(len(mu_rx)):
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,rho_rx[:,i]);
    new_rho_rx[:,i] = np.array([new_x,new_y],dtype=np.float64);
    
new_rho_tx = np.zeros([2,len(mu_tx)],dtype=np.float64);
for i in range(len(mu_tx)):
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,rho_tx[:,i]);
    new_rho_tx[:,i] = np.array([new_x,new_y],dtype=np.float64);

new_rho_rx1 = np.zeros([2,len(mu_rx1)],dtype=np.float64);
for i in range(len(mu_rx)):
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,rho_rx1[:,i]);
    new_rho_rx1[:,i] = np.array([new_x,new_y],dtype=np.float64); 

new_rho_rx2 = np.zeros([2,len(mu_rx2)],dtype=np.float64);
for i in range(len(mu_rx)):
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,rho_rx2[:,i]);
    new_rho_rx2[:,i] = np.array([new_x,new_y],dtype=np.float64); 

new_traj_red = np.zeros([2,latest_pt-earliest_pt+1],dtype=np.float64);
for i in range(0,np.shape(new_traj_red)[1]):
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,x_target[0:3,i+earliest_pt]);
    new_traj_red[:,i] = np.array([new_x,new_y],dtype=np.float64);

new_traj_green = np.zeros([2,latest_pt_rx-earliest_pt_rx+1],dtype=np.float64);
for i in range(0,np.shape(new_traj_green)[1]):
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,x_target[0:3,i+earliest_pt_rx]);
    new_traj_green[:,i] = np.array([new_x,new_y],dtype=np.float64);

new_traj_orange = np.zeros([2,latest_pt_rx1-earliest_pt_rx1+1],dtype=np.float64);
for i in range(0,np.shape(new_traj_orange)[1]):
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,x_target[0:3,i+earliest_pt_rx1]);
    new_traj_orange[:,i] = np.array([new_x,new_y],dtype=np.float64);

new_traj_purple = np.zeros([2,latest_pt_rx2-earliest_pt_rx2+1],dtype=np.float64);
for i in range(0,np.shape(new_traj_purple)[1]):
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,x_target[0:3,i+earliest_pt_rx2]);
    new_traj_purple[:,i] = np.array([new_x,new_y],dtype=np.float64);

#new_traj_blue_down = np.zeros([2,earliest_pt-plt_start_index+1],dtype=np.float64);
#for i in range(0,np.shape(new_traj_blue_down)[1]):
#    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,x_target[0:3,i+plt_start_index]);
#    new_traj_blue_down[:,i] = np.array([new_x,new_y],dtype=np.float64);  

new_traj_blue_up = np.zeros([2,plt_end_index-plt_start_index+1],dtype=np.float64);
for i in range(0,np.shape(new_traj_blue_up)[1]):
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,x_target_02[0:3,i+plt_start_index]);
    new_traj_blue_up[:,i] = np.array([new_x,new_y],dtype=np.float64);    

new_beam_rx_geo = np.zeros([2,np.shape(z_geo)[0]],dtype=np.float64) 
new_beam_tx_geo = np.zeros([2,np.shape(z_geo)[0]],dtype=np.float64) 
new_beam_rx_geo_down = np.zeros([2,np.shape(z_geo)[0]],dtype=np.float64) 
new_beam_tx_geo_down = np.zeros([2,np.shape(z_geo)[0]],dtype=np.float64) 
beam_flip = len(theta_geo)/2;
new_beam_rx1_geo = np.zeros([2,np.shape(z_geo)[0]],dtype=np.float64) 
new_beam_rx1_geo_down = np.zeros([2,np.shape(z_geo)[0]],dtype=np.float64);
new_beam_rx2_geo = np.zeros([2,np.shape(z_geo)[0]],dtype=np.float64) 
new_beam_rx2_geo_down = np.zeros([2,np.shape(z_geo)[0]],dtype=np.float64);

for i_z in range(len(z_geo)):
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,beam_rx_geo[:,i_z,0]);
    new_beam_rx_geo[:,i_z] = np.array([new_x,new_y],dtype=np.float64);
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,beam_tx_geo[:,i_z,0]);
    new_beam_tx_geo[:,i_z] = np.array([new_x,new_y],dtype=np.float64);  
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,beam_rx1_geo[:,i_z,0]);
    new_beam_rx1_geo[:,i_z] = np.array([new_x,new_y],dtype=np.float64);  
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,beam_rx2_geo[:,i_z,0]);
    new_beam_rx2_geo[:,i_z] = np.array([new_x,new_y],dtype=np.float64); 
    
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,beam_rx_geo[:,i_z,beam_flip]);
    new_beam_rx_geo_down[:,i_z] = np.array([new_x,new_y],dtype=np.float64);
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,beam_tx_geo[:,i_z,beam_flip]);
    new_beam_tx_geo_down[:,i_z] = np.array([new_x,new_y],dtype=np.float64); 
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,beam_rx1_geo[:,i_z,beam_flip]);
    new_beam_rx1_geo_down[:,i_z] = np.array([new_x,new_y],dtype=np.float64);
    new_x,new_y,new_s = GF.fnCalculate_Projection_3D_2D(normalized_vec,txrx_vec_normalized,ortho_vec_in_plane_normalized,beam_rx2_geo[:,i_z,beam_flip]);
    new_beam_rx2_geo_down[:,i_z] = np.array([new_x,new_y],dtype=np.float64);
# -------------------------------------------------------------------------- #

fig = plt.figure(1);
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
ax = fig.gca()
#fig.suptitle(r"\textbf{Bistatic plane at the best dwell-time for object %s trajectory during %s}" %(norad_id,title_string),fontsize=12)
fig.suptitle(r"\textbf{Bistatic plane for object %s transit during %s}" %(norad_id,title_string),fontsize=12)

plt.plot(new_rx[0],new_rx[1],marker=r'$\bigcirc$',c='b');
plt.annotate(r'Rx0',(new_rx[0],new_rx[1]-60),color='blue')
plt.plot(new_tx[0],new_tx[1],marker='o',c='g');
plt.annotate(r'Tx',(new_tx[0],new_tx[1]-60))

plt.annotate(
    r'\textbf{Target trajectory}', xy=(new_traj_blue_up[0,0],new_traj_blue_up[1,0] ), xycoords='data',
    xytext=(-1700,750), textcoords='data',
    arrowprops={'arrowstyle': '->'})

plt.plot(new_rx1[0],new_rx1[1],marker=r'$\times$',c='orangered');
plt.annotate(r'Rx1',(new_rx1[0]+20,new_rx1[1]+20),color='orangered');
plt.plot(new_rx2[0],new_rx2[1],marker=r'$\times$',c='purple');
plt.annotate(r'Rx2',(new_rx2[0],new_rx2[1]-120),color='purple')

plt.annotate(r"\textbf{Dwell-time in Tx beam} $= %.6f~\mathrm{s}$" %((latest_pt-earliest_pt)*delta_t),(-1500,1100) );
plt.annotate(r"\textbf{Dwell-time in Rx0 beam} $= %.6f~\mathrm{s}$" %((latest_pt_rx-earliest_pt_rx)*delta_t),(-1500,1050) );
plt.annotate(r"\textbf{Dwell-time in Rx1 beam} $= %.6f~\mathrm{s}$" %((latest_pt_rx1-earliest_pt_rx1)*delta_t),(-1500,1000) );
plt.annotate(r"\textbf{Dwell-time in Rx2 beam} $= %.6f~\mathrm{s}$" %((latest_pt_rx2-earliest_pt_rx2)*delta_t),(-1500,950) );

plt.annotate(r"\textbf{Beam centre pointing}",(-450,825) );
plt.text(-720,475,r"\begin{tabular}{|c | c | c |}  \hline & $\theta$ & $\psi$ \\ \hline \hline \textbf{Tx} & $%.6f\mathrm{^\circ}$ & $%.6f\mathrm{^\circ}$ \\ \hline \textbf{Rx0} & $%.6f\mathrm{^\circ}$ & $%.6f\mathrm{^\circ}$ \\ \hline \textbf{Rx1} & $%.6f\mathrm{^\circ}$ & $%.6f\mathrm{^\circ}$ \\ \hline \textbf{Rx2} & $%.6f\mathrm{^\circ}$ & $%.6f\mathrm{^\circ}$ \\ \hline \end{tabular}" %(math.degrees(y_sph_tx[1,tx_bw_time_max]),math.degrees(y_sph_tx[2,tx_bw_time_max]),math.degrees(y_sph_rx[1,index_for_rx0]),math.degrees(y_sph_rx[2,index_for_rx0]),math.degrees(y_sph_rx_meerkat_01[1,index_for_rx1]),math.degrees(y_sph_rx_meerkat_01[2,index_for_rx1]),math.degrees(y_sph_rx_meerkat_02[1,index_for_rx2]),math.degrees(y_sph_rx_meerkat_02[2,index_for_rx2])),size=12)

plt.annotate(r"$\Theta_{3~\mathrm{dB}}^{\text{Rx}} = %.3f \mathrm{^\circ}$" %math.degrees(beamwidth_rx),(-2300,900))
plt.annotate(r"$\Theta_{3~\mathrm{dB}}^{\text{Tx}} = %.3f \mathrm{^\circ}$" %math.degrees(beamwidth_tx),(-2400,500))

plt.plot(new_baseline_coords_ortho[0,:],new_baseline_coords_ortho[1,:],color='black')
plt.plot(new_baseline_coords[0,:],new_baseline_coords[1,:],color='black')

plt.plot(new_rho_tx[0,:],new_rho_tx[1,:],color='darkgreen')
plt.plot(new_rho_rx[0,:],new_rho_rx[1,:],color='midnightblue')
plt.plot(new_rho_rx1[0,:],new_rho_rx1[1,:],color='orangered')

plt.plot(new_traj_red[0,:],new_traj_red[1,:],color='darkgreen',linewidth=2,label=r"%s" %str(earliest_pt_epoch.isoformat())+'Z/'+str(latest_pt_epoch.isoformat())+'Z')
plt.plot(new_traj_green[0,:],new_traj_green[1,:],color='blue',linewidth=3,label=r"%s" %str(earliest_pt_rx_epoch.isoformat())+'Z/'+str(latest_pt_rx_epoch.isoformat())+'Z')
plt.plot(new_traj_orange[0,:],new_traj_orange[1,:],color='orangered',linewidth=3,label=r"%s" %str(earliest_pt_rx1_epoch.isoformat())+'Z/'+str(latest_pt_rx1_epoch.isoformat())+'Z')
plt.plot(new_traj_purple[0,:],new_traj_purple[1,:],color='purple',linewidth=3,label=r"%s" %str(earliest_pt_rx2_epoch.isoformat())+'Z/'+str(latest_pt_rx2_epoch.isoformat())+'Z')

#plt.plot(new_traj_blue_down[0,0:-1:1000],new_traj_blue_down[1,0:-1:1000],linewidth=0.7,color='darkslategray',linestyle='dashed')
plt.plot(new_traj_blue_up[0,0:120000:10000],new_traj_blue_up[1,0:120000:10000],linewidth=0.7,color='darkslategray')#,linestyle='dashed')

ax.fill_between(new_beam_rx_geo_down[0,:],new_beam_rx_geo_down[1,:],new_beam_rx_geo[1,:],facecolor='mediumblue',alpha=0.25);
ax.fill_between(new_beam_tx_geo_down[0,:],new_beam_tx_geo_down[1,:],new_beam_tx_geo[1,:],facecolor='green',alpha=0.25);
ax.fill_between(new_beam_rx1_geo_down[0,:],new_beam_rx1_geo_down[1,:],new_beam_rx1_geo[1,:],facecolor='orangered',alpha=0.25);
ax.fill_between(new_beam_rx2_geo_down[0,:],new_beam_rx2_geo_down[1,:],new_beam_rx2_geo[1,:],facecolor='purple',alpha=0.2);

plt.legend(loc='upper center',title=r"\textbf{Dwell-time interval in Tx, Rx0, Rx1 \& Rx2 beams}",bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True)

plt.xlim(-3000,1000);
plt.ylim(-200,1200);
ax.set_xlabel(r"$u_b~[\mathrm{km}]$")
ax.set_ylabel(r"$v_b~[\mathrm{km}]$"); 
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
#plt.minorticks_on()
fig.savefig('main_057_iss_11_1_bistaticplane0.pdf',bbox_inches='tight',pad_inches=0.11,dpi=10) 

# --------------------------------------------------------------------------- #
fig = plt.figure(2);
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
params = {'legend.fontsize': 8,
    'legend.handlelength': 2}
plt.rcParams.update(params);
ax = fig.gca(projection='3d');
ax.set_aspect('equal')
fig.suptitle(r"\textbf{Observation geometry for object %s trajectory in the local frame during %s}" %(norad_id,title_string),fontsize=10)
ax.plot(x_target_02[0,0:120000:10000],x_target_02[1,0:120000:10000],x_target[2,0:120000:10000],color='blue',linewidth=1)
#ax.plot(x_target[0,latest_pt+1:plt_end_index],x_target[1,latest_pt+1:plt_end_index],x_target[2,latest_pt+1:plt_end_index],color='blue',linewidth=1)
ax.plot(x_target[0,earliest_pt:latest_pt+1],x_target[1,earliest_pt:latest_pt+1],x_target[2,earliest_pt:latest_pt+1],color='crimson',linewidth=2,label=r"%s" %str(earliest_pt_epoch.isoformat())+'Z/'+str(latest_pt_epoch.isoformat())+'Z');
ax.legend();
ax.scatter(0,0,0,c='blue',marker='o');
ax.text(0+10,0-20,0-190,r'Rx',color='k')
ax.text(Tx_pos[0]-10,Tx_pos[1]+40,Tx_pos[2]-190,r'Tx',color='k')
ax.scatter(Tx_pos[0],Tx_pos[1],Tx_pos[2],c='darkgreen',marker='o');

for i_z in range(0,len(z_geo)):
    ax.plot(beam_rx_geo[0,i_z,:],beam_rx_geo[1,i_z,:],beam_rx_geo[2,i_z],'mediumslateblue',alpha=0.16);
    ax.plot(beam_tx_geo[0,i_z,:],beam_tx_geo[1,i_z,:],beam_tx_geo[2,i_z],'green',alpha=0.16);

## Bounding box for equal axes 
maxx = max(x_target_02[0,0:-1:10000].max(),Tx_pos[0],beam_tx_geo[0,:,:].max(),beam_rx_geo[0,:,:].max());
maxy = max(x_target_02[1,0:-1:10000].max(),Tx_pos[1],beam_tx_geo[1,:,:].max(),beam_rx_geo[1,:,:].max());
maxz = max(x_target[2,0:-1:10000].max(),Tx_pos[2],beam_tx_geo[2,:,:].max(),beam_rx_geo[2,:,:].max());
minx = min(x_target_02[0,0:-1:10000].min(),Tx_pos[0],beam_tx_geo[0,:,:].min(),beam_rx_geo[0,:,:].min());
miny = min(x_target_02[1,0:-1:10000].min(),Tx_pos[1],beam_tx_geo[1,:,:].min(),beam_rx_geo[1,:,:].min());
minz = min(x_target[2,0:-1:10000].min(),Tx_pos[2],beam_tx_geo[2,:,:].min(),beam_rx_geo[2,:,:].min());

max_range = np.array([maxx - minx,maxy - miny,maxz - minz]).max() / 2.0
mid_x = (maxx + minx) * 0.5
mid_y = (maxy + miny) * 0.5
mid_z = (maxz + minz) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.view_init(elev=5,azim=30)

ax.set_xlabel(r'$x~[\mathrm{km}]$')
ax.set_ylabel(r'$y~[\mathrm{km}]$')
ax.set_zlabel(r'$z~[\mathrm{km}]$',rotation=90)
fig.savefig('main_057_iss_11_1_geometry0.pdf',format='pdf',bbox_inches='tight',pad_inches=0.08,dpi=10);


# ------------------------------------------------------------------------- #
boundx = np.zeros([np.shape(baseline_coords)[1],len(mu_y)],dtype=np.float64);
boundy = np.zeros([np.shape(baseline_coords)[1],len(mu_y)],dtype=np.float64);
boundz = np.zeros([np.shape(baseline_coords)[1],len(mu_y)],dtype=np.float64);

for i_x in range(0,np.shape(baseline_coords)[1]):
    for i_y in range(len(mu_y)):
        point = GF.fnCalculate_Coords_Line3D(ortho_vec_in_plane,baseline_coords[:,i_x],mu_y[i_y]);  
        boundx[i_x,i_y] = point[0];
        boundy[i_x,i_y] = point[1];
        boundz[i_x,i_y] = point[2];
        # Proper way of creating points on bistatic grid


fig = plt.figure(3);
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
params = {'legend.fontsize': 8,
    'legend.handlelength': 2}
plt.rcParams.update(params);
ax = fig.gca(projection='3d');
ax.set_aspect('equal')
fig.suptitle(r"\textbf{Bistatic geometry for object 25544 trajectory in the local frame}",fontsize=12)
#ax.plot(x_target[0,plt_start_index:earliest_pt],x_target[1,plt_start_index:earliest_pt],x_target[2,plt_start_index:earliest_pt],color='blue',linewidth=1)
#ax.plot(x_target[0,latest_pt+1:plt_end_index],x_target[1,latest_pt+1:plt_end_index],x_target[2,latest_pt+1:plt_end_index],color='blue',linewidth=1)
#ax.plot(x_target[0,earliest_pt:latest_pt+1],x_target[1,earliest_pt:latest_pt+1],x_target[2,earliest_pt:latest_pt+1],color='crimson',linewidth=2,label=r"%s" %str(earliest_pt_epoch.isoformat())+'/'+str(latest_pt_epoch.isoformat()));

ax.plot(x_target_02[0,0:120000:10000],x_target_02[1,0:120000:10000],x_target[2,0:120000:10000],color='blue',linewidth=1)
ax.plot(x_target[0,earliest_pt:latest_pt+1],x_target[1,earliest_pt:latest_pt+1],x_target[2,earliest_pt:latest_pt+1],color='crimson',linewidth=2,label=r"%s" %str(earliest_pt_epoch.isoformat())+'Z/'+str(latest_pt_epoch.isoformat())+'Z');

ax.legend();
ax.scatter(0,0,0,c='blue',marker='o');
ax.text(0+10,0+20,0-190,r'Rx',color='k')
ax.text(Tx_pos[0],Tx_pos[1]-40,Tx_pos[2]-190,r'Tx',color='k')
ax.scatter(Tx_pos[0],Tx_pos[1],Tx_pos[2],c='darkgreen',marker='o');
ax.scatter(x_target[0,tx_bw_time_max],x_target[1,tx_bw_time_max],x_target[2,tx_bw_time_max],c='red',marker='o');

# -- bisplane --# 
ax.plot(baseline_coords[0,:],baseline_coords[1,:],baseline_coords[2,:],'black');
ax.plot(baseline_coords_ortho[0,:],baseline_coords_ortho[1,:],baseline_coords_ortho[2,:],'black');
ax.plot(rho_rx[0,:],rho_rx[1,:],rho_rx[2,:],'midnightblue');
ax.plot(rho_tx[0,:],rho_tx[1,:],rho_tx[2,:],'darkgreen');

ax.plot_surface(boundx,boundy,boundz,color='gray',alpha=0.2,linewidth=0,zorder=1);

# -- bisplane --# 
## Bounding box for equal axes # 9 May 2017: need to account for rx tx also in these calculations
maxx = max(x_target[0,plt_start_index:plt_end_index].max(),Tx_pos[0],beam_tx_geo[0,:,:].max(),beam_rx_geo[0,:,:].max(),baseline_coords[0,:].max());
maxy = max(x_target[1,plt_start_index:plt_end_index].max(),Tx_pos[1],beam_tx_geo[1,:,:].max(),beam_rx_geo[1,:,:].max(),baseline_coords[1,:].max());
maxz = max(x_target[2,plt_start_index:plt_end_index].max(),Tx_pos[2],beam_tx_geo[2,:,:].max(),beam_rx_geo[2,:,:].max(),baseline_coords[2,:].max());
minx = min(x_target[0,plt_start_index:plt_end_index].min(),Tx_pos[0],beam_tx_geo[0,:,:].min(),beam_rx_geo[0,:,:].min(),baseline_coords[0,:].min());
miny = min(x_target[1,plt_start_index:plt_end_index].min(),Tx_pos[1],beam_tx_geo[1,:,:].min(),beam_rx_geo[1,:,:].min(),baseline_coords[1,:].min());
minz = min(x_target[2,plt_start_index:plt_end_index].min(),Tx_pos[2],beam_tx_geo[2,:,:].min(),beam_rx_geo[2,:,:].min(),baseline_coords[2,:].min());

max_range = np.array([maxx - minx,maxy - miny,maxz - minz]).max() / 2.0
mid_x = (maxx + minx) * 0.5
mid_y = (maxy + miny) * 0.5
mid_z = (maxz + minz) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.view_init(elev=15,azim=60)
ax.set_xlabel(r'$x~[\mathrm{km}]$')
ax.set_ylabel(r'$y~[\mathrm{km}]$')
ax.set_zlabel(r'$z~[\mathrm{km}]$',rotation=90)
fig.savefig('main_057_iss_11_1_geometry_bistatic0.pdf',format='pdf',bbox_inches='tight',pad_inches=0.08,dpi=10);



fig = plt.figure(5);
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
ax = fig.gca()
fig.suptitle(r"\textbf{Projection of the target-radar geometry onto the bistatic plane at the best dwell-time}" ,fontsize=12)

plt.plot(new_rx[0],new_rx[1],marker='o',c='b');
plt.annotate(r'Rx',(new_rx[0],new_rx[1]-50))
plt.plot(new_tx[0],new_tx[1],marker='o',c='g');
plt.annotate(r'Tx',(new_tx[0],new_tx[1]-50));

bistatic_baseline = np.linalg.norm(new_rx - new_tx);
loww=-100
label_down = np.arange(loww,new_rx[1]+10,10,dtype=np.float64)
label_down_y = new_rx[0]*np.ones(len(label_down),dtype=np.float64);
label_up_y = new_tx[0]*np.ones(len(label_down),dtype=np.float64);

plt.annotate(
    '', xy=(new_rx[0],loww ), xycoords='data',
    xytext=(new_tx[0],loww), textcoords='data',
    arrowprops={'arrowstyle': '<->'})
plt.text((new_rx[0]),loww-50, r"$L_b =%.3f ~\mathrm{km}$" %bistatic_baseline)
plt.plot(label_down_y,label_down,color='darkslategray',linestyle='dotted')
plt.plot(label_up_y,label_down,color='darkslategray',linestyle='dotted')

plt.plot(new_tgt[0],new_tgt[1],marker='o',c='r');
#plt.annotate(r'SO',(new_tgt[0],new_tgt[1]+50));
plt.plot(new_baseline_coords_ortho[0,:],new_baseline_coords_ortho[1,:],color='black')
plt.plot(new_baseline_coords[0,:],new_baseline_coords[1,:],color='black')

ax.fill_between(new_xaxis[0,:],new_xaxis[-1,-1],new_xaxis_end[-1,-1],facecolor='gray',alpha=0.5);

plt.annotate(
    r'\textbf{Target trajectory}', xy=(new_traj_blue_up[0,0],new_traj_blue_up[1,0] ), xycoords='data',
    xytext=(-1700,750), textcoords='data',
    arrowprops={'arrowstyle': '->'})

plt.plot(new_rho_tx[0,:],new_rho_tx[1,:],color='darkgreen')
plt.plot(new_rho_rx[0,:],new_rho_rx[1,:],color='midnightblue')
plt.plot(new_traj_blue_up[0,0:120000:10000],new_traj_blue_up[1,0:120000:10000],linewidth=0.7,color='darkslategray')#,linestyle='dashed')

plt.xlim(-2000,1000);
plt.ylim(-200,900);

#plt.legend(loc=4)
ax.set_xlabel(r"$u_b~[\mathrm{km}]$")
ax.set_ylabel(r"$v_b~[\mathrm{km}]$"); 
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
#plt.minorticks_on()
fig.savefig('main_057_iss_11_1_bistaticplane_proj.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10)  