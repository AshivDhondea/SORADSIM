# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:04:37 2017

Checks if a satellite will be visible

@author: Ashiv Dhondea
"""
import numpy as np
import math

import datetime as dt
import pytz
import aniso8601

# Module for SGP4 orbit propagation
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

# My libraries
import AstroFunctions as AstFn
import GeometryFunctions as GF
import TimeHandlingFunctions as THF
import UnbiasedConvertedMeasurements as UCM
# --------------------------------------------------------------------------- #
def fnCheck_satellite_visibility(satellite_obj,lat_station,lon_station,altitude_station,timestamp_tle_epoch,delta_t,duration,min_el):
    """
    Check whether satellite will be visible from a given ground station, 
    during a given period of time.
    
    if start and end of visibility window are returned as 0, then satellite
    will not be visible from the ground station at any point during the implied
    time period.

    satellite_obj: SGP4 python satellite object, created from the TLE set.
    lat_station, lon_station, altitude_station  : latitude, longitude & 
    altitude in km of ground station.
    
    timestamp_tle_epoch: datetime object for the TLE epoch
    
    delta_t : time step in seconds for simulation
    duration: duration in seconds for simulation period
    
    min_el : minimum elevation in radians for visibility
    
    Created: 28.08.17
    """
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

    index = 0;    
    tle_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec, index, timestamp_tle_epoch)  
    # timestamps for this experiment
    experiment_timestamps = [None]*len(timevec);
    experiment_timestamps[index] = timestamp_tle_epoch.isoformat() + 'Z'; 
    satpos,satvel = satellite_obj.propagate(tle_epoch_test.year,tle_epoch_test.month,tle_epoch_test.day,tle_epoch_test.hour,tle_epoch_test.minute,tle_epoch_test.second+(1e-6)*tle_epoch_test.microsecond);
    x_state_sgp4[0:3,index] = np.asarray(satpos);
    x_state_sgp4[3:6,index] = np.asarray(satvel);
           
    # Rotate ECI position vector by GMST angle to get ECEF position
    theta_GMST[index] = GF.fnZeroTo2Pi(math.radians(THF.fn_Convert_Datetime_to_GMST(timestamp_tle_epoch)));
    xecef_sgp4[:,index] = AstFn.fnECItoECEF(x_state_sgp4[0:3,index],theta_GMST[index]);
    lat_sgp4[index],lon_sgp4[index] = AstFn.fnCarts_to_LatLon(xecef_sgp4[:,index]);

    for index in range(1,len(timevec)):
        # Find the current timestamp as a datetime object
        tle_epoch_test = THF.fnCalculate_DatetimeEpoch(timevec, index, timestamp_tle_epoch)            
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
        
    el_range_above_horizon   = np.where( y_sph_rx[1,:] >= min_el );
    try:
        # index for start and end of visibility region
        start_vis_region = el_range_above_horizon[0][0]
    except IndexError:
        start_vis_region=0
    try:
        end_vis_region = el_range_above_horizon[0][-1];
    except IndexError:
        end_vis_region=0      
    
    return start_vis_region,end_vis_region,timevec,y_sph_rx,experiment_timestamps,lat_sgp4,lon_sgp4
      