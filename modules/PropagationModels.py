# -*- coding: utf-8 -*-
"""
Module which implements propagation modelling functions.

Created on Sun Feb 26 19:56:10 2017

@author: Ashiv Dhondea


Edits:
26/02/17: created file and added the function fnCalculate_LinkTime
26/02/17: created the function fnCalculate_DownlinkTime_Iter
02/03/17: included the module MathsFn which is called by fnCalculate_DownlinkTime_Iter
02/03/17: added the function fnCalculate_UplinkTime_Iter
24/03/17: fixed the function fnCalculate_UplinkTime_Iter
04/04/17: created the function fnCalculate_RangeRate
19/04/17: fixed a typo in fnCalculate_RangeRate
19/04/17: created the function fnCalculate_Bistatic_Range
19/04/17: Created two versions of fnCalculate_RangeRate: one for bistatic range rate and one for monostatic range rate

Reference:
1. Satellite Orbits: Models, Methods, Applications. Montenbruck, Gill. 2000. Section 6.2.2
"""

import numpy as np
import MathsFunctions as MathFn
# --------------------------------------------------------------------------- #
def fnCalculate_LinkTime(target_pos,radar_pos,speed_light):
    """
    Calculates the signal propagation time over a link.
    
    target_pos = position of target 
    radar_pos = position of radar
    speed_light = speed of light in the medium of interest
    
    Called by fnCalculate_DownlinkTime_Iter 
    
    Created: 25 February 2017
    """
    dvec = target_pos - radar_pos;
    tau = (1./speed_light)*np.linalg.norm(dvec);
    return tau # validated. lighttime02.py
    
def fnCalculate_DownlinkTime_Iter(target_pos,radar_pos,timevec,index,speed_light,errTol):
    """
    Iterative procedure to find the downlink light-time for a particular signal 
    received at the ground station.
    
    target_pos = position vectors of the target at all time stamps.
    radar_pos = position vectors of the ground station at all time stamps.
    timevec = vector of time variable.
    index = time index for which the downlink time is to be found.
    speed_light = speed of light in the medium of interest.
    errTol = error tolerance is the evaluated light-time.
    
    tau_i = light-time in seconds for the downlink.
    time_idx = time index at which light was emitted
    time_el = light-time in seconds for the downlink according to our timevec
    Calls fnCalculate_LinkTime  
    
    Created: 26 February 2017
    """
    tau_i = 0.;
    for i in range(len(timevec)):
        tau_0 = tau_i;
        # Find position of the target and the station at the moment of interest.        
        tgt_pos = target_pos[:,index - i];
        site_pos = radar_pos[:,index];
        # Evaluate the downlink time.
        tau_i = fnCalculate_LinkTime(tgt_pos,site_pos,speed_light);
        if abs(tau_i - tau_0) < errTol: # if within the tolerance limits, stop the iteration.
            #~ print 'time of transmission of pulse'
            #~ print timevec[index] - tau_i
            time_el,time_idx = MathFn.find_nearest(timevec,timevec[index] - tau_i)
            #print time_idx
            #~ print 'light-time approx in [s] '
            #~ print tau_i
            break
    return tau_i,time_el,time_idx # validated. 02.03.17 in lighttime03.py

def fnCalculate_UplinkTime_Iter(target_pos,radar_pos,timevec,index,speed_light,errTol,index_d,tau_d):
    """
    Iterative procedure to find the uplink light-time for a particular signal 
    received at the ground station.
    
    target_pos = position vectors of the target at all time stamps.
    radar_pos = position vectors of the ground station at all time stamps.
    timevec = vector of time variable.
    index = time index for which the uplink time is to be found.
    speed_light = speed of light in the medium of interest.
    errTol = error tolerance is the evaluated light-time.
    index_d = the downlink time index
    tau_d = the downlink time
    
    tau_i = light-time in seconds for the uplink.
    
    Calls fnCalculate_LinkTime  
    
    Created: 01 March 2017
    Edited: 
    02.03.17: fixed a mistake in the expression for the time index of the uplink time.
    02.03.17: fixed initialization of tau_i
    24/03/17: fixed the code. validation in testlighttime_01.py
    """
    tau_i = tau_d;
    for i in range(0,len(timevec)-index_d+index): 
#        print 'i'
#        print i
        tau_0 = tau_i;
#        print 'tau_0'
#        print tau_0
        # Find position of the target and the station at the moment of interest.        
        tgt_pos = target_pos[:,index ];
        site_pos = radar_pos[:,index - i]; # 24/03/17: see proof of this being right in testlighttime_01.py
#        print index
#        print index_d
        # Evaluate the uplink time.
        tau_i = fnCalculate_LinkTime(tgt_pos,site_pos,speed_light);
#        print 'tau_i'
#        print tau_i
        if abs(tau_i - tau_0) < errTol: # if within the tolerance limits, stop the iteration.
#            print 'uplink time found'
#            print 'time of transmission of pulse'
#            print timevec[index] - tau_i
            time_el,time_idx = MathFn.find_nearest(timevec,timevec[index] - tau_d - tau_i) #24/03/17
#            print time_idx
#            print 'light-time approx in [s] '
#            print tau_i
            break
    return tau_i,time_idx # the uplink time and the time index for it.

# --------------------------------------------------------------------------------------------------------------- #
def fnCalculate_Bistatic_RangeRate(speed_light,tau_u1,tau_d1,tau_u2,tau_d2,tc):
	"""
	Calculate the average range rate. eqn 6.37 in Montenbruck 2000.
	tc =  length of integration interval, i.e. length of CPI
	Created: 04/04/17
	"""
	range_rate = (speed_light/tc)*(tau_u2+tau_d2-tau_u1-tau_d1); # removed 0.5 factor. 19.04.17
	return range_rate

def fnCalculate_Bistatic_Range(speed_light,tau_d,tau_u):
    """
	Calculate the bistatic range. 
	Created: 19/04/17
    """
    tau = np.add(tau_d,tau_u);
    actual_range = speed_light*tau;
    return actual_range
	
def fnCalculate_Monostatic_RangeRate(speed_light,tau_u1,tau_d1,tau_u2,tau_d2,tc):
	"""
	Calculate the average range rate. eqn 6.37 in Montenbruck 2000.
	tc =  length of integration interval, i.e. length of CPI
	Created: 19/04/17
	"""
	range_rate = (0.5*speed_light/tc)*(tau_u2+tau_d2-tau_u1-tau_d1); 
	return range_rate

def fnCalculate_Monostatic_Range(speed_light,tau_d,tau_u):
    """
	Calculate the monostatic range. 
	Created: 19/04/17
    """
    tau = np.add(tau_d,tau_u);
    actual_range = 0.5*speed_light*tau;
    return actual_range