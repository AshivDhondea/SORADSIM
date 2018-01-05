# -*- coding: utf-8 -*-
"""
RadarSystem

Functions which implement various radar engineering functions 
relevant to the space debris detection and tracking project.

Created on Fri May 26 15:04:53 2017
Edited:
31/05/17: added the functions fnCalculate_AntennaGain & fnCalculate_AntennaBeamwidth
31/05/17: added the functions fnCalculate_MaxUnamRange, fn_Power_to_dB, fn_dB_to_Power
02/06/17: added the functions fnCalculate_RadarHorizon & fnCalculate_MinTargetHeight
06/05/17: edited the function fn_Power_to_dB
19/06/17: added the function fnCalculate_MaxUnamRangeRate
08/07/17: added the function fnCalculate_Beamshape_Loss 

@author: Ashiv Dhondea
"""
import math
import AstroConstants as AstCnst

import numpy as np
# --------------------------------------------------------------------------- #
def fnCalculate_ReceivedPower(P_Tx,G_Tx,G_Rx,rho_Rx,rho_Tx,wavelength,RCS):
    """
    Calculate the received power at the bistatic radar receiver.
    
    equation 5 in " PERFORMANCE ASSESSMENT OF THE MULTIBEAM RADAR
    SENSOR BIRALES FOR SPACE SURVEILLANCE AND TRACKING"

    Note: ensure that the distances rho_Rx,rho_Tx,wavelength are converted to
    metres before passing into this function.
    
    Created on: 26 May 2017
    """
    denominator = (4*math.pi)**3 * (rho_Rx**2)*(rho_Tx**2);
    numerator = P_Tx*G_Tx*G_Rx*RCS*(wavelength**2);
    P_Rx = numerator/denominator;
    return P_Rx

def fnCalculate_ReceivedSNR(P_Rx,T0,bandwidth):
    """
    Calculate the SNR at the bistatic radar receiver.
    
    equation 7 in " PERFORMANCE ASSESSMENT OF THE MULTIBEAM RADAR
    SENSOR BIRALES FOR SPACE SURVEILLANCE AND TRACKING"
    
    Note: output is not in decibels.
    
    Created on: 26 May 2017
    """
    k_B = AstCnst.boltzmann_constant;
    snr = P_Rx/(k_B*bandwidth*T0);
    return snr
# --------------------------------------------------------------------------- #
def fnCalculate_AntennaGain(radius,wavelength,antenna_efficiency):
	"""
	Calculate the gain of a parabolic antenna
	Theory:
	G is the gain over an isotropic source in dB
    k is the efficiency factor which is generally around 50% to 60%, i.e. 0.5 to 0.6
    D is the diameter of the parabolic reflector in metres
    lambda is the wavelength of the signal in metres
	
	Note:
	ensure radius, wavelength are in [m], not [km].
		
	Created on: 31 May 2017
	"""
	gain = (math.pi*2*radius/wavelength)**2 * antenna_efficiency; # not in [dB]!!
	return gain # validated in testradareqn.py

def fnCalculate_AntennaBeamwidth(k,wavelength,radius):
	"""
	Calculate the Half Power Beam Width for a parabolic antenna
	k == 57.3 usually
	
	Output is in [deg] not [rad]
	
	Created: 31 May 2017
	"""
	return (k*wavelength/(2.*radius)) 
# --------------------------------------------------------------------------- #
def fnCalculate_MaxUnamRange(pulse_repetition_time,pulse_width):
    """
    Calculate the maximum unambiguous range
    
    Created: 31 May 2017 [ported from MATLAB code from 2016]
    """
    return AstCnst.c*(pulse_repetition_time - pulse_width)/2.0;
    
def fn_Power_to_dB(p):
    return 10*np.log10(p); # 06/05/17: changed from math to numpy

def fn_dB_to_Power(dB):
    return 10**(0.1*dB)
# --------------------------------------------------------------------------- #
def fnCalculate_RadarHorizon(Re,H):
    """
    Calculate the radar horizon.
    Ref: Wikipedia.
    Does not take into account refraction through the atmosphere
    Date: 20 March 2017
    """
    Dh = math.sqrt(2.*H*Re+H**2);
    return Dh
    
def fnCalculate_MinTargetHeight(target_range,Re,H):
    """
    Calculate the minimum target height that is visible by the radar.
    Ref: Wikipedia
    Objects below this height are in the radar shadow.    
    """
    min_target_height = (target_range - math.sqrt(2.*H*Re))**2/(2.*Re);
    return min_target_height
# --------------------------------------------------------------------------- #
def fnCalculate_MaxUnamRangeRate(prf,centre_frequency):
	"""
	Calculate the maximum unambiguous range rate in the case of unknown direction
	
	Date: 19 June 2017
	"""    
	return AstCnst.c*prf/(4.*centre_frequency);
# --------------------------------------------------------------------------- #
def fnCalculate_Beamshape_Loss(beamwidth_ra,beamwidth_dec,delta_ra,delta_dec,gain_rx_dB):
    """
    Calculate the receiver gain in dB which takes into consideration the elliptical
    model of the beam from "PERFORMANCE ASSESSMENT OF THE MULTIBEAM RADAR
    SENSOR BIRALES FOR SPACE SURVEILLANCE AND TRACKING"
    Equation 6
    Created on: 8 July 2017
    """
    gain_rx_radec_dB = gain_rx_dB - 12.*((delta_ra/beamwidth_ra)**2 + (delta_dec/beamwidth_dec)**2);
    return gain_rx_radec_dB