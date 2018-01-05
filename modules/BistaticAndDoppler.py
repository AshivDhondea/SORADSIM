"""
Bistatic and Doppler-related functions

Author: AshivD, RRSG, UCT.
Created: 13/12/16
Edited: 
15/12/16: added the function fnMonostaticRadar_Range_Doppler
18/12/16: added the function fnJacobianH_Doppler to calculate the Doppler-related sensitivity matrix variables.
19/12/16: added the function fnJacobianH_Monostatic_RangeAndDoppler to evaluate the observation sensitivity matrix when
          the observation consists of monostatic range and Doppler.
19/12/16: added the function fnCalculate_Monostatic_RangeAndDoppler to calculate the radar measurement vector consisting of 
          slant-range Doppler in the monostatic case.
20/12/16: edited the functions fnCalculate_Doppler_Shift_2D and fnCalculate_Doppler_Shift_3D: assumed negative sign of Doppler shift
20/12/16: removed the function fnMonostaticRadar_Range_Doppler
26/12/16: edited the function fnCalculate_Monostatic_RangeAndDoppler: removed the assumption that the sensor is located at the origin.
27/12/16: moved the functions relating to bistatic radar from UCM to BistaticAndDoppler.py
27/12/16: added the function fnJacobianH_Bistatic_RangeAndDoppler
27/12/16: edited a doppler function to make use of the wave number such as 
          fnJacobianH_Doppler, fnCalculate_Doppler_Shift_2D, fnCalculate_Doppler_Shift_3D
19/01/17: created the function fnCalculate_Monostatic_RangeAndDoppler_OD
20/01/17: fixed the Doppler in fnCalculate_Monostatic_RangeAndDoppler_OD and also imported the AstCnst.
22/01/17: added the function fnJacobianH_Monostatic_RangeAndDoppler_OD
22/01/17: debugged fnCalculate_Bistatic_RangeAndDoppler.
22/01/17: created the function fnCalculate_Bistatic_RangeAndDoppler_OD and also imported AstFn
23/01/17: created the function fnJacobianH_Bistatic_RangeAndDoppler_OD
24/01/17: streamlined the function fnCalculate_Monostatic_RangeAndDoppler_OD
26/01/17: fixed the function fnJacobianH_Bistatic_RangeAndDoppler_OD
26/01/17: fixed a typo concerning the wavenumber in every function concerning Doppler shift.
09/02/17: added the function fnCalculate_Bistatic_Angle
10/02/17: fixed a typo concerning the wavenumber in every function concerning Doppler shift. Essentially, reversing the edits made on 26/01/17
17/04/17: Created the function fnCalculate_Bistatic_Range_Resolution & fnCalculate_Bistatic_Doppler_Resolution
22/04/17: added the function fnCalculate_PlaneNormal
22/04/17: added the functions fnCalculate_LawOfSines_Angles, fnCalculate_Bistatic_Coordinates and fnCalculate_Carts_to_RUV and fnCalculate_RUV_to_Carts
26/04/17: removed the functions fnCalculate_PlaneNormal and fnCalculate_LawOfSines_Angles
26/09/17: debugged the functions fnCalculate_Monostatic_RangeAndDoppler_OD & fnJacobianH_Monostatic_RangeAndDoppler_OD
26/09/17: created the function fnCalculate_Bistatic_RangeAndDoppler_OD_fast
26/09/17: edited the function fnJacobianH_Bistatic_RangeAndDoppler_OD
"""
import math
import numpy as np
import AstroConstants as AstCnst
import AstroFunctions as AstFn
# ------------------------------------------------------------------- #
## Calculate Doppler shift
def fnCalculate_Doppler_Shift_2D(wavelength,x_target,pos_sensor):
    """
    Calculate the Doppler shift of the target measured at the
    sensor located at pos_sensor.
    Note: We assume a 2-dimensional problem, with x_target being [pos_x,pos_y,vel_x,vel_y]
    Also, wavelength is in [km]
    Date: 11/12/16
    Edited: 
    20/12/16: sign for Doppler shift.
    26/01/17: fixed typo in wavenumber
    10/02/17: fixed typo in wavenumber
    """
    relative_position = x_target[0:2] - pos_sensor;
    #kd =(-2.*math.pi/wavelength); # typo fixed. 26/01/17
    kd =(-1./wavelength); # typo fixed. 10/02/17. Ref. Skolnik Bistatic radar
    doppler_shift = kd*(relative_position[0]*x_target[2] + relative_position[1]*x_target[3])/(np.linalg.norm(relative_position));
    return doppler_shift

def fnCalculate_Doppler_Shift_3D(wavelength,x_target,pos_sensor):
    """
    Calculate the Doppler shift of the target measured at the
    sensor located at pos_sensor.
    Note: We assume a 3-dimensional problem, with x_target being [pos_x,pos_y,pos_z,vel_x,vel_y,vel_z]
    Also, wavelength is in [km]
    Date: 11/12/16
    Edited: 
    20/12/16: sign for Doppler shift
    26/01/17: fixed typo in wavenumber
    10/02/17: fixed typo in wavenumber
    """
    relative_position = x_target[0:3] - pos_sensor;
    #kd = (-2.*math.pi/wavelength);# 26.01.17: typo in wavenumber
    kd =(-1./wavelength); # typo fixed. 10/02/17. Ref. Skolnik Bistatic radar
    doppler_shift = kd*(relative_position[0]*x_target[3] + relative_position[1]*x_target[4] + relative_position[2]*x_target[5])/(np.linalg.norm(relative_position));
    return doppler_shift # Validated in main_056_iss_25.py on 26/09/17
# -------------------------------------------------------------------------------------------------------- #
def fnJacobianH_Doppler(wavelength,Xnom):
    """
    The variables relating to Doppler in the Observation sensitivity matrix.
    Xnom: 6x1
    0:3 -> x,y,z position
    3:6 -> x,y,z velocity
    Note: wavelength is in [km]
    Ref:
    Tracking filter engineering, Morrison. 2013.
    Date: 18 December 2016
    Edited: 
    26/01/17: fixed typo in wavenumber
    10/02/17: fixed typo in wavenumber
    """
    x = Xnom[0];y = Xnom[1];z = Xnom[2];
    xdot = Xnom[3];ydot = Xnom[4];zdot = Xnom[5];
    rho = np.linalg.norm(Xnom[0:3]);
    #kd = (-2.*math.pi/wavelength); # wave number 26.01.17
    kd =(-1./wavelength); # typo fixed. 10/02/17. Ref. Skolnik Bistatic radar
    doppler_partials = np.zeros([6],dtype=np.float64);
    doppler_partials[0] = kd*(xdot*(y**2 + z**2) - x*(y*ydot+z*zdot))/(rho**3);
    doppler_partials[1] = kd*(ydot*(x**2 + z**2) - y*(x*xdot+z*zdot))/(rho**3);
    doppler_partials[2] = kd*(zdot*(y**2 + x**2) - z*(y*ydot+x*xdot))/(rho**3);
    doppler_partials[3] = kd*(x)/(rho);
    doppler_partials[4] = kd*(y)/(rho);
    doppler_partials[5] = kd*(z)/(rho);
    return doppler_partials
    
def fnJacobianH_Monostatic_RangeAndDoppler(wavelength,Xnom):
    """
    Measurement sensitivity matrix for monostatic measurements consisting of
    slant-range and Doppler.
    
    wavelength = wavelength in [km]
    Xnom: 6x1 state vector
    0:3 -> x,y,z position
    3:6 -> x,y,z velocity
    
    Calls fnJacobianH_Doppler.
    
    Created: 19 December 2016
    Edited:
    """
    rho = np.linalg.norm(Xnom[0:3]);
    Mmatrix = np.zeros([2,6],dtype=np.float64);
    Mmatrix[0,0] = Xnom[0]/rho; 
    Mmatrix[0,1] = Xnom[1]/rho; 
    Mmatrix[0,2] = Xnom[2]/rho; 
    doppler_partials = fnJacobianH_Doppler(wavelength,Xnom);
    Mmatrix[1,:] = doppler_partials;
    return Mmatrix

def fnCalculate_Monostatic_RangeAndDoppler(wavelength,x_target,pos_sensor):
    """
    Measurement function for monostatic radar measuring slant-range and Doppler only.
    wavelength = wavelength in [km]
    x_target:6x1 state vector
    0:3 -> x,y,z position
    3:6 -> x,y,z velocity
    pos_sensor -> position of the radar sensor.
    
    Created: 19/12/16
    Edited: 26/12/16: removed the assumption that the sensor is located at the origin.
    """
    y_radar = np.zeros([2],dtype=np.float64);
    # Slant-range
    y_radar[0] = np.linalg.norm(np.subtract(x_target[0:3],pos_sensor));
    # Doppler-shift
    y_radar[1] = fnCalculate_Doppler_Shift_3D(wavelength,x_target,pos_sensor);
    return y_radar # Validated in main_056_iss_25.py on 26/09/17
# -------------------------------------------------------------------------------------------------------------------- #
## Functions for bistatic radar measurements.
def fnCalculate_Spherical_Bistatic(pos_target,pos_rx,pos_tx):
    """
    Calculate measurement vector for 3D bistatic case.
    pos_rx, pos_tx = position of Rx and Tx in [km].
    pos_target = position of target in [km].
    Eqn 5.13 Bordonaro thesis
    
    Date: 05/12/16
    Edited:
    21/12/16 : removed the assumptions on position of Rx and Tx.
    23/12/16: optimized code.
    """
    target_rx = np.subtract(pos_target,pos_rx);
    target_tx = np.subtract(pos_target,pos_tx);
    # Sensor measures bistatic range and look angles.
    Xinput_sph = np.zeros([3],dtype=np.float64);
    # Calculate range
    Xinput_sph[0] = np.linalg.norm(target_rx) + np.linalg.norm(target_tx);
    # Calculate elevation
    Xinput_sph[1] = math.atan(target_rx[2]/np.linalg.norm(target_rx[0:2]));
    # Calculate azimuth
    Xinput_sph[2] = math.atan2(target_rx[1],target_rx[0]);
    return Xinput_sph    

def fnJacobianH_Spherical_Bistatic(Xnom,pos_rx,pos_tx):
    """
    Observation sensitivity matrix.
    Radar measurement vector consists of bistatic range, azimuth, elevation.
    
    Based on fnJacobianH
    This function pertains to the transformation in fnCalculate_Spherical_Bistatic
    
    Date: 05 December 2016
    Edited: 
    9 and 10 December 2016: fixed a couple of typos.
    18/12/16: fixed a calculus mistake. 
    21/12/16: Due to changed definition of Rx and Tx, had to re-evaluate the partials
    """
    pos_target = Xnom[0:3];
    target_rx = np.subtract(pos_target,pos_rx);
    target_tx = np.subtract(pos_target,pos_tx);
    
    rho = np.linalg.norm(target_rx);
    s = np.linalg.norm(target_rx);
    rho_b = np.linalg.norm(target_tx);
    
    Mmatrix = np.zeros([3,3],dtype=np.float64);
    Mmatrix[0,0] = target_rx[0]/rho  + target_tx[0]/rho_b; 
    Mmatrix[0,1] = target_rx[1]/rho  + target_tx[1]/rho_b; 
    Mmatrix[0,2] = target_rx[2]/rho  + target_tx[2]/rho_b; 

    Mmatrix[1,0] = -target_rx[0]*target_rx[2]/(s*(rho**2)); 
    Mmatrix[1,1] = -target_rx[1]*target_rx[2]/(s*(rho**2)); 
    Mmatrix[1,2] = s/(rho**2); 

    Mmatrix[2,0] = -target_rx[1]/s**2;
    Mmatrix[2,1] = target_rx[0]/s**2;           
    return Mmatrix # edited: 21/12/16 due to changed definition of Tx and Rx
    

#def fnCalculate_Bistatic_Spherical(Xinput_sph,pos_rx,pos_tx):
#    """
#    Reverses fnCalculate_Spherical_Bistatic.
#    Calculates Cartesian measurement vector from bistatic measurement vector
#    Eqn 5.16 in Bordonaro's thesis.
#    
#    Xinput_sph = 3D bistatic measurement vector consisting of bistatic range, elevation and azimuth angles.
#    
#    Author: AshivD.
#    Date: 09/12/16
#    Edited: 21/12/16: due to change in fnCalculate_Spherical_Bistatic, had to change this function.
#    """
#    b = Xinput_sph[0];el = Xinput_sph[1]; az = Xinput_sph[2];
#    # r1 is the monostatic slant range
#    r1 = (L**2 - b**2)/(2.*(L*math.cos(az)*math.cos(el)-b));
#    # We already have a function to convert from spherical to Cartesian,  
#    # so we reuse with a measurement covariance full of zeros.
#    R_spherical = np.zeros([3,3],dtype=np.float64);
#    X_sph = np.array([r1,el,az],dtype=np.float64);
#    Xinput_carts = fnCalculate_UCM3D_SEZ(X_sph,R_spherical);
#    return Xinput_carts # Validated 9/12/16, see main_001_iss.py
### Need to change the above function.

# -------------------------------------------------------------------------------------------------------------------------------- #
## Bistatic range and Doppler
def fnCalculate_Bistatic_RangeAndDoppler(pos_target,vel_target,pos_rx,pos_tx,wavelength):
    """
    Calculate measurement vector consisting of bistatic range and Doppler shift for 3D bistatic case.
    pos_rx, pos_tx = position of Rx and Tx in [km].
    pos_target = position of target in [km].
    wavelength = wavelength of radar transmitter in [km].
    Validated in main_iss_bistatic_rangedopp_01.py
    Date: 27/12/16
    Edited:
    22/01/17: fixed a bug in the expression for Doppler shift. Forgot to include the Doppler shift due to the transmitter.
    
    """
    target_rx = np.subtract(pos_target,pos_rx);
    target_tx = np.subtract(pos_target,pos_tx);
    
    y_radar = np.zeros([2],dtype=np.float64);
    # bistatic range
    y_radar[0] = np.linalg.norm(target_rx) + np.linalg.norm(target_tx)
    # Doppler shift
    pos_vel = np.hstack((pos_target,vel_target));
    y_radar[1] = fnCalculate_Doppler_Shift_3D(wavelength,pos_vel,pos_rx) + fnCalculate_Doppler_Shift_3D(wavelength,pos_vel,pos_tx); # fixed: 22/01/17
    return y_radar

def fnJacobianH_Bistatic_RangeAndDoppler(wavelength,Xnom,pos_rx,pos_tx):
    """
    Measurement sensitivity matrix for bistatic measurements consisting of
    bistatic range and Doppler.
    
    wavelength = wavelength in [km]
    Xnom: 6x1 state vector
    0:3 -> x,y,z position
    3:6 -> x,y,z velocity
    
    Created: 27 December 2016
    Edited:
    
    21.09.17: cleaned up
    """
    pos_target = Xnom[0:3];
    target_rx = np.subtract(pos_target,pos_rx);
    target_tx = np.subtract(pos_target,pos_tx);
    
    rho = np.linalg.norm(target_rx);
    rho_b = np.linalg.norm(target_tx);
    
    Mmatrix = np.zeros([2,6],dtype=np.float64);
    # the partials related to bistatic range are obtained from fnJacobianH_Spherical_Bistatic
    Mmatrix[0,0] = target_rx[0]/rho  + target_tx[0]/rho_b; 
    Mmatrix[0,1] = target_rx[1]/rho  + target_tx[1]/rho_b; 
    Mmatrix[0,2] = target_rx[2]/rho  + target_tx[2]/rho_b; 
    # the partials related to Doppler shift are obtained from fnJacobianH_Monostatic_RangeAndDoppler
    doppler_partials = fnJacobianH_Doppler(wavelength,Xnom);
    Mmatrix[1,:] = doppler_partials;
    return Mmatrix # note 23/01/17: need to fix the doppler partials here.

"""
The above function needs to get sorted. 25.05.17

Doppler partials wrt Tx are missing here. 20.09.17
"""

# ---------------------------------------------------------------------------------------------------------------------- #
def fnCalculate_Monostatic_RangeAndDoppler_OD(target_eci,radar_ecef,theta,wavelength):   
    """
    Different way of calculating monostatic range and Doppler.
    Based on ASEN5070 notes. First developed in testrangerate2.py
    Created: 19 January 2017
    Edited: 
    20 January 2017: fixed the Doppler part and validated the results with those from fnCalculate_Monostatic_RangeAndDoppler
    24 January 2017: streamlined the code to match with its bistatic counterpart.
    26 January 2017: fixed typo in wavenumber
    10/02/17: fixed typo in wavenumber
    20/09/17: commented out what's not needed
    """ 
    # Position variables of target in the ECI frame.
    target_x = target_eci[0];target_y = target_eci[1];target_z = target_eci[2];
    # Velocity variables of target in the ECI frame.
    target_vx = target_eci[3];target_vy = target_eci[4];target_vz = target_eci[5];
    # Position vector of radar in the ECEF frame.
    rx_x = radar_ecef[0];rx_y = radar_ecef[1];rx_z = radar_ecef[2];
    
    rho_vec = np.zeros([2],dtype=np.float64);
    rx_eci = AstFn.fnECEFtoECI(radar_ecef,theta);
    rx_vec = np.subtract(target_eci[0:3],rx_eci);
    rx_mag = np.linalg.norm(rx_vec); rho_vec[0] = rx_mag;
    kd =(-1./wavelength); # typo fixed. 10/02/17. Ref. Skolnik Bistatic radar
    # Doppler shift
    rho_vec[1] = (kd/rx_mag)*(target_x*target_vx + target_y*target_vy + target_z*target_vz - (target_vx*rx_x + target_vy*rx_y)*math.cos(theta) + AstCnst.theta_dot*(target_x*rx_x+target_y*rx_y)*math.sin(theta)
                  +(target_vx*rx_y - target_vy*rx_x)*math.sin(theta) + AstCnst.theta_dot*(target_x*rx_y - target_y*rx_x)*math.cos(theta) - target_vz*rx_z)
    return rho_vec # validated 25.09.17 in main_056_iss_25.py, main_056_iss_27.py

def fnJacobianH_Monostatic_RangeAndDoppler_OD(target_eci,radar_ecef,theta,wavelength):
    """
    Calculates the Jacobian matrix for the monostatic range and Doppler shift for the OD problem.
    Based on ASEN5070 notes. First developed in testrangerate2.py
    Created: 21 January 2017
    Edited:
    24/01/17: fixed errors in derivation of partial derivatives.
    26/01/17: fixed typo in wavenumber
    10/02/17: fixed typo in wavenumber
    25/09/17: debugged the expressions related to Doppler shift partials
    27/09/17: cleaned up variables dx and dy
    """
    # Position variables of target in the ECI frame.
    target_x = target_eci[0];target_y = target_eci[1];target_z = target_eci[2];
    # Velocity variables of target in the ECI frame.
    target_vx = target_eci[3];target_vy = target_eci[4];target_vz = target_eci[5];
    # Position vector of radar in the ECEF frame.
    radar_x = radar_ecef[0];radar_y = radar_ecef[1];radar_z = radar_ecef[2];
    # Calculate the monostatic range and Doppler shift measurement.
    rangedopp = fnCalculate_Monostatic_RangeAndDoppler_OD(target_eci,radar_ecef,theta,wavelength);
    kd =(-1./wavelength); # typo fixed. 10/02/17. Ref. Skolnik Bistatic radar
    rho = rangedopp[0]; 
    rhodot = rangedopp[1]; # 24.09.17
    # partial derivatives between range and position variables
    dx = np.shape(target_eci)[0];
    dy = np.shape(rangedopp)[0];
    Mmatrix = np.zeros([dy,dx],dtype=np.float64);#edited: 27.09.17
    Mmatrix[0,0] = (target_x-radar_x*math.cos(theta)+radar_y*math.sin(theta))/rho;
    Mmatrix[0,1] = (target_y-radar_y*math.cos(theta)-radar_x*math.sin(theta))/rho;
    Mmatrix[0,2] = (target_z-radar_z)/rho;
    # partial derivatives between range and velocity variables are 0.
    
    # Partial derivatives wrt to range re-derived and verified on 25.09.17 in main_056_iss_25.py
    Mmatrix[1,0] = (kd/rho**2)*(rho*(target_vx+radar_x*math.sin(theta)*AstCnst.theta_dot + radar_y*math.cos(theta)*AstCnst.theta_dot) - target_x*rhodot);
                    
    Mmatrix[1,1] = (kd/rho**2)*(rho*(target_vy-radar_x*math.cos(theta)*AstCnst.theta_dot + radar_y*math.sin(theta)*AstCnst.theta_dot) - target_y*rhodot);
    Mmatrix[1,2] = (kd/rho**2)*(rho*target_vz - target_z*rhodot);
    # partial derivatives between Doppler shift and velocity variables.
    Mmatrix[1,3] = (kd/rho)*(target_x - radar_x*math.cos(theta) + radar_y*math.sin(theta));
    Mmatrix[1,4] = (kd/rho)*(target_y - radar_x*math.sin(theta) - radar_y*math.cos(theta));
    Mmatrix[1,5] = (kd/rho)*(target_z - radar_z);   
    return Mmatrix

# ------------------------------------------------------------------------------------------------------------------------------ #
def fnCalculate_Bistatic_RangeAndDoppler_OD(target_eci,rx_ecef,tx_ecef,theta,wavelength):
    """
    Calculate the bistatic range and Doppler shift for the OD problem.
    Validated in main_iss_bistatic_rangedopp_01.py
    Created: 22 January 2017
    Edited:
    26/01/17: fixed typo in wavenumber
    10/02/17: fixed typo in wavenumber
    """   
    # Position variables of target in the ECI frame.
    target_x = target_eci[0];target_y = target_eci[1];target_z = target_eci[2];
    # Velocity variables of target in the ECI frame.
    target_vx = target_eci[3];target_vy = target_eci[4];target_vz = target_eci[5];
    # Position vector of radar receiver in the ECEF frame.
    rx_x = rx_ecef[0];rx_y = rx_ecef[1];rx_z = rx_ecef[2];
    # Position vector of radar transmitter in the ECEF frame.
    tx_x = tx_ecef[0];tx_y = tx_ecef[1];tx_z = tx_ecef[2];
    
    rho_vec = np.zeros([2],dtype=np.float64);
    rx_eci = AstFn.fnECEFtoECI(rx_ecef,theta);
    tx_eci = AstFn.fnECEFtoECI(tx_ecef,theta);
    rx_vec = np.subtract(target_eci[0:3],rx_eci);
    tx_vec = np.subtract(target_eci[0:3],tx_eci);
    rx_mag = np.linalg.norm(rx_vec);
    tx_mag = np.linalg.norm(tx_vec);
    rho_vec[0] = rx_mag + tx_mag;
    #kd =(-2.*math.pi/wavelength); #wave number. fixed typo on 26/01/17
    kd =(-1./wavelength); # typo fixed. 10/02/17. Ref. Skolnik Bistatic radar
    # Doppler shift
    rx_doppler = (kd/rx_mag)*(target_x*target_vx+target_y*target_vy+target_z*target_vz +AstCnst.theta_dot*math.sin(theta)*(target_x*rx_x + target_y*rx_y)-math.cos(theta)*(target_vx*rx_x+target_vy*rx_y)+ AstCnst.theta_dot*math.cos(theta)*(target_x*rx_y - target_y*rx_x)+math.sin(theta)*(target_vx*rx_y-target_vy*rx_x) -target_vz*rx_z);
    tx_doppler = (kd/tx_mag)*(target_x*target_vx+target_y*target_vy+target_z*target_vz +AstCnst.theta_dot*math.sin(theta)*(target_x*tx_x + target_y*tx_y)-math.cos(theta)*(target_vx*tx_x+target_vy*tx_y)+ AstCnst.theta_dot*math.cos(theta)*(target_x*tx_y - target_y*tx_x)+math.sin(theta)*(target_vx*tx_y-target_vy*tx_x) -target_vz*tx_z);
    rho_vec[1] = tx_doppler + rx_doppler;
    return rho_vec # Results match fnCalculate_Bistatic_RangeAndDoppler in main_056_iss_27.py

def fnJacobianH_Bistatic_RangeAndDoppler_OD(target_eci,rx_ecef,tx_ecef,theta,wavelength):
    """
    Calculate the Jacobian matrix for the bistatic range and Doppler shift for the OD problem.
    Created: 23 January 2017
    Edited: 
    26/01/2017 : fixed typo in wavenumber
    10/02/17: fixed typo in wavenumber
    20/09/17: commented out what's not needed
    26/09/17: deleted outdated implementation, replaced by faster implementation
    """   
    # partial derivatives between range and position variables
    dy = 2;
    dx = np.shape(target_eci)[0];
    Mmatrix = np.zeros([dy,dx],dtype=np.float64);
    rx_m = fnJacobianH_Monostatic_RangeAndDoppler_OD(target_eci,rx_ecef,theta,wavelength);
    tx_m = fnJacobianH_Monostatic_RangeAndDoppler_OD(target_eci,tx_ecef,theta,wavelength);
    Mmatrix = np.add(rx_m,tx_m)
    return Mmatrix # validated in main_iss_bistatic_rangedopp_02.py

# ----------------------------------------------------------------------------------------------------------------------------- #
def fnCalculate_Bistatic_Angle(pos,Rx_pos,Tx_pos):
    """
    Calculate the bistatic angle from the target position and the receiver and transmitter positions.
    Created on: 08/02/17
    """
    vec_rx = np.subtract(pos,Rx_pos);
    vec_tx = np.subtract(pos,Tx_pos);
    bis_angle = math.acos(np.dot(vec_rx,vec_tx)/(np.linalg.norm(vec_rx)*np.linalg.norm(vec_tx)));
    return bis_angle   # Validated in main_bis_rd_fengyun_bistaticangle.py    
    
def fnCalculate_Bistatic_Range_Resolution(speed_light,bis_angle,bandwidth):
	"""
	Calculates the bistatic range resolution corresponding to a given bistatic angle, given
	the bandwidth of the radar system.
	
	Equation 11.44 on page 236 in Cherniakov: Bistatic Radar: Principles & Practice.
	
	Created on: 17 April 2017
	"""
	range_resolution = speed_light/(2*math.cos(0.5*bis_angle)*bandwidth);
	return range_resolution

def fnCalculate_Bistatic_Doppler_Resolution(wavelength,t_cpi,bis_angle):
	"""
	Calculates the bistatic Doppler resolution corresponding to a given bistatic angle,
	given the CPi length in seconds of the radar system.
	
	Equation 7.5b on page 134 in Willis: Bistatic Radar
	
	Created on: 17 April 2017
	"""
	doppler_resolution = wavelength/(2.*t_cpi*math.cos(bis_angle*0.5));
	return doppler_resolution

# ---------------------------------------------------------------------------------------------------------------------------------- #
def fnCalculate_Bistatic_Coordinates(a,B):
	"""
	Calculate the coordinates of the target in the bistatic plane
	
	A,B,C = angles in the triangle
	a,b,c = length of the side opposite the angle
	
	Created: 22 April 2017
	"""
	u = a*math.cos(B);
	v = a*math.sin(B);
	return u,v

def fnCalculate_Carts_to_RUV(xpos):
	"""
	Calculate the r-u-v coordinates (direction cosines) 
	of a target from its Cartesian position.
	
	Refer to the Tian, Bar-Shalom paper
	
	Created: 22 April 2017
	"""
	r = np.linalg.norm(xpos);
	u = xpos[0]/xpos[1];
	v = xpos[1]/r;
	return r,u,v
	
def fnCalculate_RUV_to_Carts(r,u,v):
	"""
	Calculate the Cartesian position vector of
	a target from its r-u-v coordinates.
	
	Refer to: Tian, Bar-Shalom 05259184.pdf
	
	r-u-v coordinates are used in bistatic and
	multistatic radar tracking scenarios.
	
	This function reverses the operation of
	fnCalculate_Carts_to_RUV
	
	Created: 22 April 2017
	"""
	xpos = np.zeros([3],dtype=np.float64);
	xpos[0] = r*u;
	xpos[1] = r*v;
	xpos[2] = r*math.sqrt(1-u**2-v**2);
	return xpos

# --------------------------------------------------------------------------- #
def fnCalculate_Bistatic_RangeAndDoppler_OD_fast(target_eci,rx_ecef,tx_ecef,theta,wavelength):
    """
    Calculate the bistatic range and Doppler shift for the OD problem.
    Validated in main_056_iss_27.py

    Alternative to fnCalculate_Bistatic_RangeAndDoppler_ODs    
    
    Created: 26 September 2017
    Edited:

    """   
    rho_vec_tx = fnCalculate_Monostatic_RangeAndDoppler_OD(target_eci,tx_ecef,theta,wavelength);
    rho_vec_rx = fnCalculate_Monostatic_RangeAndDoppler_OD(target_eci,rx_ecef,theta,wavelength);
    rho_vec = rho_vec_tx+rho_vec_rx;
    return rho_vec # same performance as fnCalculate_Bistatic_RangeAndDoppler_OD. See main_056_iss_27.py. 26.09/17