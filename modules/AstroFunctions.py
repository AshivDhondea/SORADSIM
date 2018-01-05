# -*- coding: utf-8 -*-
"""
AstroFunctions.py

Description:
Various Python functions useful for astrodynamics applications.
Most of these functions are based on Fundamentals of Astrodynamics, Vallado. 4th ed.

Created by: Ashiv Dhondea, RRSG, UCT.

Date created: 20 June 2016

Edits: 
02.08.16: added a number of functions for coordinate transformations, etc from Vallado book.
10.09.16: added functions for RAZEL. Functions needed for the observation framework.
13.09.16: added two functions for extracting Keplerians from a Two Line Element set.
14.09.16: added functions to convert Cartesians to Keplerians
14.09.16: added functions to transform covariance matrices for linea transformations.
28.09.16: edited the function fnTLEtoKeps to also return the epoch 
28.09.16: added function fn_Calculate_Epoch_Time to calculate the epoch time from a TLE file.
28.09.16: added function fn_epoch_date to calculate the date.
28.09.16: edited several functions: use math instead of np for non-array operations.
28.09.16: added function to convert from seconds to hours and minutes.
29.09.16: added function for time duration.
02.10.16: added function to convert from hours and minutes to seconds
05.10.16: added function JulianDate and function to compute GMST from Julian Day.
06.10.16: added comments.
07.10.16: added function to wrap angles.
16.10.16: added function fnZeroToPi to wrap angles.
12.11.16: added functions fnHerrickGibbs and fnAngle to compute an initial orbit / nominal trajectory.
15.11.16: added the function fnSEZtoECIobsv
16.11.16: added the function fnwrapToPi
06.12.16: moved the function fnCarts_to_LatLon to AstroFunctions
08.12.16: edited the function fn_Convert_Datetime_to_GMST to account for microseconds.
05.01.17: cleaned up file.
05.01.17: added the function fnECItoECEF_vel
06.01.17: added the function fnVel_Cartesian
15.01.17: renamed the function fnVel_Cartesian to fnVel_ECI_to_SEZ, debugged it and validated it.
15.01.17: created the function  fnVel_SEZ_to_ECI. Validated.
11.04.17: added the function fnCalc_ApogeePerigee
24.05.16: added the function fnCalculate_DatetimeEpoch
29.05.17: added the function fnConvert_AZEL_to_Topo_RADEC from test_radec.py
29.05.17: added the function fnConvert_Topo_RADEC_to_AZEL from test_radec.py
28.08.17: added the function fnRead_Experiment_Timestamps
29.08.17: moved 3 rotation functions to GeometryFunctions
29.08.17: moved all functions concerning time TimeHandlingFunctions
29.08.17: cleaned up presentation
"""
# ------------------------- #
# Import libraries
import numpy as np
import math

import AstroConstants as AstCnst
import GeometryFunctions as GF
# -------------------------- #
def fnKepsToCarts(a,e, i, omega,BigOmega, nu):
    # semi-parameter 
    p = a*(1 - e**2);
    # pqw: perifocal coordinate system 
    # Find R and V in pqw coordinate system
    R_pqw = np.zeros([3],dtype=np.float64);
    V_pqw = np.zeros([3],dtype=np.float64);
    
    R_pqw[0] = p*math.cos(nu)/(1 + e*math.cos(nu));
    R_pqw[1] = p*math.sin(nu)/(1 + e*math.cos(nu));
    R_pqw[2] = 0;  
    V_pqw[0] = -math.sqrt(AstCnst.mu_E/p)*math.sin(nu);
    V_pqw[1] =  math.sqrt(AstCnst.mu_E/p)*(e + math.cos(nu));
    V_pqw[2] =   0;
    # Then rotate into ECI frame
    R = np.dot(np.dot(GF.fnRotate3(-BigOmega),GF.fnRotate1(-i)),np.dot(GF.fnRotate3(-omega),R_pqw));
    V = np.dot(np.dot(GF.fnRotate3(-BigOmega),GF.fnRotate1(-i)),np.dot(GF.fnRotate3(-omega),V_pqw));
    xstate = np.hstack((R,V));
    return xstate # Validated against Vallado's example 2-3. 20/06/16.
    # Edited: 28 September 2016: use math instead of np for non-array operations.

def fnKeplerOrbitalPeriod(a):
    # T is the orbital period in [s]
    T = 2*math.pi*math.sqrt(a**3/AstCnst.mu_E);
    return T
    # Edited: 28 September 2016: use math instead of np for non-array operations.
    
# --Frame conversion functions------------------------------------------------ #
## 3 August 2016: note that these are only approximate transforms. Precession,
# nutation, polar motion are ignored. Can be used in simulations, but not for real-life applications.
def fnECItoECEF(ECI,theta):
    ECEF = np.zeros([3],dtype=np.float64);
    # Rotating the ECI vector into the ECEF frame via the GST angle about the Z-axis
    ECEF = np.dot(GF.fnRotate3(theta),ECI);
    return ECEF
def fnECEFtoECI(ECEF,theta):
    ECI = np.zeros([3],dtype=np.float64);
    # Rotating the ECEF vector into the ECI frame via the GST angle about the Z-axis
    ECI = np.dot(GF.fnRotate3(-theta),ECEF);
    return ECI

def fnECItoECEF_vel(ECI_vel,theta,R_ECEF):
    """
    Velocity transformation from ECI to ECEF, featuring velocity correction term.
    Date: 05/01/17
    Edited: 15/01/17: debugged.
    """
    omega = np.array([0.,0.,-AstCnst.theta_dot]);
    ECEF = fnECItoECEF(ECI_vel,theta) + np.cross(omega,R_ECEF);
    return ECEF # Validated in testval.py, from example 7-1 on pg 431 in Vallado.

def fnECEFtoECI_vel(ECEF_vel,theta,R_ECI):
    """
    Velocity transformation from ECEF to ECI, featuring velocity correction term.
    Date: 06/01/17
    """
    omega = np.array([0.,0.,AstCnst.theta_dot]);
    ECI = fnECEFtoECI(ECEF_vel,theta) + np.cross(omega,R_ECI);
    return ECI# validated in testval.py, from example 7-1 on pg 431 in Vallado.
# -- Algorithms from Vallado, 4th edition ---------------------------------------------- #
## Refer to algorithm 51 in Vallado, 4th ed.
def fnSiteTrack(latitude_gd,longitude,altitude,rho,az,el,theta):
    """
    fnSiteTrack implements Algorithm 51 in Vallado, 4th edition.
    
    Notes (10 September 2016)
    1. latitude_gd,longitude,altitude come from the radar sensor
    2. rho,az,el pertain to the target
    3. theta is an angle in radians related to the Earth's rotation. i.e. related to time.    
    """
    # Find radar sensor's location in the ECEF frame.
    Radar_ECEF = fnRadarSite(latitude_gd,longitude,altitude);
    # Convert observations in the radar centered (local) coordinate frame to the SEZ Cartesian frame.
    R_SEZ = fnRadarToSEZ(rho,az,el );
    # Convert position in SEZ frame to ECEF frame.
    R_ECEF = fnSEZtoECEF( R_SEZ,latitude_gd,longitude );

    r_ecef = Radar_ECEF + R_ECEF;
    # Convert r_ecef to R_ECI using algorithm 24.
    # No, I prefer to use the simplified approach.
    R_ECI = fnECEFtoECI(r_ecef,theta);
    return R_ECI

def fnSiteTrack_Cartesian(latitude_gd,longitude,altitude,R_SEZ,theta):
    """
    The Cartesian version of SiteTrack
    """
    # Find radar sensor's location in the ECEF frame.
    Radar_ECEF = fnRadarSite(latitude_gd,longitude,altitude);
    
    # Convert position in SEZ frame to ECEF frame.
    R_ECEF = fnSEZtoECEF( R_SEZ,latitude_gd,longitude );

    r_ecef = Radar_ECEF + R_ECEF;
    # Convert r_ecef to R_ECI using algorithm 24.
    # No, I prefer to use the simplified approach.
    R_ECI = fnECEFtoECI(r_ecef,theta);
    return R_ECI

def fnRadarSite(latitude_gd,longitude,altitude):
    """
    % fnRadarSite determines the ECEF position of the radar tracking station.
    % It emulates site.m from Vallado's book. [part of Algorithm 51].
    Validated 02/08/16 with example 7-1 in Vallado book.
    """
    # Earth's shape eccentricity
    e_earth_squared = (2*AstCnst.flattening-AstCnst.flattening**2);

    # Calculate 2 auxiliary points
    cearth= AstCnst.R_E/np.sqrt( 1 - (e_earth_squared*(math.sin( latitude_gd ))**2 ) );
    searth = cearth*(1-e_earth_squared);

    h_ellp = fnEllipsoidalHeight( altitude,latitude_gd ); #  Validated 02/08/16 with example 7-1 in Vallado book.
    rdel  = (cearth+ h_ellp )*math.cos( latitude_gd);
    rk    = (searth + h_ellp )*math.sin( latitude_gd );

    # Radar_ECEF is the position vector in the Cartesian ECEF frame.    
    Radar_ECEF = np.zeros(3,dtype=np.float64);
    Radar_ECEF[0] = rdel * math.cos( longitude );
    Radar_ECEF[1] = rdel * math.sin( longitude );
    Radar_ECEF[2] = rk;
    return Radar_ECEF

def fnEllipsoidalHeight(altitude,latitude_gd):
    """
    % fnEllipsoidalHeight finds the ellipsoidal height for a target at a given
    % altitude
    Validated 02/08/16 with example 7-1 in Vallado book.
    """
    latitude_gc = fnGeodeticToGeocentricLatitude(latitude_gd);
    h_ellp = altitude/(math.cos(latitude_gd - latitude_gc));
    return h_ellp

def fnGeodeticToGeocentricLatitude(latitude_gd):
    """
    % fnGeodeticToGeocentricLatitude converts the latitude from geodetic to
    % geocrentric. This is necessary because of the oblateness of the Earth.
    Validated 02/08/16 with example 7-1 in Vallado book.
    """
    latitude_gc = math.atan((1-AstCnst.flattening**2)*math.tan(latitude_gd));
    return latitude_gc
    # Edited: 28 September 2016: use math instead of np for non-array operations.

def fnRadarToSEZ(rho,az,el ):
    """
    % fnRadarToSEZ converts observations (rho,az,el) to the SEZ Cartesian
    % coordinate system.
    % Based on raz2rvs.m from Vallado's book

    % Note that 
    % 1. angles are assumed to be in degrees in this function. [02/08/16: no, in radians.]
    % 2. the azimuth angle is measured from the negative x axis to the positive
    % y axis in the xy plane.
    % 3. the elevation angle is measured from the positive xy plane to the
    % positive z axis.
    # Validated with example 7-1
    """
    R_SEZ = np.zeros(3,dtype=np.float64);
    R_SEZ[0] = -rho*math.cos(el)*math.cos(az); #% x
    R_SEZ[1] =  rho*math.cos(el)*math.sin(az); #% y
    R_SEZ[2] =  rho*math.sin(el);      #  % z
    return R_SEZ

def fnSEZtoECEF( R_SEZ,latitude_gd,longitude ):
    """
    %% fnSEZtoECEF transforms a position vector in the SEZ frame whose origin
    % is at a tracking station with coordinates (latitude,longitude) on Earth
    % to the ECEF frame.
    Validated 02/08/16 with example 7-1 in Vallado book.
    """
    R_ECEF = np.dot(np.dot(GF.fnRotate3(-longitude),GF.fnRotate2(-(0.5*math.pi - latitude_gd))),R_SEZ);
    return R_ECEF
    
def fnSEZtoECIobsv(latitude_gd,longitude,theta_gmst):
    """
    Based on fnSEZtoECEF and fnECEFtoECI
    
    Transformation matrix needed for Mixed Coordinates Filtering.
    
    Created: 15/11/16.
    Refer to pg 802 Vallado. 
    """
    SEZ_ECEF = np.dot(GF.fnRotate3(-longitude),GF.fnRotate2(-(0.5*math.pi - latitude_gd)));
    ECEF_ECI = GF.fnRotate3(-theta_gmst);
    SEZ_ECI = np.dot(SEZ_ECEF,ECEF_ECI);
    return SEZ_ECI;

def fnSEZtoRadar(R_SEZ):
    """
    fnSEZtoRadar is the reverse of fnRadartoSEZ.
    This functions converts observations in the Cartesian SEZ coordinate frame to
    the spherical coordinate system.
    Date: 10 September 2016
    Reference: Vallado, 4th ed.
    Validated with example 7-1 as the reverse of fnRadartoSEZ.
    """
    rho = np.linalg.norm(R_SEZ);
    s = np.linalg.norm(R_SEZ[0:2]);
    el = math.atan(R_SEZ[2]/s);
    az = math.atan2(R_SEZ[0],R_SEZ[1]) + 0.5*math.pi;
    return rho,az,el

def fnECEFtoSEZ(R_ECEF,latitude_gd,longitude):
    """
    fnECEFtoSEZ is the reverse of fnSEZtoECEF.
    Date: 10 September 2016
    Reference: Vallado, 4th ed.
    Validated 10/09/16 with example 7-1 in Vallado book as the reverse of fnSEZtoECEF
    """
    R_SEZ = np.dot(np.dot(GF.fnRotate2((0.5*math.pi - latitude_gd)),GF.fnRotate3(longitude)),R_ECEF);
    return R_SEZ

def fnRAZEL(latitude_gd,longitude,altitude,R_ECI,theta):
    """
    fnRAZEL implements the function RAZEL: Algorithm  27 in Vallado, 4th ed.
    Created: 10 September 2016
    RAZEL converts a position vector in the ECI frame with theta representing the time tage
    into a sensor-centered rho,az,el vector based in an SEZ frame.
    Validated 10/09/16 with example 7-1 in Vallado 4th ed.
    """
    R_ECEF = fnECItoECEF(R_ECI,theta);
    Radar_ECEF =fnRadarSite(latitude_gd,longitude,altitude);
    rho_ECEF = R_ECEF - Radar_ECEF;
    R_SEZ =fnECEFtoSEZ(rho_ECEF,latitude_gd,longitude);
    rho,az,el = fnSEZtoRadar(R_SEZ);
    return rho,az,el
    
def fnRAZEL_Cartesian(latitude_gd,longitude,altitude,R_ECI,theta):
    """
    Cartesian version of fnRAZEL.
    """
    R_ECEF = fnECItoECEF(R_ECI,theta);
    Radar_ECEF =fnRadarSite(latitude_gd,longitude,altitude);
    #~ print R_ECEF
    #print Radar_ECEF
    rho_ECEF = R_ECEF - Radar_ECEF;
    #print rho_ECEF # 16 oct
    R_SEZ =fnECEFtoSEZ(rho_ECEF,latitude_gd,longitude);
    return R_SEZ

def fnVel_ECI_to_SEZ(V_ECI,R_ECEF,latitude_gd,longitude,theta):
    """
    Transform velocity vector from ECI to SEZ
    Date: 06 January 2017
    Edited: 15/01/17: changed name from fnVel_Cartesian
    """
    V_ECEF = fnECItoECEF_vel(V_ECI,theta,R_ECEF);
    V_SEZ = fnECEFtoSEZ(V_ECEF,latitude_gd,longitude);
    return V_SEZ # Validated on 15/01/17 as reverse as fnVel_SEZ_to_ECI
    
def fnVel_SEZ_to_ECI(V_SEZ,R_ECI,latitude_gd,longitude,theta):
    """
    Transform velocity vector from SEZ to ECI.
    Date: 15 January 2017
    """
    V_ECEF = fnSEZtoECEF( V_SEZ,latitude_gd,longitude );
    V_ECI = fnECEFtoECI_vel(V_ECEF,theta,R_ECI)
    return V_ECI # validated 15/01/17 with example 7-1 on pg430 in Vallado.

def fnSEZtoECIobsv_vel(latitude_gd,longitude,theta_gmst):
    """
    Transformation matrix needed for Mixed Coordinates Filtering.
    
    Created: 18 January 2017
    Refer to function fnSEZtoECIobsv  
    """
    SEZ_ECEF = np.dot(GF.fnRotate3(-longitude),GF.fnRotate2(-(0.5*math.pi - latitude_gd)));
    ECEF_ECI = GF.fnRotate3(-theta_gmst);
    SEZ_ECI = np.dot(SEZ_ECEF,ECEF_ECI);
    """
    omega = np.array([0.,0.,AstCnst.theta_dot]);
    ECI = np.dot(GF.fnRotate3(-theta),ECEF_vel) + np.cross(omega,R_ECI);
    """
    return SEZ_ECI;

# ----------------------------------------------------------------------------------------- #
## Astrodynamics conversion functions.
def fnKeplerEquation(M,e,errTol):
    """
    Iteratively solve Kepler's equation
    Date: 13 September 2016 (validated 13.09.16)
    """
    E0 = M;
    #t=1;
    while True:
        E = M + e*math.sin(math.radians(E0));
        if abs(E - E0) < errTol:
            #t = 0;
            break;
        E0 = E;
    return E
    # Edited: 28 September 2016: use math instead of np for non-array operations.

def fnTLEtoKeps(line1,line2):
    """
    fnTLEtoKeps extracts Keplerian elements from the two lines of
    a TLE file.
    (validated 13.09.16)
    Date: 13 September 2016
    """
    l1 = line1.split();
    l2 = line2.split();
    epoch = l1[3];
    i = float(l2[2]);
    BigOmega = float(l2[3]);
    e = float(l2[4])/float(1.0e7);
    omega = float(l2[5]);
    M = float(l2[6]);
    n = float(l2[7]);

    a = (AstCnst.mu_E/(n*2*math.pi/(24*3600))**2)**(1.0/3);
    E = fnKeplerEquation(M,e,1e-10); # can use np.finfo(np.float64).eps for tolerance
    nu = 2*math.atan2(math.sqrt(1-e)*math.cos(E/2),math.sqrt(1+e)*math.sin(E/2));
    
    return a,e,i,BigOmega,omega,E,nu,epoch # edited 28 September 2016
    # Edited: 28 September 2016: use math instead of np for non-array operations.




# -------------------------------------------------------------------------------------- #
# Convert Cartesian elements to Keplerians
def fnCartsToKeps(x):
    """
    fnCartsToKeps converts a Cartesian ECI state vector to Keplerians.
    Reference: Orbital mechanics for engineering students. Curtis.
    Date: 13 September 2016 (validated 13.09.16)
    """
    R = x[0:3]; V = x[3:6];
    
    r = np.linalg.norm(R);
    v = np.linalg.norm(V);
    
    vr = np.inner(R,V)/r;
    
    # Specific angular momentum
    H = np.cross(R,V);
    h = np.linalg.norm(H);
    
    # inclination
    i = math.acos(H[2]/h);
    N = np.cross(np.array([0.0,0.0,1.0]),H);
    n = np.linalg.norm(N);
    
    if n != 0:
        BigOmega = math.acos(N[0]/n); # aka RA right ascencsion
        if N[1] < 0:
            BigOmega = 2*math.pi - BigOmega;
    else:
        BigOmega = 0;
    
    E = (1/AstCnst.mu_E)*((v**2 - AstCnst.mu_E/r)*R - r*vr*V); # eccentric anomaly
    e = np.linalg.norm(E); # eccentricity
    
    if n != 0:
        if e > np.finfo(float).eps:
            # argument of perigee
            omega = math.acos(np.inner(N,E)/n/e);
            if E[2] < 0:
                omega = 2*math.pi - omega;
        else:
            omega = 0;
    else:
        omega = 0;
            
    if e > np.finfo(float).eps:
        # true anomaly
        TA = math.acos(np.inner(E,R)/e/r);
        if vr<0:
            TA = - TA;# want TA between -pi and pi
    else:
        cp = np.cross(N,R);
        if cp[2] >= 0:
            TA = math.acos(np.inner(N,R)/n/r);
        else:
            TA = 2*math.pi - math.acos(np.inner(N,R)/n/r);
    a = h**2/AstCnst.mu_E/(1-e**2);
    return a, i, e, BigOmega, omega, TA
            
            
def fnCartsToKeps_Vallado(x):
    """
    Function to convert Cartesians to Keplerians
    Based on: Vallado, Fundamentals of Astrodynamics, 4th ed.
    Date: 14 September 2016
    (validated 14.09.16)
    """
    h = np.cross(x[0:3],x[3:6]);
    evec = np.cross(x[3:6],h)/AstCnst.mu_E - x[0:3]/np.linalg.norm(x[0:3]);
    n = np.cross(np.array([0.0,0.0,1.0]),h);
    
    true_anom = math.acos(np.inner(evec,x[0:3])/(np.linalg.norm(evec)*np.linalg.norm(x[0:3])));
    if np.inner(x[0:3],x[3:6]) < 0:
        #~ print 'test'
        true_anom = - true_anom;
    
    i = math.acos(h[2]/np.linalg.norm(h));
    e = np.linalg.norm(evec);
    
    E = 2*math.atan(math.tan(true_anom/2.0)/math.sqrt((1+e)/(1-e)));
    
    # RAAN
    BigOmega = math.acos(n[0]/np.linalg.norm(n));
    if n[1] < 0:
        BigOmega = 2*math.pi - BigOmega;
        
    omega =math.acos(np.inner(n,evec)/(np.linalg.norm(evec)*np.linalg.norm(n)));
    if evec[2] < 0:
        omega = 2*math.pi - omega;
        
    #M = E - e*math.sin(E);
    
    a = 1/((2/np.linalg.norm(x[0:3])) - np.linalg.norm(x[3:6])**2/AstCnst.mu_E);
    return a,e,i,BigOmega,omega,E,true_anom

# fnCartsToKeps and fnCartsToKEpsVallado should give identical results.

# ------------------------------------------------------------------------------- #
## Covariance matrix transformation functions
def fnCov_SEZ_to_ECEF(R_SEZ,longitude,latitude_gd):
    """
    Transform the covariance matrix from SEZ to ECEF frame.
    Date: 14 September 2016
    """
    Trans = np.dot(GF.fnRotate3(-longitude),GF.fnRotate2(-(0.5*math.pi - latitude_gd)));
    R_sez_ecef = np.dot(Trans,np.dot(R_SEZ,Trans.T));
    return R_sez_ecef

def fnCov_ECEF_to_SEZ(R_ECEF,longitude,latitude_gd):
    """
    Transform the covariance matrix from ECEF to SEZ frame.
    Date: 14 September 2016
    """
    Trans = np.dot(GF.fnRotate2(0.5*math.pi - latitude_gd),GF.fnRotate3(longitude));
    R_ecef_sez = np.dot(Trans,np.dot(R_ECEF,Trans.T));
    return R_ecef_sez

def fnCov_ECEF_to_ECI(R_ECEF,theta):
    """
    Transform the covariance matrix from ECEF to ECI
    Date: 14 September 2016
    """
    Trans = GF.fnRotate3(-theta);
    R_ecef_eci = np.dot(Trans,np.dot(R_ECEF,Trans.T));
    return R_ecef_eci
    
def fnCov_ECI_to_ECEF(R_ECI,theta):
    """
    Transform the covariance matrix from ECI to ECEF
    Date: 14 September 2016
    """
    Trans = GF.fnRotate3(theta);
    R_eci_ecef = np.dot(Trans,np.dot(R_ECI,Trans.T));
    return R_eci_ecef
# ----------------------------------------------------------------------------------------- #
## 12 Nov 2016. Functions for track initiation
# See testherrickgibbs.py and testherrickgibbs.m for description and validation.


def fnHerrickGibbs(PosVec1,PosVec2,PosVec3,JD1,JD2,JD3):
    """
    Python implementation of fnHerrickGibbs.m
    
    % fnHerrickGibbs.m uses the Herrick-Gibbs method of preliminary orbit determination to compute
    % a velocity vector from 3 position vectors.

    % Implements Algorithm 55 in [Ref 1]. Based on hgibbs.m from Vallado,
    % updated for speed of execution.

    %% Parameters:
    % PosVec1,2,3 are three position vectors in the ECI frame.
    % VelVec2 is the velocity vector computed for the second measurement. 
    % errorFlag indicates whether the three position vectors are coplanar.
    % 0 if yes. 1 if no. [Ref 1]

    %% References
    % 1. Fundamentals of astrodynamics and application. Vallado. 2013
    
    Note that JD1, JD2, JD3 are fractions of a day.
    
    Date: 12 November 2016
    """
    #%% Calculations:
    #% Find the time steps
    dt21= (JD2-JD1)*86400.0;
    dt31= (JD3-JD1)*86400.0;   
    dt32= (JD3-JD2)*86400.0;

    #% Find the magnitudes of the 3 vectors
    magr1 = np.linalg.norm(PosVec1);
    magr2 = np.linalg.norm(PosVec2);
    magr3 = np.linalg.norm(PosVec3);
    
    tolangle= 0.01745329251994;
    
    #% Find cross-product
    Z23 = np.cross(PosVec2,PosVec3);
    
    dotProd = np.dot(Z23,PosVec1)/(np.linalg.norm(Z23)*np.linalg.norm(PosVec1));
    if np.absolute( dotProd ) > 0.017452406:
        errorFlag = 1;
    else:
        errorFlag = 0;
       
    theta = GF.fnAngle(PosVec1, PosVec2, True);
    theta1 = GF.fnAngle(PosVec2, PosVec3, True);
    
    print 'Coplanar angles are: '
    print math.degrees(theta)
    print math.degrees(theta1)
    print 'Below 1 [deg], HG is superior. Above 5[deg], Gibbs is superior. There is a cross-over point in between'
    
    if theta > tolangle or theta1 > tolangle:
        errorFlag = 1;
    
    term1= -dt32*( 1.0/(dt21*dt31) + AstCnst.mu_E/(12.0*magr1**3) );
    term2= (dt32-dt21)*( 1.0/(dt21*dt32) + AstCnst.mu_E/(12.0*magr2**3) );
    term3=  dt21*( 1.0/(dt32*dt31) + AstCnst.mu_E/(12.0*magr3**3) );

    VelVec2 =  term1*PosVec1 + term2* PosVec2 + term3* PosVec3;
    return VelVec2,errorFlag # Validated with example 7-4 in the Vallado book (2013).


# --------------------------------------------------------------------------------------------------------------------------- #
def fnCarts_to_LatLon(R):
    """
    function which converts ECEF position vectors to latitude and longitude
    Based on rvtolatlong.m in Richard Rieber's orbital library on mathwork.com
    
    Note that this is only suitable for groundtrack visualization, not rigorous 
    calculations.
    Date: 18 September 2016
    Edit: 6 December 2016: moved to AstroFunctions folder.
    """
    r_delta = np.linalg.norm(R[0:2]); # 30.01.17
    sinA = R[1]/r_delta;
    cosA = R[0]/r_delta;

    Lon = math.atan2(sinA,cosA);

    if Lon < -math.pi:
        Lon = Lon + 2*math.pi;

    Lat = math.asin(R[2]/np.linalg.norm(R));
    return Lat,Lon
# ------------------------------------------------------------------------- #
def fnCalc_ApogeePerigee(a,e):
    """
    Calculate the apogee and perigee of a space-object
    based on the semi-major axis a and the eccentricity e.
    Created: 07/04/17
	Added here: 11/04/17
    """
    perigee = a*(1.-e);
    apogee = a*(1.+e);
    return perigee, apogee # Validated in testgabbard.py


    


# --------------------------------------------------------------------------------- #
## 29 May 2017
def fnConvert_AZEL_to_Topo_RADEC(elevation, azimuth,latitude,longitude,theta):
	"""
	Convert azimuth and elevation to right ascension declination in the topocentric frame.
	
	NOTE: latitude and longitude should be in [rad] and not in [deg]!!!!!!!!!!!
	Also, latitude is latitude_gc
    30/05/17: Note that the azimuth angle here is defined from North to South. [clockwise]
    Azimuth in UCM.py is defined from South to North [anticlockwise]
	
	Based on Vallado book algorithm AZELTORADEC [The page number changed from edition to edition.]
	Validated using example 3-7 in vallado book
	Created on: 29 May 2017
	"""
	sin_declination = math.sin(elevation)*math.sin(latitude) + math.cos(elevation)*math.cos(latitude)*math.cos(azimuth);
	declination = math.asin(sin_declination);
	sin_LHA = -math.sin(azimuth)*math.cos(elevation)*math.cos(latitude)/(math.cos(declination)*math.cos(latitude));
	cos_LHA = (math.sin(elevation) - math.sin(latitude)*math.sin(declination))/(math.cos(declination)*math.cos(latitude));
	local_hour_angle = math.atan2(sin_LHA,cos_LHA);
	local_sidereal_time = longitude + theta; # theta == theta_GMST = GMST angle in [rad]
	right_ascension = GF.fnZeroTo2Pi(local_sidereal_time - local_hour_angle);
	return right_ascension, declination # Validated in test_radec.py on 29/05/17

def fnConvert_Topo_RADEC_to_AZEL(right_ascension,declination,latitude,longitude,theta):
	"""
	Convert topocentric right ascension and declination to azimuth and elevation.
	
	This function reverses the operation of fnConvert_AZEL_to_Topo_RADEC
	
	NOTE: latitude and longitude should be in [rad] and not in [deg]!!!!!!!!!!!
	Also, latitude is latitude_gc
     30/05/17: Note that the azimuth angle here is defined from North to South. [clockwise]
     Azimuth in UCM.py is defined from South to North [anticlockwise]
	
	Based on Vallado book algorithm AZELTORADEC [The page number changed from edition to edition.]
	Validated using example 3-7 in vallado book
	Created on: 29 May 2017
	"""
	local_sidereal_time = longitude + theta; # theta == theta_GMST = GMST angle in [rad]
	local_hour_angle = local_sidereal_time - right_ascension;
	sin_elevation = math.sin(latitude)*math.sin(declination) + math.cos(latitude)*math.cos(declination)*math.cos(local_hour_angle);
	elevation = math.asin(sin_elevation);
	sin_azimuth = - (math.sin(local_hour_angle)*math.cos(declination)*math.cos(latitude))/(math.cos(elevation)*math.cos(latitude));
	cos_azimuth = (math.sin(declination) - math.sin(elevation)*math.sin(latitude))/(math.cos(elevation)*math.cos(latitude));
	azimuth = math.atan2(sin_azimuth,cos_azimuth);	
	return elevation,GF.fnZeroTo2Pi(azimuth)# Validated in test_radec.py on 29/05/17

"""
after validating with Vallado book, need to make sure to flip the required signs for azimuth
because my azimuth angle is defined in a different way
"""
