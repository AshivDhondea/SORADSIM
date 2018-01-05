"""

Unbiased Converted Measurements.py

Reference:
1. Bar Shalom, Longbin 1998: Unbiased converted measurements
2. Javier Areta note on Spherical to Cartesian measurement converion with MATLAB code.
3. S. Bordonaro thesis 2015 on Converted Measurement Kalman Filtering.

Author: Ashiv Dhondea, RRSG, UCT.
Date: 24 September 2016.
Edited: 25 September 2016: added comments.
        03 October 2016: added functions for a different definition of the angles and axes. 
        13 November 2016: added the UCM3D covariance transformation matrix fnCalculate_UCM3D_Cov_SEZ_Rp
        14 November 2016: added the observation sensitivity matrix function fnJacobianH
        08 December 2016: added the functions fnCalculate_Spherical_Bistatic and fnJacobianH_Spherical_Bistatic for Bistatic Radar measurements.
        09 December 2016: added the function fnCalculate_Bistatic_Spherical to convert from bistatic spherical to monostatic spherical.
        18 December 2016: fixed a calculus mistake in the function fnJacobianH_Spherical_Bistatic
        19 December 2016: fixed a sign error in the function fnJacobianH_Spherical_Bistatic
        21 December 2016: edited the function fnCalculate_Spherical_Bistatic, fnJacobianH_Spherical_Bistatic
        27 December 2016: moved the functions relating to bistatic radars to BistaticAndDoppler.py
"""
import math 
import numpy as np

#~ import AstroFunctions as AstFn

def fnCalculate_UCM3D(Xinput,R_spherical):
    """
    fnCalculate_UCM3D transforms the (range,elevation,azimuth) measurement of a target's position 
    into (x,y,z) Cartesian position coordinates using the  UCM
    (Unbiased Converted Measurement) method of Longbin, Bar-Shalom. Since the original data is
    in spherical coordinates, the term '3D' is appended to the name.
    
    Based on: Unbiased converted measurements for tracking. Longbin, Bar-Shalom 1998
    
    Good explanation in: S. Bordonaro PhD thesis (2015) "Converted measurement trackers
    for systems with nonlinear measurement functions"
    
    Refer to: fnHinv for the conventional converted measurement function
    Created: 25 August 2016
    """
    # Xinput[0] = range
    # Xinput[1] = elevation
    # Xinput[2] = azimuth
    # R_spherical: measurement covariance matrix in spherical coordinates
    
    l_el_1 = math.exp(0.5*R_spherical[1,1]);
    l_az_1 = math.exp(0.5*R_spherical[2,2]);
    
    Xout = np.zeros([3],dtype=np.float64);
    Xout[0] = l_az_1*l_el_1*Xinput[0]*math.cos(Xinput[1])*math.sin(Xinput[2]); # x
    Xout[1] = l_az_1*l_el_1*Xinput[0]*math.cos(Xinput[1])*math.cos(Xinput[2]); # y
    Xout[2] =        l_el_1*Xinput[0]*math.sin(Xinput[1]); # z
    return Xout
    
def fnCalculate_UCM3D_Cov(Xinput,R_spherical):
    """
    Unbiased spherical to Cartesian covariance matrix. 
    Xinput_spherical is the position vector in the spherical coordinate system, 
    consisting of (range,elevation,azimuth) in this order.
    R_spherical is the covariance matrix in spherical coordinates.
    Created: 25 August 2016
    Edited: 08 September 2016
    Based on MATLAB code in Coordinate conversion from spherical to Cartesian. Javier Areta, UCONN
    which is based on "Unbiased converted measurements for tracking" Longbin, Bar-Shalom.
    """
    r = Xinput[0];
    e = Xinput[1];
    B = Xinput[2];
    
    sr = math.sqrt(R_spherical[0,0]);
    se = math.sqrt(R_spherical[1,1]);
    sb = math.sqrt(R_spherical[2,2]);
    
    le = math.exp(-0.5*se**2);
    lep = math.exp(-2*se**2);
    lB = math.exp(-0.5*sb**2);
    lBp = math.exp(-2*sb**2);

    R = np.zeros([3,3],dtype=np.float64);
    R[1,1]=0.25*lB**(-2)*le**(-2)*(r**2+2*sr**2)*(1+lBp**2*math.cos(2*B))*(1+lep**2*math.cos(2*e))-0.25*(r**2+sr**2)*(1+lBp*math.cos(2*B))*(1+lep*math.cos(2*e));

    R[0,0]=0.25*lB**(-2)*le**(-2)*(r**2+2*sr**2)*(1-lBp**2*math.cos(2*B))*(1+lep**2*math.cos(2*e))-0.25*(r**2+sr**2)*(1-lBp*math.cos(2*B))*(1+lep*math.cos(2*e));

    R[2,2]=0.5* le**(-2)*(r**2+2*sr**2)*(1-lep**2*math.cos(2*e))-1/2*(r**2+sr**2)*(1-lep*math.cos(2*e));

    R[0,1]=0.25*lB**(-2)*le**(-2)*lBp**2*(r**2+2*sr**2)*math.sin(2*B)*(1+lep**2*math.cos(2*e))-0.25*(r**2+sr**2)* lBp*math.sin(2*B) *(1+lep*math.cos(2*e));

    R[1,2]=0.5*lB *le**(-2)*(r**2+2*sr**2)*math.cos(B)*lep**2*math.sin(2*e)-0.5*(r**2+sr**2)* lB *math.cos(B) * lep*math.sin(2*e);

    R[0,2]=0.5*lB *le**(-2)*(r**2+2*sr**2)*math.sin(B)*lep**2*math.sin(2*e)-0.5*(r**2+sr**2)* lB *math.sin(B) * lep*math.sin(2*e);

    # and for symmetry
    R[1,0] = R[0,1];
    R[2,0] = R[0,2];
    R[2,1] = R[1,2];
    return R

# --------------------------------------------------------------------------------------------------------------- #
def fnCalculate_UCM3D_SEZ(Xinput,R_spherical):
    """
    Based on fnCalculate_UCM3D 
    Created: 03 October 2016
    Based on MATLAB code in Coordinate conversion from spherical to Cartesian. Javier Areta, UCONN
    which is based on "Unbiased converted measurements for tracking" Longbin, Bar-Shalom.
    """
    # Xinput[0] = range
    # Xinput[1] = elevation
    # Xinput[2] = azimuth
    # R_spherical: measurement covariance matrix in spherical coordinates
    
    l_el_1 = math.exp(0.5*R_spherical[1,1]);
    l_az_1 = math.exp(0.5*R_spherical[2,2]);
    
    Xout = np.zeros([3],dtype=np.float64);
    Xout[0] = l_az_1*l_el_1*Xinput[0]*math.cos(Xinput[1])*math.cos(Xinput[2]); # x
    Xout[1] = l_az_1*l_el_1*Xinput[0]*math.cos(Xinput[1])*math.sin(Xinput[2]); # y
    Xout[2] =        l_el_1*Xinput[0]*math.sin(Xinput[1]); # z
    return Xout

def fnCalculate_Spherical(Xinput):
    """
    Calculate spherical coordinates of target.
    should be the inverse of fnCalculate_UCM3D_SEZ.
    Date: 10 October 2016
    """
    # Sensor measures range and look angles.
    Xinput_sph = np.zeros([3],dtype=np.float64);
    # Calculate range
    Xinput_sph[0] = np.linalg.norm(Xinput);
    # Calculate elevation
    Xinput_sph[1] = math.atan(Xinput[2]/np.linalg.norm(Xinput[0:2]));
    #Xinput_sph[1] = AstFn.fnZeroToPi(Xinput_sph[1]); # 13/11/16: ensure angle is wrapped.
    # Calculate azimuth
    Xinput_sph[2] = math.atan2(Xinput[1],Xinput[0]);
    #Xinput_sph[2] = AstFn.fnZeroTo2Pi(Xinput_sph[2]); # 13/11/16: ensure angle is wrapped.
    return Xinput_sph    

## The Jacobian matrix for the inverse transformation is fnJacobianH
def fnJacobianH(Xnom):
    """
    Observation sensitivity matrix.
    Used for mixed coordinates filtering. (Case 4 in Gauss-Newton parlance).
    refer to page 803 in Vallado textbook.
    
    This function pertains to the transformation in fnCalculate_Spherical in UCM code.
    
    Date: 14 November 2016
    """
    # Jacobian of nonlinear measurement function fnH
    # Xnom[0] = range
    # Xnom[1] = elevation
    # Xnom[2] = azimuth
    rho = np.linalg.norm(Xnom);
    s = np.linalg.norm(Xnom[0:2]);
    Mmatrix = np.zeros([3,3],dtype=np.float64);
    Mmatrix[0,0] = Xnom[0]/rho; 
    Mmatrix[0,1] = Xnom[1]/rho; 
    Mmatrix[0,2] = Xnom[2]/rho; 

    Mmatrix[1,0] = -Xnom[0]*Xnom[2]/(s*rho**2);
    Mmatrix[1,1] = -Xnom[1]*Xnom[2]/(s*rho**2); 
    Mmatrix[1,2] = s/(rho**2);

    Mmatrix[2,0] = -Xnom[1]/s**2;
    Mmatrix[2,1] = Xnom[0]/s**2;           
    return Mmatrix
       
    
def fnCalculate_UCM3D_Cov_SEZ(Xinput,R_spherical):
    """
    Created: 03 October 2016
    Based on MATLAB code in Coordinate conversion from spherical to Cartesian. Javier Areta, UCONN
    which is based on "Unbiased converted measurements for tracking" Longbin, Bar-Shalom.
    """
    r = Xinput[0];
    e = Xinput[1];
    B = Xinput[2];
    
    sr = math.sqrt(R_spherical[0,0]);
    se = math.sqrt(R_spherical[1,1]);
    sb = math.sqrt(R_spherical[2,2]);
    
    le = math.exp(-0.5*se**2);
    lep = math.exp(-2*se**2);
    lB = math.exp(-0.5*sb**2);
    lBp = math.exp(-2*sb**2);

    R = np.zeros([3,3],dtype=np.float64);
    R[0,0]=0.25*lB**(-2)*le**(-2)*(r**2+2*sr**2)*(1+lBp**2*math.cos(2*B))*(1+lep**2*math.cos(2*e))-0.25*(r**2+sr**2)*(1+lBp*math.cos(2*B))*(1+lep*math.cos(2*e));

    R[1,1]=0.25*lB**(-2)*le**(-2)*(r**2+2*sr**2)*(1-lBp**2*math.cos(2*B))*(1+lep**2*math.cos(2*e))-0.25*(r**2+sr**2)*(1-lBp*math.cos(2*B))*(1+lep*math.cos(2*e));

    R[2,2]=0.5* le**(-2)*(r**2+2*sr**2)*(1-lep**2*math.cos(2*e))-1/2*(r**2+sr**2)*(1-lep*math.cos(2*e));

    R[0,1]=0.25*lB**(-2)*le**(-2)*lBp**2*(r**2+2*sr**2)*math.sin(2*B)*(1+lep**2*math.cos(2*e))-0.25*(r**2+sr**2)* lBp*math.sin(2*B) *(1+lep*math.cos(2*e));

    R[0,2]=0.5*lB *le**(-2)*(r**2+2*sr**2)*math.cos(B)*lep**2*math.sin(2*e)-0.5*(r**2+sr**2)* lB *math.cos(B) * lep*math.sin(2*e);

    R[1,2]=0.5*lB *le**(-2)*(r**2+2*sr**2)*math.sin(B)*lep**2*math.sin(2*e)-0.5*(r**2+sr**2)* lB *math.sin(B) * lep*math.sin(2*e);

    # and for symmetry
    R[1,0] = R[0,1];
    R[2,0] = R[0,2];
    R[2,1] = R[1,2];
    return R


def fnCalculate_UCM3D_Cov_SEZ_Rp(Xinput,R_spherical):
    """
    Created: 13 November 2016
    Based on "Unbiased converted measurements for tracking" Longbin, Bar-Shalom.
    
    fnCalculate_UCM3D_Cov_SEZ implements the Rm variant of the UCM3D covariance matrix.
    
    """
    r = Xinput[0];
    e = Xinput[1];
    B = Xinput[2];
    
    sr = math.sqrt(R_spherical[0,0]);
    se = math.sqrt(R_spherical[1,1]);
    sb = math.sqrt(R_spherical[2,2]);
    
    le = math.exp(-0.5*se**2);
    lep = math.exp(-2*se**2);
    lB = math.exp(-0.5*sb**2);
    lBp = math.exp(-2*sb**2);

    R = np.zeros([3,3],dtype=np.float64);
    R[0,0] = ((lB*le)**(-2)-2)*(r**2 * math.cos(B)**2 * math.cos(e)**2) + 0.25*(r**2+sr**2)*(1+lBp*math.cos(2*B))*(1+lep*math.cos(2*e));

    R[1,1] = ((lB*le)**(-2)-2)*(r**2 * math.sin(B)**2 * math.cos(e)**2) + 0.25*(r**2+sr**2)*(1-lBp*math.cos(2*B))*(1+lep*math.cos(2*e));

    R[2,2] = (le**(-2) - 2)*(r**2 * math.sin(e)**2) + 0.5*(r**2+sr**2)*(1-lep*math.cos(2*e));

    R[0,1] =((lB*le)**(-2)-2)*r**2*math.sin(B)*math.cos(B)*math.cos(e)**2 +0.25*(r**2+sr**2)* lBp*math.sin(2*B) *(1+lep*math.cos(2*e));

    R[0,2] = ((lB**(-1))*(le**-(2)) - lB**(-1) -lB)*(r**2)*math.cos(B)*math.sin(e)*math.cos(e) + 0.5*(r**2+sr**2)* lB *math.cos(B) * lep*math.sin(2*e);

    R[1,2] = ((lB**(-1))*(le**(-2)) - lB**(-1) -lB)*r**2*math.sin(B)*math.sin(e)*math.cos(e) + 0.5*(r**2+sr**2)* lB *math.sin(B) * lep*math.sin(2*e);

    # and for symmetry
    R[1,0] = R[0,1];
    R[2,0] = R[0,2];
    R[2,1] = R[1,2];
    return R 
    
"""
Note on  UCM3D Spherical Measurements to Cartesian Conversion of Covariance Matrix

fnCalculate_UCM3D_Cov_SEZ_Rp and fnCalculate_UCM3D_Cov_SEZ both implement covariance transformation matrices 
from the Longbin (1998) paper.

As noted in this paper, the Rp matrix shows better consistency with the conversion errors than the
Rm matrix. In my particular application, Rp gives me much better Cartesian measurements which
are adequate for a Gauss-Newton filter.

"""
