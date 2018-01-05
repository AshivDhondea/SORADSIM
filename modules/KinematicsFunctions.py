"""

KinematicsFunctions.py

Kinematics functions for Kinematic Models such as CVM, CAM and NCVM, NCAM.

References: 
1. Tracking Filter Engineering, Norman Morrison, 2013.
2. Estimation with applications to tracking and navigation. Bar Shalom, Li, Kirubarajan, 2001.

Author: Ashiv Dhondea, RRSG, UCT.
Date: 12/12/16
Edited: 13/12/16 : changed the function which creates the process noise covariance matrix for CWPA
"""
# ------------------------- #
# Import libraries
import numpy as np
import scipy
import math
# Ashiv's own libraries
import MathsFunctions as MathsFn
# --------------------------------------------------------------------------------------------#
## Polynomial model functions
def fn_Generate_STM_polynom(zeta,nStates):
    """
    fn_Generate_STM_polynom creates the state transition matrix for polynomial models 
    of degree (nStates-1) over a span of transition of zeta [s].
    Polynomial models are a subset of the class of constant-coefficient linear DEs.
    Refer to: Tracking Filter Engineering, Norman Morrison.
    """
    stm = np.eye(nStates,dtype=np.float64);
    for yindex in range (0,nStates):
        for xindex in range (yindex,nStates): # STM is upper triangular
            stm[yindex,xindex] = np.power(zeta,xindex-yindex)/float(math.factorial(xindex-yindex));
    return stm;     

def fn_Generate_STM_polynom_3D(zeta,nStates,dimensionality):
    """
    fn_Generate_STM_polynom_3D generates the full state transition matrix for 
    the required dimensionality.
    """
    stm = fn_Generate_STM_polynom(zeta,nStates);
    stm3 = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(stm,dimensionality-1);
    return stm3;

def fn_CWPA_Discretized_Covariance_Matrix(delta_t,intensity,acc):
    """
    fn_CWPA_Discretized_Covariance_Matrix generates the discretized covariance matrix in the
    Continuous Wiener Process Acceleration version of the Nearly Constant Acceleration Model (NCAM).
    Created: 24 July 2016.
    Find continuous-time process covariance matrix, then discretize it.
    qtilde = intensity*acc/np.sqrt(dt); # Choice of process noise intensity.
    The theory behind Discretized Continuous-time kinematic models is discussed thoroughly
    in Bar-Shalom, Rong Li, Kirubarajan: Estimation with applications to tracking and navigation.
    
    Edited: 13/12/16
    intensity: intensity of process noise
    """
    # Find continuous-time process covariance matrix, then discretize it.
    qtilde = intensity*acc/math.sqrt(delta_t);
    Qc00 = (delta_t**5)/20.; # See discussion in section 6.2.3
    Qc01 = (delta_t**4)/8.;
    Qc02 = (delta_t**3)/6.;
    Qc21 = (delta_t**2)/2.;
    Qc = np.array([[Qc00,Qc01,Qc02],[Qc01,Qc02*2,Qc21],[Qc02,Qc21,delta_t]],dtype=np.float64);
    Qd_x = Qc*qtilde[0];
    Qd_y = Qc*qtilde[1];
    Qd_z = Qc*qtilde[2];
    Qd3 = scipy.linalg.block_diag(Qd_x,Qd_y,Qd_z);
    return Qd3

#~ def fn_DWNA_Covariance_Matrix(dt,variance):
    #~ """
    #~ Discrete White Noise Acceleration model
    
    #~ The covariance matrix of the process noise. eqn 6.3.2-4 in Bar-Shalom 2001.
    
    #~ Nearly Constant Velocity Model (NCVM).
    
    #~ Created: 13/12/16
    #~ """
    #~ Qd = np.zeros([2,2],dtype=np.float64);
    #~ Qd[0,0] = 0.25*dt**4;
    #~ Qd[0,1] = 0.5*dt**3;
    #~ Qd[1,0] = 0.5*dt**3;
    #~ Qd[1,1] = dt**2;
    #~ Qd = Qd*variance;
    #~ return Qd #13.12.16 to be continued.
