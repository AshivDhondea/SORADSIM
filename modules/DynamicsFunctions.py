"""
DynamicsFunctions.py

Description:
A collection of functions which implement dynamics-related functions for orbiting objects.

Created by: Ashiv Dhondea, RRSG, UCT.
Date created: 20 June 2016
Edits:
21/06/2016 : cleaned up array indices in fnKepler_J2_augmented.
23/06/16 : fixed function fnKepler_J2_augmented. Validated against its MATLAB counterpart.
23/06/16: added in 3 functions for polynomial models. (i.e. kinematic models).
7 July 2016: added function to generate nominal trajectory for Two-Body problem.
14 July 2016: added function to implement differential equations for the mean and covariance matrix. Integrated by RK4 for the MC-RK4 method for CD-EKF.
15 July 2016: edited yesterday's function.
16 July 2016: edited the function fnMoment_DE to be integrated by fnRK4_moment(f, dt, x,t,L,Q)
19 July 2016: edited the function fn_Create_Concatenated_Block_Diag_Matrix. Probably a better implementation.
22 July 2016: removed the function fn_Create_Concatenated_Block_Diag_Matrix and moved it to MathsFunctions.py
22 July 2016: cleaned up all functions so that code is reused. import MathsFunctions for Cholesky decomposition of positive semidefinite matrices.
24 July 2016: created the function fn_CWPA_Discretized_Covariance_Matrix(dt,intensity,acc) which generates the discretized process noise covariance matrix for NCAM-CWPA models.
15 September 2016: added function to retrodict the estimated state vector and estimated error covariance matrix
14 October 2016: edited the function which retrodicts the state vector.
24 February 2017: created the function fnGenerate_Nominal_Trajectory_RK4
24 February 2017: added the functions fnGenerate_Nominal_Trajectory_RK2 and fnGenerate_Nominal_Trajectory_RK1
References
1. Tracking Filter Engineering, Norman Morrison. 2013
2. Estimation with applications to tracking and navigation, Bar Shalom, Li, Kirubarajan. 2001
"""
# ------------------------- #
# Import libraries
import numpy as np
#from numpy import linalg
import scipy
import math
# Ashiv's own libraries
import AstroConstants as AstCnst
import Num_Integ as ni
import MathsFunctions as MathsFn
# -------------------------------------------------------------------- #
def fnKepler_J2(t,X):
    # emulates fnKepler_J2.m. 20 June 2016
    # fnKepler_J2 implements the dynamics function of the J-2 perturbed two-body problem.
    r = np.linalg.norm(X[0:3]);
    Xdot = np.zeros([6],dtype=np.float64);
    Xdot[0:3] = X[3:6]; # dx/dt = xdot; dy/dt = ydot; dz/dt = zdot
    expr = 1.5*AstCnst.J_2*(AstCnst.R_E/r)**2 ;

    Xdot[3] = -(AstCnst.mu_E/r**3)*X[0]*(1-expr*(5*(X[2]/r)**2 -1));
    Xdot[4] = -(AstCnst.mu_E/r**3)*X[1]*(1-expr*(5*(X[2]/r)**2 -1));
    Xdot[5] = -(AstCnst.mu_E/r**3)*X[2]*(1-expr*(5*(X[2]/r)**2 -3));
    return Xdot

def fnKepler_J2_augmented(t,X):
    # validated against the equivalent MATLAB function on 23 June 2016.
    # fnKepler_J2_augmented implements the dynamics function of the J-2 perturbed two-body problem
    # augmented by the state transition matrix(STM) evaluated on the state vector X.
    
    # state vector augmented by reshaped state transition matrix
    Xdot = np.zeros([6 + 6*6],dtype=np.float64);
    Xdot[0:6] = fnKepler_J2(t,X);
    # The state transition matrix's elements are the last 36 elements in the input state vector
    Phi = np.reshape(X[6:6+6*6+1],(6,6)); # extract Phi matrix from the input vector X
    
    # Find matrix A (state sensitivity matrix)
    Amatrix = fnJacobian_Kepler_J2(X);
    # The state transition matrix's differential equation.
    PhiDot = np.dot(Amatrix,Phi); # Amatrix is a time dependent variable.i.e. not constant
    Xdot[6:6+6*6+1] = np.reshape(PhiDot,6*6);
    return Xdot

def fnGenerate_Nominal_Trajectory(Xdash,timevec):
    # fnGenerate_Nominal_Trajectory generates a nominal trajectory Xstate over the time period timevec
    # starting from Xdash.
    # Dynamical model assumed: fnKepler_J_2_augmented.
    # This function is required when the Gauss-Newton filter operates in iterative mode.
    dt = timevec[1]-timevec[0];
    Xstate = np.zeros([6+6*6,len(timevec)],dtype=np.float64);
    Xstate[0:6,0] = Xdash;
    Xstate[6:,0] = np.reshape(np.eye(6,dtype=np.float64),6*6);
    
    for index in range(1,len(timevec)):
        # Perform RK4 numerical integration on the J2-perturbed dynamics.
        Xstate[:,index] = ni.fnRK4_vector(fnKepler_J2_augmented,dt,Xstate[:,index-1],timevec[index-1]);
    return Xstate
# --------------------------------------------------------------------------------------------#
## Polynomial model functions
def fn_Generate_STM_polynom(zeta,nStates):
    # fn_Generate_STM_polynom creates the state transition matrix for polynomial models 
    # of degree (nStates-1) over a span of transition of zeta [s].
    # Polynomial models are a subset of the class of constant-coefficient linear DEs.
    # Refer to: Tracking Filter Engineering, Norman Morrison.
    stm = np.eye(nStates,dtype=np.float64);
    for yindex in range (0,nStates):
        for xindex in range (yindex,nStates): # STM is upper triangular
            stm[yindex,xindex] = np.power(zeta,xindex-yindex)/float(math.factorial(xindex-yindex));
    return stm;     

def fn_Generate_STM_polynom_3D(zeta,nStates,dimensionality):
    # fn_Generate_STM_polynom_3D generates the full state transition matrix for 
    # the required dimensionality.
    stm = fn_Generate_STM_polynom(zeta,nStates);
    stm3 = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(stm,dimensionality-1);
    return stm3;

def fn_CWPA_Discretized_Covariance_Matrix(dt,intensity,acc):
    # fn_CWPA_Discretized_Covariance_Matrix generates the discretized covariance matrix in the
    # Continuous Wiener Process Acceleration version of the Nearly Constant Acceleration Model (NCAM).
    # 24 July 2016.
    # Find continuous-time process covariance matrix, then discretize it.
    qtilde = intensity*acc/np.sqrt(dt); # Choice of process noise intensity.
    # The theory behind Discretized Continuous-time kinematic models is discussed thoroughly
    # in Bar-Shalom, Rong Li, Kirubarajan: Estimation with applications to tracking and navigation.
    Qc00 = np.power(dt,5)/20.0; # See discussion in section 6.2.3
    Qc01 = np.power(dt,4)/8.0;
    Qc02 = np.power(dt,3)/6.0;
    Qc21 = np.square(dt)/2.0;
    Qc = np.array([[Qc00,Qc01,Qc02],[Qc01,Qc02*2,Qc21],[Qc02,Qc21,dt]],dtype=np.float64);
    Qd = np.diag(qtilde); # process noise covariance matrix for the 1D case
    Qd3 = np.kron(Qd,Qc); # Generate the corresponding matrix for the 3D case
    return Qd3
# -------------------------------------------------------------------------------------------------------------- #
## fnMoment_DE implements the DEs for propagation of the mean and covariance.
    """ Based on "Various ways to compute the continuous-discrete extended Kalman filter"
     @article{frogerais2012various,
      title={Various ways to compute the continuous-discrete extended Kalman filter},
      author={Frogerais, Paul and Bellanger, Jean-Jacques and Senhadji, Lotfi},
      journal={Automatic Control, IEEE Transactions on},
      volume={57},
      number={4},
      pages={1000--1004},
      year={2012},
      publisher={IEEE}
    } """
def fnMoment_DE(t,X,Q,L):
    # 14 July 2016: moment propagation differential equations. Based on Frogerais 2012.
    # Function called by numerical integration method called by CD-EKF MC-RK4 (fnCD_EKF_predict_MC_RK4)
    # Edit: 28/07/16: added Q and L to the arguments list.

    # t is the time.
    # X is the mean state vector augmented by the reshaped covariance matrix.
    # L is the dispersion matrix.
    # Q is the process noise covariance matrix.
        
    # state vector
    Xdot = np.zeros([6 + 6*6],dtype=np.float64);
    Xdot[0:6] = fnKepler_J2(t,X);
    # Extract covariance matrix from state vector.
    P = np.reshape(X[6:],(6,6));
    
    # Find matrix A (state sensitivity matrix)
    Amatrix = fnJacobian_Kepler_J2(X);
    # Covariance matrix propagation equation.
    P = np.dot(Amatrix,P) + np.dot(P,np.transpose(Amatrix)) + np.dot(np.dot(L,Q),np.transpose(L));
    
    # to force symmetry for covariance matrix P.
    P = 0.5*(P + np.transpose(P));
    # and positive definiteness
    S,definite = MathsFn.schol(P);
    P = np.dot(S,S.T);

    # Pop covariance matrix into state vector
    Xdot[6:] = np.reshape(P,6*6);
    return Xdot;

def fnJacobian_Kepler_J2(X):
    # fnJacobian_Kepler_J2 evaluates the Jacobian matrix of function fnKepler_J2 at X.
    
    r = np.linalg.norm(X[0:3]);
    expr = 1.5*AstCnst.J_2*(AstCnst.R_E/r)**2 ;
    expr1 = -(AstCnst.mu_E/r**3)*(1 - expr*(5*((X[2]/r)**2) -1));
    # Find matrix A (state sensitivity matrix)
    # We start with a matrix of zeros and then add in the non-zero elements.
    Amatrix = np.zeros([6,6],dtype=np.float64);
    Amatrix[0,3]=1.0;
    Amatrix[1,4]=1.0;
    Amatrix[2,5]=1.0;

    expr2 = 3*(AstCnst.mu_E/r**5)*(1 - 2.5*AstCnst.J_2*(AstCnst.R_E/r)**2*(7*(X[2]/r)**2 -1)); # fixed 14 Oct 2016
    Amatrix[3,0] = expr1 + expr2*X[0]**2; 
    Amatrix[3,1] = expr2*X[0]*X[1]; 
    expr3 =  3*(AstCnst.mu_E/r**5)*(1 - 2.5*AstCnst.J_2*(AstCnst.R_E/r)**2*(7*(X[2]/r)**2 -3)); # fixed 14 Oct 2016
    Amatrix[3,2] = expr3*X[0]*X[2];
    
    Amatrix[4,0] = Amatrix[3,1];
    Amatrix[4,1] = expr1 + expr2*X[1]**2;
    Amatrix[4,2] = expr3*X[1]*X[2];

    Amatrix[5,0] = Amatrix[3,2];
    Amatrix[5,1] = Amatrix[4,2];
    Amatrix[5,2] =-(AstCnst.mu_E/r**3)*(1-expr*(5*(X[2]/r)**2 -3)) + 3*(AstCnst.mu_E/r**5)*(1 - 2.5*AstCnst.J_2*(AstCnst.R_E/r)**2*(7*(X[2]/r)**2) -5)*X[2]**2;
    return Amatrix

# ------------------------------------------------------------------------------------------------------------ #
def fnRetrodict_State_Vec(Xnom,timevec,fnNomiTraj,S_hat):
    """
    Retrodicts the state vector and covariance matrix
    
    Date: 15 September 2016
    Edited: 14 October 2016
    """
    timevector = np.flipud(timevec);
    
    xout = fnNomiTraj(Xnom,timevector);
    
    # Extract STM for the time interval of interest
    dx = np.shape(S_hat)[0];
    Sout = np.zeros([dx*dx,len(timevec)],dtype=np.float64);
    for i in range (0,len(timevec)):
        stm = np.reshape(xout[dx:,i],(dx,dx));
        sout = np.dot(stm,np.dot(S_hat,stm.T));
        Sout[:,i] = np.reshape(sout,dx*dx);
        
    # Output should be in chronological order
    xout_sorted = np.zeros_like(xout);
    
    for index in range(0,dx + dx*dx):
        xout_sorted[index,:] = np.fliplr([xout[index,:]])[0]; 
    
     # Output should be in chronological order
    Sout_sorted = np.zeros_like(Sout);
    
    for index in range(0,dx*dx):
        Sout_sorted[index,:] = np.fliplr([Sout[index,:]])[0];  
            
    return xout_sorted,Sout_sorted

# ------------------------------------------------------------------------------------------------------------------------------------------------------- #
## 24/02/17
def fnGenerate_Nominal_Trajectory_RK4(Xdash,timevec,L,fnF,fnA):
    """
    Generate the nominal trajectory with an embedeed Runge-Kutta 4th order scheme.
    
    Author: Ashiv Dhondea
    Created: 21 February 2017
    Edited: 
    23/02/17: debugged
    24/02/17: moved to DynamicsFunctions
    Originally named fntestprop in testgnf.py
    """
    dx = np.shape(Xdash)[0];
    delta_t = timevec[1] - timevec[0];
    Xnom = np.zeros([dx,len(timevec)],dtype=np.float64);
    Xnom[:,0] = Xdash;
    stm = np.zeros([dx,dx,len(timevec)],dtype=np.float64);
    iden = np.eye(dx,dtype=np.float64);
    stm[:,:,0] = iden;
    for index in range(1,L):
        k1 = fnF(timevec[index],Xnom[:,index-1]);
        k2 = fnF(timevec[index]+0.5*delta_t,Xnom[:,index-1]+0.5*k1*delta_t);
        k3 = fnF(timevec[index]+0.5*delta_t,Xnom[:,index-1]+0.5*k2*delta_t);
        k4 = fnF(timevec[index]+delta_t,Xnom[:,index-1]+k3*delta_t);
        Xnom[:,index] = Xnom[:,index-1] + (delta_t/6.)*(k1+2.*k2+2.*k3+k4);
        J1x = fnA(Xnom[:,index-1]);
        J2x = np.dot(fnA(Xnom[:,index-1]+0.5*k1*delta_t), np.add(iden,0.5*delta_t*J1x) );
        J3x = np.dot(fnA(Xnom[:,index-1]+0.5*k2*delta_t), np.add(iden,0.5*delta_t*J2x) );
        J4x = np.dot(fnA(Xnom[:,index-1]+    k3*delta_t), np.add(iden,    delta_t*J3x) );
        stm[:,:,index] = np.dot(np.add(iden, (delta_t/6.)*np.add(J1x,np.add(2.*J2x,np.add(2.*J3x,J4x)))),stm[:,:,index-1]);
    return Xnom,stm
    
def fnGenerate_Nominal_Trajectory_RK1(Xdash,timevec,L,fnF,fnA):
    """
    Generate the nominal trajectory with an embedeed Runge-Kutta 1st order scheme, i.e., an Euler-scheme.
    
    Author: Ashiv Dhondea
    Created: 24 February 2017
    Edited: 
    """
    dx = np.shape(Xdash)[0];
    delta_t = timevec[1] - timevec[0];
    Xnom = np.zeros([dx,len(timevec)],dtype=np.float64);
    Xnom[:,0] = Xdash;
    stm = np.zeros([dx,dx,len(timevec)],dtype=np.float64);
    iden = np.eye(dx,dtype=np.float64);
    stm[:,:,0] = iden;
    for index in range(1,L):
        k1 = fnF(timevec[index],Xnom[:,index-1]);
        Xnom[:,index] = Xnom[:,index-1] + (delta_t/1.)*(k1);
        J1x = fnA(Xnom[:,index-1]);
        stm[:,:,index] = np.dot(np.add(iden, (delta_t/1.)*J1x),stm[:,:,index-1]);
    return Xnom,stm
    
def fnGenerate_Nominal_Trajectory_RK2(Xdash,timevec,L,fnF,fnA):
    """
    Generate the nominal trajectory with an embedeed Runge-Kutta 2nd order scheme.
    
    Author: Ashiv Dhondea
    Created: 21 February 2017
    Edited: 
    """
    dx = np.shape(Xdash)[0];
    delta_t = timevec[1] - timevec[0];
    Xnom = np.zeros([dx,len(timevec)],dtype=np.float64);
    Xnom[:,0] = Xdash;
    stm = np.zeros([dx,dx,len(timevec)],dtype=np.float64);
    iden = np.eye(dx,dtype=np.float64);
    stm[:,:,0] = iden;
    for index in range(1,L):
        k1 = fnF(timevec[index],Xnom[:,index-1]);
        k2 = fnF(timevec[index]+delta_t,Xnom[:,index-1]+k1*delta_t);
        Xnom[:,index] = Xnom[:,index-1] + (delta_t/2.)*(k1+k2);
        J1x = fnA(Xnom[:,index-1]);
        J2x = np.dot(fnA(Xnom[:,index-1]+k1*delta_t), np.add(iden,delta_t*J1x) );
        stm[:,:,index] = np.dot(np.add(iden, (delta_t/2.)*np.add(J1x,J2x)),stm[:,:,index-1]);
    return Xnom,stm
