"""
GNF.py

Gauss-Newton filter functions

Author: Ashiv Dhondea, RRSG, UCT.
Date: 12 September 2016
Edited: 
14 September 2016 : debugged the old functions.
15 September 2016: make use of the maths function invSymQuadForm.
9 November 2016: added function for GNF in expanding memory mode.
14 November 2016: cleaned up GNF in expanding memory mode.
15 November 2016: added functions for Case 4: Mixed coordinates filtering. expanding memory mode and batch mode.
23 November 2016: added the function fnGoodness_of_Fit_test to check for goodness of fit
24 November 2016: added the function to calculate thresholds for GOF test.
25 November 2016: edited the GNF Case 3 batch fast function for Goodness of Fit test
16 December 2016: created the function fnGaussNewtonBatch_Case2_BistaticRangeAndAngles for Kinematic tracking based on bistatic range and angles measurements.
19 December 2016: created the function fnGaussNewtonBatch_Case2_MonostaticRangeAndDoppler for Kinematic tracking based on monostatic range and Doppler measurements.
26 December 2016: edited the function fnGaussNewtonBatch_Case2_MonostaticRangeAndDoppler as a result of changes done in BistaticAndDoppler.py
28 December 2016: created the function fnGaussNewtonBatch_Case2_BistaticRangeAndDoppler. works fine.
17 January 2017: The spherical bistatic Jacobian function was moved from UCM to BD, hence updated to reflect this change.
23 January 2017: sorted out the Jacobian function in fnGaussNewtonBatch_Case4_monostatic_rangedopp and included the measurement function.
26 January 2017: created the function fnGaussNewtonBatch_Case4_bistatic_rangedopp
24 February 2017: created the function fnGaussNewtonBatch_Case3_RK
24 February 2017: created the function fnGaussNewtonBatch_Case4_RK

28/09/17: created the function fnGaussNewtonBatch_Case4_RK_OD_bistatic_rangedopp_simple
References:
1. Tracking Filter Engineering, Norman Morrison 2013.
"""
## Import general stuff
import numpy as np
import math
## Import my stuff
import AstroFunctions as AstFn
import BistaticAndDoppler as BD
import MathsFunctions as MathsFn
import UnbiasedConvertedMeasurements as UCM
# -------------------------------------------------------------------------------- #
## Gauss-Newton filter -- Batch mode i.e. growing memory mode.
def fnMVA(fim,TotalObsvMatT,Ryn,delta_Y):
    """
    Implements the Minimum Variance Algorithm from Morrison 2013
    
    Created: 14 September 2016 based on previous code.
    Edited: 11 October 2016 : forgot to invert Ryn. sorted it.
    
    fim : Fisher Information Matrix
    TotalObsvMatT : Total Observation Matrix Transpose.
    Ryn : Total Measurement Covariance matrix.
    delta_Y : Total observation perturbation vector
    """
    A = fim;
    b = np.dot(TotalObsvMatT,np.linalg.inv(Ryn));
    W = np.linalg.solve(A,b);
    S_hat = np.dot(np.dot(W,Ryn),np.transpose(W)); # error covariance matrix
    delta_X_hat = np.dot(W,delta_Y); # estimated perturbation vector.
    return delta_X_hat, S_hat

def fnGaussNewtonBatch_Case3(Xdash,timevec,L,fnNomiTraj,M,Xradar,Ryn):
    """
    Created: 15 September 2016 based on previous code.
    
    Xdash : initial state vector (estimated)
    timevec : time vector
    L : filter window length
    M : observation matrix
    Xradar : measurement vector
    fim : Fisher Information Matrix
    TotalObsvMatT : Total Observation Matrix Transpose.
    Ryn : Total Measurement Covariance matrix.
    delta_Y : Total observation perturbation vector
    """
    dy = np.shape(M)[0];
    dx = np.shape(M)[1];
    # From initial estimate of state vector xdash, generate a nominal trajectory over L 
    Xnom = fnNomiTraj(Xdash,timevec[0:L]);
    
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    
    for index in range(0,L):
        # state transition matrix
        stm = np.reshape(Xnom[dx:,index],(dx,dx));
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm));
        TotalObsvVec[dy*index:dy*index+dy] = np.dot(M,Xnom[0:dx,index]);
        delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy];
        
    fim = MathsFn.invSymQuadForm(TotalObsvMat,Ryn);
    delta_X_hat,S_hat = fnMVA(fim,TotalObsvMatT,Ryn,delta_Y);
    Xdash = Xnom[0:dx,L-1] + delta_X_hat; # edit: 13 November 2016

    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = MathsFn.invSymQuadForm(delta_Y,Ryn);#np.dot(np.transpose(delta_Y),np.dot(RynInv,delta_Y));
    
    return Xdash,S_hat,Ji,delta_X_hat # edit: 29 October 2016


def fnGaussNewtonFilter_Case3(Xdash,timevec,L,fnNomiTraj,M,Xradar,Ryn):
    """
    Created: 15 September 2016 based on previous code.
    
    Xdash : initial state vector (estimated)
    timevec : time vector
    L : filter window length
    M : observation matrix
    Xradar : measurement vector
    fim : Fisher Information Matrix
    TotalObsvMatT : Total Observation Matrix Transpose.
    Ryn : Total Measurement Covariance matrix.
    delta_Y : Total observation perturbation vector
    """
    dy = np.shape(M)[0];
    dx = np.shape(M)[1];
    # From initial estimate of state vector xdash, generate a nominal trajectory over L 
    Xnom = fnNomiTraj(Xdash,timevec);
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    
    # estimates
    x_hat = np.zeros([dx,len(timevec)],dtype=np.float64);
    S_hat = np.zeros([dx,dx,len(timevec)],dtype=np.float64);
    
    # initialize
    index = 0;
    # state transition matrix
    stm = np.reshape(Xnom[dx:,index],(dx,dx));
    TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm);
    TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm));
    TotalObsvVec[dy*index:dy*index+dy] = np.dot(M,Xnom[0:dx,index]);
    delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy];
    
    for index in range(1,len(timevec)):
        # state transition matrix
        stm = np.reshape(Xnom[dx:,index],(dx,dx));
        if index < L:
            TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm);
            TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm));
            TotalObsvVec[dy*index:dy*index+dy] = np.dot(M,Xnom[0:dx,index]);
            delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy];
        
            fim = MathsFn.invSymQuadForm(TotalObsvMat[0:dy*index+dy,:],Ryn[0:dy*index+dy,0:dy*index+dy]);
            delta_X_hat,covmat = fnMVA(fim,TotalObsvMatT[:,0:dy*index+dy],Ryn[0:dy*index+dy,0:dy*index+dy],delta_Y[0:dy*index+dy]);
            Xdash = Xnom[0:dx,index] + delta_X_hat;
        else:
            # Forget outdated data
            # Perform circular shift to get rid of old data
            print 'Cycling the filter...';
            # Replace the oldest sample by new data
            TotalObsvMat[dy*0:dy*0+dy,:] = np.dot(M,stm);
            TotalObsvMatT[:,dy*0:dy*0+dy] = np.transpose(np.dot(M,stm));
            TotalObsvVec[dy*0:dy*0+dy] = np.dot(M,Xnom[0:dx,0]);
            delta_Y[dy*0:dy*0+dy] = Xradar[:,index] - TotalObsvVec[dy*0:dy*0+dy];
            # and reset the matrices
            TotalObsvMat = np.roll(TotalObsvMat,-3,axis=0);
            TotalObsvMatT = np.roll(TotalObsvMatT,-3,axis=1);
            TotalObsvVec = np.roll(TotalObsvVec,-3);
            
            fim = MathsFn.invSymQuadForm(TotalObsvMat,Ryn[dy*index+dy - L*dy:dy*index+dy,dy*index+dy - L*dy:dy*index+dy]);
            delta_X_hat,covmat = fnMVA(fim,TotalObsvMatT,Ryn[dy*index+dy -L*dy:dy*index+dy,dy*index+dy -L*dy:dy*index+dy],delta_Y);
            Xdash = Xnom[0:dx,index] + delta_X_hat;
            
        x_hat[:,index] = Xdash;
        covmat = 0.5*np.dot(covmat,covmat.T);
        S,defi = MathsFn.schol(covmat);
        S_hat[:,:,index] = np.dot(S,S.T);
    return x_hat,S_hat


def fnGNF_ExpandingMemoryFiltering(nominal_trajectory,timevec,fnNomiTraj,M,y_eci,R_eci):
    """
    Nonlinear Gauss-Newton filter (Case 3).
    Expanding Memory mode.
    
    nominal_trajectory : initial state vector (estimated)
    timevec : time vector
    L : filter window length
    M : observation matrix
    y_eci : measurement vectors
    R_eci : the measurement covariance matrices associated with the measurement vectors
    fim : Fisher Information Matrix
    TotalObsvMatT : Total Observation Matrix Transpose.
    Ryn : Total Measurement Covariance matrix.
    delta_Y : Total observation perturbation vector
    
    Author: AshivD
    Date: 09 November 2016
    Edited: 14 November 2016: cleaned up code.
    """
    L = len(timevec);
    dy = np.shape(M)[0];
    dx = np.shape(M)[1];
    # From initial estimate of state vector xdash, generate a nominal trajectory over L 
    Xnom = fnNomiTraj(nominal_trajectory,timevec[0:L+1]);
    X_hat = np.zeros([dx,L],dtype=np.float64);
    index = 0;
    X_hat[:,index] = nominal_trajectory;
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    Ryn = MathsFn.fnStack_Block_Diag(R_eci[:,:,:],L);
    S_hat = np.zeros([dx,dx,L],dtype=np.float64);
    Ji = np.zeros([L],dtype=np.float64);
    
    stm = np.reshape(Xnom[dx:,index],(dx,dx));
    TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm);
    TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm));
    TotalObsvVec[dy*index:dy*index+dy] = np.dot(M,Xnom[0:dx,index]);
    delta_Y[dy*index:dy*index+dy] = y_eci[:,index] - TotalObsvVec[dy*index:dy*index+dy];
    
    for index in range(1,L):
        stm = np.reshape(Xnom[dx:,index],(dx,dx));
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm));
        TotalObsvVec[dy*index:dy*index+dy] = np.dot(M,Xnom[0:dx,index]);
        delta_Y[dy*index:dy*index+dy] = y_eci[:,index] - TotalObsvVec[dy*index:dy*index+dy];
    
        fim = MathsFn.invSymQuadForm(TotalObsvMat[0:dy*index+dy,:],Ryn[0:dy*index+dy,0:dy*index+dy]);
        delta_X_hat,S_hat[:,:,index] = fnMVA(fim,TotalObsvMatT[:,0:dy*index+dy],Ryn[0:dy*index+dy,0:dy*index+dy],delta_Y[0:dy*index+dy]);
        X_hat[:,index] = Xnom[0:dx,index] + delta_X_hat;
        # Cost Ji of cost function. (Crassidis pg 28)
        Ji[index] = MathsFn.invSymQuadForm(delta_Y[0:dy*index+dy],Ryn[0:dy*index+dy,0:dy*index+dy]);
    return X_hat,S_hat,Ji
# ---------------------------------------------------------------------------------------------------------------------------------------- #
def fnGaussNewtonBatch_Case3_fast(Xdash,timevec,L,fnNomiTraj,M,Xradar,Ryn):
    """
    Created: 14 November 2016 
    Edited: 23,24,25 November 2016 for Goodness of fit test
    Faster implementation of fnGaussNewtonBatch_Case3 because we do not make use of invSymQuadForm 
    which calls the very slow schol function.
    
    Xdash : initial state vector (estimated)
    timevec : time vector
    L : filter window length
    M : observation matrix
    Xradar : measurement vector
    fim : Fisher Information Matrix
    TotalObsvMatT : Total Observation Matrix Transpose.
    Ryn : Total Measurement Covariance matrix.
    delta_Y : Total observation perturbation vector
    """
    dy = np.shape(M)[0];
    dx = np.shape(M)[1];
    # From initial estimate of state vector xdash, generate a nominal trajectory over L 
    Xnom = fnNomiTraj(Xdash,timevec[0:L]);
    
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    
    for index in range(0,L):
        # state transition matrix
        stm = np.reshape(Xnom[dx:,index],(dx,dx));
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm));
        TotalObsvVec[dy*index:dy*index+dy] = np.dot(M,Xnom[0:dx,index]);
        delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy];
        
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.linalg.pinv(np.dot(TotalObsvMatT,np.dot(RynInv,TotalObsvMat)));
    delta_X_hat = np.dot(S_hat,np.dot(TotalObsvMatT,np.dot(RynInv,delta_Y)));
    
    Xdash = Xnom[0:dx,L-1] + delta_X_hat; 
    nominal_trajectory = Xnom[0:dx,0] + delta_X_hat;

    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = MathsFn.invSymQuadForm(delta_Y,Ryn);#np.dot(np.transpose(delta_Y),np.dot(RynInv,delta_Y));
    
    # 23.11.16 test for goodness of fit.
    if np.shape(TotalObsvVec)[0] > np.shape(nominal_trajectory)[0] + dx:
        fit = fnGoodness_of_Fit_test(nominal_trajectory,TotalObsvVec,TotalObsvMat,RynInv);
    else:
        fit = True

    return Xdash,S_hat,Ji,delta_X_hat,fit 

def fnGNF_ExpandingMemoryFiltering_fast(nominal_trajectory,timevec,fnNomiTraj,M,y_eci,R_eci):
    """
    Nonlinear Gauss-Newton filter (Case 3).
    Expanding Memory mode.
    
    FASTER VERSION!!!!! 14/11/16.
    
    nominal_trajectory : initial state vector (estimated)
    timevec : time vector
    L : filter window length
    M : observation matrix
    y_eci : measurement vectors
    R_eci : the measurement covariance matrices associated with the measurement vectors
    fim : Fisher Information Matrix
    TotalObsvMatT : Total Observation Matrix Transpose.
    Ryn : Total Measurement Covariance matrix.
    delta_Y : Total observation perturbation vector
    
    Author: AshivD
    Date: 14 November 2016
    """
    L = len(timevec);
    dy = np.shape(M)[0];
    dx = np.shape(M)[1];
    # From initial estimate of state vector xdash, generate a nominal trajectory over L 
    Xnom = fnNomiTraj(nominal_trajectory,timevec[0:L+1]);
    X_hat = np.zeros([dx,L],dtype=np.float64);
    index = 0;
    X_hat[:,index] = nominal_trajectory;
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    Ryn = MathsFn.fnStack_Block_Diag(R_eci[:,:,:],L);
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.zeros([dx,dx,L],dtype=np.float64);
    Ji = np.zeros([L],dtype=np.float64);
    
    stm = np.reshape(Xnom[dx:,index],(dx,dx));
    TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm);
    TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm));
    TotalObsvVec[dy*index:dy*index+dy] = np.dot(M,Xnom[0:dx,index]);
    delta_Y[dy*index:dy*index+dy] = y_eci[:,index] - TotalObsvVec[dy*index:dy*index+dy];
    
    for index in range(1,L):
        stm = np.reshape(Xnom[dx:,index],(dx,dx));
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm));
        TotalObsvVec[dy*index:dy*index+dy] = np.dot(M,Xnom[0:dx,index]);
        delta_Y[dy*index:dy*index+dy] = y_eci[:,index] - TotalObsvVec[dy*index:dy*index+dy];
        
        S_hat[:,:,index] = np.linalg.pinv(np.dot(TotalObsvMatT[:,0:dy*index+dy],np.dot(RynInv[0:dy*index+dy,0:dy*index+dy],TotalObsvMat[0:dy*index+dy,:])));
        delta_X_hat = np.dot(S_hat[:,:,index],np.dot(TotalObsvMatT[:,0:dy*index+dy],np.dot(RynInv[0:dy*index+dy,0:dy*index+dy],delta_Y[0:dy*index+dy])));
        X_hat[:,index] = Xnom[0:dx,index] + delta_X_hat;
        
        # Cost Ji of cost function. (Crassidis pg 28)
        Ji[index] = MathsFn.invSymQuadForm(delta_Y[0:dy*index+dy],Ryn[0:dy*index+dy,0:dy*index+dy]);
    return X_hat,S_hat,Ji

def fnGaussNewtonBatch_Case4_fast(Xdash,timevec,L,fnNomiTraj,theta_GMST,latitude_gd,longitude,Xradar,R):
    """
    Author: AshivD.
    Created: 15 November 2016 
    Edited: 25 November 2016
    
    Based on: Vallado book.
    Xdash : initial state vector (estimated)
    timevec : time vector
    L : filter window length
    M : observation matrix
    Xradar : measurement vector
    fim : Fisher Information Matrix
    TotalObsvMatT : Total Observation Matrix Transpose.
    Ryn : Total Measurement Covariance matrix.
    delta_Y : Total observation perturbation vector
    """
    dy = np.shape(R)[0];
    dx = np.shape(Xdash)[0];
    # From initial estimate of state vector xdash, generate a nominal trajectory over L 
    Xnom = fnNomiTraj(Xdash,timevec[0:L]);
      
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    Ryn = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1);
    
    for index in range(0,L):
        # state transition matrix
        stm = np.reshape(Xnom[dx:,index],(dx,dx));
        H = UCM.fnJacobianH(Xnom[0:dy,index]); 
        #H = UCM.fnJacobianH_Spherical_Bistatic(baseline,Xnom[dx:,index]);
        SEZtoECI = AstFn.fnSEZtoECIobsv(latitude_gd,longitude,theta_GMST[index]);
        M = np.hstack((np.dot(H,SEZtoECI),np.zeros([3,3],dtype=np.float64)));
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm));
        TotalObsvVec[dy*index:dy*index+dy] = np.dot(M,Xnom[0:dx,index]); # possible bug here, see the algorithm for Case 4 GNF filtering. 27.09.17
        delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy];
    
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.linalg.pinv(np.dot(TotalObsvMatT,np.dot(RynInv,TotalObsvMat)));
    delta_X_hat = np.dot(S_hat,np.dot(TotalObsvMatT,np.dot(RynInv,delta_Y)));
    Xdash = Xnom[0:dx,L-1] + delta_X_hat; 

    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = MathsFn.invSymQuadForm(delta_Y,Ryn);
    
    # 25.11.16 test for goodness of fit.
    if np.shape(TotalObsvVec)[0] > np.shape(Xdash)[0] + dx:
        fit = fnGoodness_of_Fit_test(Xdash,TotalObsvVec,TotalObsvMat,RynInv);
    else:
        fit = True

    return Xdash,S_hat,Ji,delta_X_hat,fit 

def fnGNF_ExpandingMemoryFiltering_Case4(nominal_trajectory,timevec,fnNomiTraj,theta_GMST,latitude_gd,longitude,y,R):
    """
    Nonlinear Gauss-Newton filter (Case 4) for the radar orbit determination problem.
    Expanding Memory mode.
    
    nominal_trajectory : initial state vector (estimated)
    timevec : time vector
    L : filter window length
    M : observation matrix
    y_eci : measurement vectors
    R_eci : the measurement covariance matrices associated with the measurement vectors
    fim : Fisher Information Matrix
    TotalObsvMatT : Total Observation Matrix Transpose.
    Ryn : Total Measurement Covariance matrix.
    delta_Y : Total observation perturbation vector
    
    Author: AshivD
    Date: 14 November 2016
    Edited: 15 November 2016
    """
    L = len(timevec);
    dy = np.shape(R)[0];
    dx = np.shape(nominal_trajectory)[0];
    # From initial estimate of state vector xdash, generate a nominal trajectory over L 
    Xnom = fnNomiTraj(nominal_trajectory,timevec[0:L+1]);
    X_hat = np.zeros([dx,L],dtype=np.float64);
    index = 0;
    X_hat[:,index] = nominal_trajectory;
    # Total observation matrix and its transpose
    H = UCM.fnJacobianH(Xnom[0:dy,index]);
    SEZtoECI = AstFn.fnSEZtoECIobsv(latitude_gd,longitude,theta_GMST[index]);
    M = np.hstack((np.dot(H,SEZtoECI),np.zeros([3,3],dtype=np.float64)));
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    Ryn = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1);
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.zeros([dx,dx,L],dtype=np.float64);
    Ji = np.zeros([L],dtype=np.float64);
    
    stm = np.reshape(Xnom[dx:,index],(dx,dx));
    TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm);
    TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm));
    TotalObsvVec[dy*index:dy*index+dy] = np.dot(M,Xnom[0:dx,index]);
    delta_Y[dy*index:dy*index+dy] = y[:,index] - TotalObsvVec[dy*index:dy*index+dy];
    
    for index in range(1,L):
        stm = np.reshape(Xnom[dx:,index],(dx,dx));
        H = UCM.fnJacobianH(Xnom[0:dy,index]);
        SEZtoECI = AstFn.fnSEZtoECIobsv(latitude_gd,longitude,theta_GMST[index]);
        M = np.hstack((np.dot(H,SEZtoECI),np.zeros([3,3],dtype=np.float64)));
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm));
        TotalObsvVec[dy*index:dy*index+dy] = np.dot(M,Xnom[0:dx,index]);# possible bug here, see the algorithm for Case 4 GNF filtering. 27.09.17
        delta_Y[dy*index:dy*index+dy] = y[:,index] - TotalObsvVec[dy*index:dy*index+dy];
        
        S_hat[:,:,index] = np.linalg.pinv(np.dot(TotalObsvMatT[:,0:dy*index+dy],np.dot(RynInv[0:dy*index+dy,0:dy*index+dy],TotalObsvMat[0:dy*index+dy,:])));
        delta_X_hat = np.dot(S_hat[:,:,index],np.dot(TotalObsvMatT[:,0:dy*index+dy],np.dot(RynInv[0:dy*index+dy,0:dy*index+dy],delta_Y[0:dy*index+dy])));
        X_hat[:,index] = Xnom[0:dx,index] + delta_X_hat;
        
        # Cost Ji of cost function. (Crassidis pg 28)
        Ji[index] = MathsFn.invSymQuadForm(delta_Y[0:dy*index+dy],Ryn[0:dy*index+dy,0:dy*index+dy]);
    return X_hat,S_hat,Ji
    
def fnGoodness_of_Fit_test(x_hat,y,T,RynInv):
    """
    GOF test for Gauss-Newton filtering
    Based on Goodness of Fit test from Tracking Filter Engineering book.
    Created: 23.11.16
    Edited: 24/11/16, 25/11/16: sorted out the dimensions
    """
    # number of degrees of freedom
    k=np.shape(y)[0] - np.shape(x_hat)[0]; # edited: 24/11/16,25/11/16
    threshold = fnGetThreshold(k); # edited: 24/11/16
    diff = y - np.dot(T,x_hat);
    ssr = np.dot(np.transpose(diff),np.dot(RynInv,diff));
    nssr = ssr/k;
    if nssr <= threshold:
        # good fit
        fit = True;
    else:
        fit = False;
    return fit

def fnGetThreshold(k):
    """
    Get Threshold function for Chi squared test.
    Created: 24/11/16
    """
    threshold = 1.0 + 3.29/math.sqrt(k);
    return threshold


def fnGaussNewtonBatch_Case4_fast_bistatic(Xdash,timevec,L,fnNomiTraj,theta_GMST,latitude_gd,longitude,Xradar,R,baseline):
    """
    Author: AshivD.
    Created: 9 December 2016 
    Edited: 
    
    Based on: Vallado book.
    Xdash : initial state vector (estimated)
    timevec : time vector
    L : filter window length
    M : observation matrix
    Xradar : measurement vector
    fim : Fisher Information Matrix
    TotalObsvMatT : Total Observation Matrix Transpose.
    Ryn : Total Measurement Covariance matrix.
    delta_Y : Total observation perturbation vector
    """
    dy = np.shape(R)[0];
    dx = np.shape(Xdash)[0];
    # From initial estimate of state vector xdash, generate a nominal trajectory over L 
    Xnom = fnNomiTraj(Xdash,timevec[0:L]);

    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    Ryn = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1);
    
    for index in range(0,L):
        # state transition matrix
        stm = np.reshape(Xnom[dx:,index],(dx,dx));
        H = UCM.fnJacobianH_Spherical_Bistatic(baseline,Xnom[dx:,index]);
        SEZtoECI = AstFn.fnSEZtoECIobsv(math.radians(latitude_gd),math.radians(longitude),theta_GMST[index]);
        M = np.hstack((np.dot(H,SEZtoECI),np.zeros([3,3],dtype=np.float64)));
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm));
        TotalObsvVec[dy*index:dy*index+dy] = UCM.fnCalculate_Spherical_Bistatic(baseline,Xnom[dx:,index]) #np.dot(M,Xnom[0:dx,index]);
        delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy];
    
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.linalg.pinv(np.dot(TotalObsvMatT,np.dot(RynInv,TotalObsvMat)));
    delta_X_hat = np.dot(S_hat,np.dot(TotalObsvMatT,np.dot(RynInv,delta_Y)));
    Xdash = Xnom[0:dx,L-1] + delta_X_hat; 
    Ji = MathsFn.invSymQuadForm(delta_Y,Ryn);
    if np.shape(TotalObsvVec)[0] > np.shape(Xdash)[0] + dx:
        fit = fnGoodness_of_Fit_test(Xdash,TotalObsvVec,TotalObsvMat,RynInv);
    else:
        fit = True

    return Xdash,S_hat,Ji,delta_X_hat,fit 
# ------------------------------------------------------------------------------------------------------- #
def fnGaussNewtonBatch_Case2_BistaticRangeAndAngles(Xdash,stm,timevec,Xradar,R,pos_rx,pos_tx):
    """
    Kinematic target model (CAM), nonlinear observation model (bistatic range + angles).
    
    Created: 16 December 2016 
    Edited: 18 December 2016: fixed a couple of issues.
    21 December 2016: removed assumptions about location of Tx and Rx.
    17 January 2017: The spherical bistatic Jacobian function was moved from UCM to BD, hence updated to reflect this change.
    """
    L = len(timevec);
    dy = np.shape(Xradar)[0];
    dx = np.shape(stm)[1];
    # Nominal trajectory
    Xnom = np.zeros([np.shape(stm)[0],len(timevec)],dtype=np.float64);
    # Initial state vector. position in [km], velocity in [km/s]
    Xnom[:,0] = Xdash;
    # Simulate target trajectory
    for index in range(1,len(timevec)):
        Xnom[:,index] = np.dot(stm,Xnom[:,index-1]);
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    Ryn = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1);
    H = np.zeros([dy,dx],dtype=np.float64);
    for index in range(0,L):
        pos_target = np.array([Xnom[0,index],Xnom[3,index],Xnom[6,index]],dtype=np.float64);
        M = BD.fnJacobianH_Spherical_Bistatic(pos_target,pos_rx,pos_tx); # 21.12.16: this function has changed. 17.01.17: this function was moved from UCM to BD
        H[:,0]=M[:,0];H[:,3]=M[:,1];H[:,6]=M[:,2]; 
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(H,stm);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(H,stm));
        TotalObsvVec[dy*index:dy*index+dy] = UCM.fnCalculate_Spherical_Bistatic(pos_target,pos_rx,pos_tx); # 21.12.16: this function has changed.
        delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy]; # here we should find the smallest difference in angles
    
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.linalg.pinv(np.dot(TotalObsvMatT,np.dot(RynInv,TotalObsvMat)));
    delta_X_hat = np.dot(S_hat,np.dot(TotalObsvMatT,np.dot(RynInv,delta_Y)));
    Xdash = Xnom[:,L-1] + delta_X_hat; 
    nominal_trajectory = Xnom[0:dx,0] + delta_X_hat;
    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = MathsFn.invSymQuadForm(delta_Y,Ryn);
    if np.shape(TotalObsvVec)[0] > np.shape(nominal_trajectory)[0] + dx:
        fit = fnGoodness_of_Fit_test(nominal_trajectory,TotalObsvVec,TotalObsvMat,RynInv);
    else:
        fit = True
    return Xdash,S_hat,Ji,delta_X_hat,fit # 27/12/16: validated.
# ------------------------------------------------------------------------------------------------------------------------ #
def fnGaussNewtonBatch_Case2_MonostaticRangeAndDoppler(Xdash,stm,timevec,Xradar,R,wavelength,pos_rx):
    """
    Kinematic target model (CAM), nonlinear observation model (range + Doppler).
    
    Created: 19 December 2016 
    Edited: 
    26/12/16 : edited this function due to changes in BistaticAndDoppler.py
    """
    L = len(timevec);
    dy = np.shape(Xradar)[0];
    dx = np.shape(stm)[1];
    # Nominal trajectory
    Xnom = np.zeros([np.shape(stm)[0],len(timevec)],dtype=np.float64);
    # Initial state vector. position in [km], velocity in [km/s]
    Xnom[:,0] = Xdash;
    # Simulate target trajectory
    for index in range(1,len(timevec)):
        Xnom[:,index] = np.dot(stm,Xnom[:,index-1]);
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    Ryn = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1);
    H = np.zeros([dy,dx],dtype=np.float64);
    for index in range(0,L):
        pos_target = np.array([Xnom[0,index],Xnom[3,index],Xnom[6,index]],dtype=np.float64);
        vel_target = np.array([Xnom[1,index],Xnom[4,index],Xnom[7,index]],dtype=np.float64);
        x_target = np.hstack((pos_target,vel_target)); # edited: 26/12/16
        M = BD.fnJacobianH_Monostatic_RangeAndDoppler(wavelength,x_target); # edited: 26/12/16
        
        H[:,0]=M[:,0];H[:,3]=M[:,1];H[:,6]=M[:,2]; 
        H[:,1]=M[:,3];H[:,4]=M[:,4];H[:,7]=M[:,5]; 
        
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(H,stm);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(H,stm));
        TotalObsvVec[dy*index:dy*index+dy] = BD.fnCalculate_Monostatic_RangeAndDoppler(wavelength,x_target,pos_rx) # edited: 26/12/16
        
        delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy]; 
    
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.linalg.pinv(np.dot(TotalObsvMatT,np.dot(RynInv,TotalObsvMat)));
    delta_X_hat = np.dot(S_hat,np.dot(TotalObsvMatT,np.dot(RynInv,delta_Y)));
    Xdash = Xnom[:,L-1] + delta_X_hat; 
    nominal_trajectory = Xnom[0:dx,0] + delta_X_hat;
    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = MathsFn.invSymQuadForm(delta_Y,Ryn);
    if np.shape(TotalObsvVec)[0] > np.shape(nominal_trajectory)[0] + dx:
        fit = fnGoodness_of_Fit_test(nominal_trajectory,TotalObsvVec,TotalObsvMat,RynInv);
    else:
        fit = True
    return Xdash,S_hat,Ji,delta_X_hat,fit # 27/12/16: validated.

# ---------------------------------------------------------------------------------------------------------------- #
def fnGaussNewtonBatch_Case2_BistaticRangeAndDoppler(Xdash,stm,timevec,Xradar,R,wavelength,pos_rx,pos_tx):
    """
    Kinematic target model (CAM), nonlinear observation model (bistatic range + Doppler).
    
    Created: 28 December 2016 
    Edited: 
    
    """
    L = len(timevec);
    dy = np.shape(Xradar)[0];
    dx = np.shape(stm)[1];
    # Nominal trajectory
    Xnom = np.zeros([np.shape(stm)[0],len(timevec)],dtype=np.float64);
    # Initial state vector. position in [km], velocity in [km/s]
    Xnom[:,0] = Xdash;
    # Simulate target trajectory
    for index in range(1,len(timevec)):
        Xnom[:,index] = np.dot(stm,Xnom[:,index-1]);
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    Ryn = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1);
    H = np.zeros([dy,dx],dtype=np.float64);
    for index in range(0,L):
        pos_target = np.array([Xnom[0,index],Xnom[3,index],Xnom[6,index]],dtype=np.float64);
        vel_target = np.array([Xnom[1,index],Xnom[4,index],Xnom[7,index]],dtype=np.float64);
        x_target = np.hstack((pos_target,vel_target));
        M = BD.fnJacobianH_Bistatic_RangeAndDoppler(wavelength,x_target,pos_rx,pos_tx)
        
        H[:,0]=M[:,0];H[:,3]=M[:,1];H[:,6]=M[:,2]; 
        H[:,1]=M[:,3];H[:,4]=M[:,4];H[:,7]=M[:,5]; 
        
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(H,stm);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(H,stm));
        TotalObsvVec[dy*index:dy*index+dy] = BD.fnCalculate_Bistatic_RangeAndDoppler(pos_target,vel_target,pos_rx,pos_tx,wavelength);
        
        delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy]; 
    
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.linalg.pinv(np.dot(TotalObsvMatT,np.dot(RynInv,TotalObsvMat)));
    delta_X_hat = np.dot(S_hat,np.dot(TotalObsvMatT,np.dot(RynInv,delta_Y)));
    Xdash = Xnom[:,L-1] + delta_X_hat; 
    nominal_trajectory = Xnom[0:dx,0] + delta_X_hat;
    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = MathsFn.invSymQuadForm(delta_Y,Ryn);
    if np.shape(TotalObsvVec)[0] > np.shape(nominal_trajectory)[0] + dx:
        fit = fnGoodness_of_Fit_test(nominal_trajectory,TotalObsvVec,TotalObsvMat,RynInv);
    else:
        fit = True
    return Xdash,S_hat,Ji,delta_X_hat,fit # 28/12/16: validated.

# ------------------------------------------------------------------------------------------------------------------------------------------------- #
def fnGaussNewtonBatch_Case4_monostatic_rangedopp(Xdash,timevec,L,fnNomiTraj,theta_GMST,Xradar,R,wavelength,radar_ecef):
    """
    Based on fnGaussNewtonBatch_Case4_fast_bistatic
    Assumes monostatic radar measuring range and Doppler shift. For radar orbit determination.
    Author: AshivD.
    Created: 17 January 2017 
    Edited: 
    23/01/17: included the Jacobian function and the measurement function.
    """
    dy = np.shape(R)[0];
    dx = np.shape(Xdash)[0];
    # From initial estimate of state vector xdash, generate a nominal trajectory over L 
    Xnom = fnNomiTraj(Xdash,timevec[0:L]);
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    Ryn = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1);
    
    for index in range(0,L):
        # state transition matrix
        stm = np.reshape(Xnom[dx:,index],(dx,dx));
        H = BD.fnJacobianH_Monostatic_RangeAndDoppler_OD(Xnom[0:dx,index],radar_ecef,theta_GMST[index],wavelength); # 23.01.17
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(H,stm);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(H,stm));
        TotalObsvVec[dy*index:dy*index+dy] = BD.fnCalculate_Monostatic_RangeAndDoppler_OD(Xnom[0:dx,index],radar_ecef,theta_GMST[index],wavelength); # 23/01/17
        delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy];
    
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.linalg.pinv(np.dot(TotalObsvMatT,np.dot(RynInv,TotalObsvMat)));
    delta_X_hat = np.dot(S_hat,np.dot(TotalObsvMatT,np.dot(RynInv,delta_Y)));
    Xdash = Xnom[0:dx,L-1] + delta_X_hat; 

    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = MathsFn.fn_invSymQuadForm(delta_Y,Ryn); # 23.01.17
    return Xdash,S_hat,Ji,delta_X_hat # validated in main_tiangong_monostatic_rangedopp_02.py

def fnGaussNewtonBatch_Case4_bistatic_rangedopp(Xdash,timevec,L,fnNomiTraj,theta_GMST,Xradar,R,wavelength,radar_ecef,tx_ecef):
    """
    Based on fnGaussNewtonBatch_Case4_monostatic_rangedopp
    Assumes bistatic radar measuring range and Doppler shift. For radar orbit determination.
    Author: AshivD.
    Created: 26 January 2017 
    Edited: 
    """
    dy = np.shape(R)[0];
    dx = np.shape(Xdash)[0];
    # From initial estimate of state vector xdash, generate a nominal trajectory over L 
    Xnom = fnNomiTraj(Xdash,timevec[0:L]);
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    Ryn = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1);
    for index in range(0,L):
        # state transition matrix
        stm = np.reshape(Xnom[dx:,index],(dx,dx));
        H = BD.fnJacobianH_Bistatic_RangeAndDoppler_OD(Xnom[0:dx,index],radar_ecef,tx_ecef,theta_GMST[index],wavelength); # 26.01.17
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(H,stm);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(H,stm));
        TotalObsvVec[dy*index:dy*index+dy] = BD.fnCalculate_Bistatic_RangeAndDoppler_OD(Xnom[0:dx,index],radar_ecef,tx_ecef,theta_GMST[index],wavelength);
        delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy];
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.linalg.pinv(np.dot(TotalObsvMatT,np.dot(RynInv,TotalObsvMat)));
    delta_X_hat = np.dot(S_hat,np.dot(TotalObsvMatT,np.dot(RynInv,delta_Y)));
    Xdash = Xnom[0:dx,L-1] + delta_X_hat; 
    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = MathsFn.fn_invSymQuadForm(delta_Y,Ryn); 
    return Xdash,S_hat,Ji,delta_X_hat # validated in main_iss_bistatic_rangedopp_02.py
# --------------------------------------------------------------------------------------------------------------------------------------- #
def fnGaussNewtonBatch_Case3_RK(Xdash,timevec,L,fnNomiTrajSTM,fnF,fnA,M,Xradar,Ryn):
    """
    Efficient implementation of batch GNF based on the recently developed DynFn.fnGenerate_Nominal_Trajectory_RK4
    Created: 24 February 2017
    Edited: 8 March 2017
    """
    dy = np.shape(M)[0];
    dx = np.shape(M)[1];
    # Generate nominal trajectory and STMs.
    Xnom,stm = fnNomiTrajSTM(Xdash,timevec,L,fnF,fnA);
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    #Ryn = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1); # edited: 08 March 2017
    
    for index in range(0,L):
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm[:,:,index]);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm[:,:,index]));
        TotalObsvVec[dy*index:dy*index+dy] = np.dot(M,Xnom[:,index]);
        delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy];
        
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.linalg.pinv(np.dot(TotalObsvMatT,np.dot(RynInv,TotalObsvMat)));
    delta_X_hat = np.dot(S_hat,np.dot(TotalObsvMatT,np.dot(RynInv,delta_Y)));
    
    Xdash = Xnom[:,L-1] + delta_X_hat; 
    #nominal_trajectory = Xnom[:,0] + delta_X_hat; # commented out 9 march 2017

    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = MathsFn.fn_invSymQuadForm(delta_Y,Ryn);
    return Xdash,S_hat,Ji,delta_X_hat

def fnGaussNewtonBatch_Case4_RK(Xdash,timevec,L,fnNomiTrajSTM,fnF,fnA,theta_GMST,latitude_gd,longitude,altitude,Xradar,R):
    """
    Author: AshivD.
    Created: 24 February 2017
    Edited: 
    """
    dy = np.shape(R)[0];
    dx = np.shape(Xdash)[0];
    # Generate nominal trajectory and STMs.
    Xnom,stm = fnNomiTrajSTM(Xdash,timevec,L,fnF,fnA);
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    Ryn = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1);
    
    for index in range(0,L):
        R_ECI = Xnom[0:dy,index];
        R_SEZ = AstFn.fnRAZEL_Cartesian(latitude_gd,longitude,altitude,R_ECI,theta_GMST[index]);
        
        H = UCM.fnJacobianH(R_SEZ); 
        SEZtoECI = AstFn.fnSEZtoECIobsv(latitude_gd,longitude,theta_GMST[index]);
        M = np.hstack((np.dot(H,SEZtoECI),np.zeros([dy,dy],dtype=np.float64)));
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(M,stm[:,:,index]);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(M,stm[:,:,index]));
        
        TotalObsvVec[dy*index:dy*index+dy] = UCM.fnCalculate_Spherical(R_SEZ)# 26.09.17
		#np.dot(M,Xnom[0:dx,index]);
        delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy];
    
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.linalg.pinv(np.dot(TotalObsvMatT,np.dot(RynInv,TotalObsvMat)));
    delta_X_hat = np.dot(S_hat,np.dot(TotalObsvMatT,np.dot(RynInv,delta_Y)));
    Xdash = Xnom[0:dx,L-1] + delta_X_hat; 

    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = MathsFn.fn_invSymQuadForm(delta_Y,Ryn);
    return Xdash,S_hat,Ji,delta_X_hat

# --------------------------------------------------------------------------- #
def fnGaussNewtonBatch_Case4_RK_OD_bistatic_rangedopp(Xdash,timevec,L,fnNomiTrajSTM,fnF,fnA,theta_GMST,Xradar,R,wavelength,radar_ecef,tx_ecef):
    """
    Author: AshivD.
    Created: 21.09.17
    Edited: 
    """
    dy = np.shape(R)[0];
    dx = np.shape(Xdash)[0];
    # Generate nominal trajectory and STMs.
    Xnom,stm = fnNomiTrajSTM(Xdash,timevec,L,fnF,fnA);
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    Ryn = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1);
    
    for index in range(0,L):
        # state transition matrix
        H = BD.fnJacobianH_Bistatic_RangeAndDoppler_OD(Xnom[0:dx,index],radar_ecef,tx_ecef,theta_GMST[index],wavelength);
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(H,stm[:,:,index]);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(H,stm[:,:,index]));
        TotalObsvVec[dy*index:dy*index+dy] = BD.fnCalculate_Bistatic_RangeAndDoppler_OD_fast(Xnom[0:dx,index],radar_ecef,tx_ecef,theta_GMST[index],wavelength);
        #np.dot(H,Xnom[:,index]);
        #BD.fnCalculate_Bistatic_RangeAndDoppler_OD(Xnom[0:dx,index],radar_ecef,tx_ecef,theta_GMST[index],wavelength);
        delta_Y[dy*index:dy*index+dy] = np.subtract(Xradar[:,index],TotalObsvVec[dy*index:dy*index+dy]);
    
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.linalg.pinv(np.dot(TotalObsvMatT,np.dot(RynInv,TotalObsvMat)));
    delta_X_hat = np.dot(S_hat,np.dot(TotalObsvMatT,np.dot(RynInv,delta_Y)));
    Xdash = Xnom[0:dx,L-1] + delta_X_hat; 

    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = MathsFn.fn_invSymQuadForm(delta_Y,Ryn);
    return Xdash,S_hat,Ji,delta_X_hat
    
def fnGaussNewtonBatch_Case4_RK_OD_monostatic_rangedopp(Xdash,timevec,L,fnNomiTrajSTM,fnF,fnA,theta_GMST,Xradar,R,wavelength,radar_ecef):
    """
    Author: AshivD.
    Created: 22.09.17
    Edited: 
    """
    dy = np.shape(R)[0];
    dx = np.shape(Xdash)[0];
    # Generate nominal trajectory and STMs.
    Xnom,stm = fnNomiTrajSTM(Xdash,timevec,L,fnF,fnA);
    # Total observation matrix and its transpose
    TotalObsvMat = np.zeros([dy*L,dx],dtype=np.float64);
    TotalObsvMatT = np.zeros([dx,dy*L],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dy*L],dtype=np.float64);
    
    # simulated perturbation vector
    delta_Y = np.zeros([dy*L],dtype=np.float64);
    Ryn = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1);
    
    for index in range(0,L):
        # state transition matrix
        H = BD.fnJacobianH_Monostatic_RangeAndDoppler_OD(Xnom[0:dx,index],radar_ecef,theta_GMST[index],wavelength);
        TotalObsvMat[dy*index:dy*index+dy,:] = np.dot(H,stm[:,:,index]);
        TotalObsvMatT[:,dy*index:dy*index+dy] = np.transpose(np.dot(H,stm[:,:,index]));
        TotalObsvVec[dy*index:dy*index+dy] = BD.fnCalculate_Monostatic_RangeAndDoppler_OD(Xnom[0:dx,index],radar_ecef,theta_GMST[index],wavelength);
        #np.dot(H,Xnom[:,index]); # 22.09.17
        #BD.fnCalculate_Monostatic_RangeAndDoppler_OD(Xnom[0:dx,index],radar_ecef,theta_GMST[index],wavelength);
        delta_Y[dy*index:dy*index+dy] = Xradar[:,index] - TotalObsvVec[dy*index:dy*index+dy];
    
    RynInv = np.linalg.inv(Ryn);
    S_hat = np.linalg.pinv(np.dot(TotalObsvMatT,np.dot(RynInv,TotalObsvMat)));
    delta_X_hat = np.dot(S_hat,np.dot(TotalObsvMatT,np.dot(RynInv,delta_Y)));
    Xdash = Xnom[0:dx,L-1] + delta_X_hat; 

    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = MathsFn.fn_invSymQuadForm(delta_Y,Ryn);
    return Xdash,S_hat,Ji,delta_X_hat

# --------------------------------------------------------------------------- #
def fnGaussNewtonBatch_Case4_RK_OD_bistatic_rangedopp_simple(Xdash,timevec,L,fnNomiTrajSTM,fnF,fnA,theta_GMST,Xradar,R,wavelength,radar_ecef,tx_ecef):
    """
    Follows Batch Processing Algorithm Flowchart in Chapter 4 in Tapley, Schutz, Born
    See fig 4.6.1 on pg 197 in the 2004 edition
    Author: AshivD.
    Created: 28.09.17
    Edited: 
    """
    dy = np.shape(R)[0];
    dx = np.shape(Xdash)[0];
    # Generate nominal trajectory and STMs.
    Xnom,stm = fnNomiTrajSTM(Xdash,timevec,L,fnF,fnA);
    # Total observation matrix 
    TotalObsvMat = np.zeros([dx,dx],dtype=np.float64);
    # total observation vector
    TotalObsvVec = np.zeros([dx],dtype=np.float64);
    
    # simulated perturbation vector
    delta_Y = np.zeros([dy],dtype=np.float64);
    Rinv = np.linalg.inv(R)
    for index in range(0,L):
        # state transition matrix
        H = BD.fnJacobianH_Bistatic_RangeAndDoppler_OD(Xnom[0:dx,index],radar_ecef,tx_ecef,theta_GMST[index],wavelength);
        Hi = np.dot(H,stm[:,:,index]);
        HiT = np.transpose(np.dot(H,stm[:,:,index]));
        Ynom = BD.fnCalculate_Bistatic_RangeAndDoppler_OD_fast(Xnom[0:dx,index],radar_ecef,tx_ecef,theta_GMST[index],wavelength);
       
        delta_Y = np.subtract(Xradar[:,index],Ynom);
        
        TotalObsvMat = TotalObsvMat + np.dot(HiT,np.dot(Rinv,Hi));
        TotalObsvVec = TotalObsvVec + np.dot(HiT,np.dot(Rinv,delta_Y));
    
    S_hat = np.linalg.pinv(TotalObsvMat);
    delta_X_hat = np.dot(S_hat,TotalObsvVec);
    Xdash = Xnom[0:dx,L-1] + delta_X_hat; 

    # Cost Ji of cost function. (Crassidis pg 28)
    Ji = MathsFn.fn_invSymQuadForm(delta_Y,R);
    return Xdash,S_hat,Ji,delta_X_hat # check main_056_iss_28_3.py 28.09.17