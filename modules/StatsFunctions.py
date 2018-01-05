"""
StatsFunctions.py

Statistics functions to assess the performance of an estimator

Author: Ashiv Dhondea, RRSG, UCT.

Date: 23 November 2016

Edited: 
23 November 2016: added function fnCalcNees and fnChiSquaredTest to calculate the NEES and perform the Chi-Squared test.
13 December 2016: changed the name of fnCalcNees to fnCalc_NEES. Made sure that the NEES is normalized.
09/03/2017: added the function fnCalc_NES
19/03/2017: cleaned up several functions.
20/03/2017: rewrote the 3sigma test 
22/09/2017: added a comment to fn3SigmaTest
28/09/2017: added the function fnCalculate_MSEs

Theoretical background:
1. Tracking Filter Engineering, Norman Morrison. 2013.
2. Estimation with applications to tracking and navigation. Bar Shalom.
"""
import numpy as np
import MathsFunctions as MathsFn
# -------------------------------------------------------------------------------------------------- #
## NEES according to Bar Shalom book. 23.11.16
def fnCalc_NEES(errorvec,covmat):
    simul = np.shape(errorvec)[1];
    nVars = float(np.shape(errorvec)[0]);
    nees = np.zeros([simul],dtype=np.float64);
    for index in range(0,simul):
        nees[index] = MathsFn.fn_invSymQuadForm(errorvec[:,index],covmat[:,:,index]); # edited: 09/03/17: used a faster implementation of invsymquadform
    return nees/nVars # edited: 13/12/16: reminder that the NEES is NORMALIZED by the dimensionality of the state vector

## One-shot Chi-squared test according to Tracking Filter Engineering book. 23.11.16 
def fnChiSquaredTest(errorvec,covmat):
    nees = fnCalc_NEES(errorvec,covmat);
    #dx = 6; # we already know dx =6 in this project.
    simul = np.shape(errorvec)[1];
    threshold_6 = 16.81;
    chi_metric = np.full([simul],False,dtype=np.bool); # edited: 20/03/17
    for index in range(0,simul):
        if nees[index] < threshold_6:
            chi_metric[index] = True;
    return chi_metric
    
## One-shot 3-sigma test according to Tracking Filter Engineering book. 23.11.16
def fn3SigmaTest(errorvec,covmat):
    """
    3-sigma test
    Created: 19 March 2017    
    Edited: 
    20/03/17: rewrote according to the theory in TFE
	22/09/17: added a comment
    """
    dx = np.shape(errorvec)[0];
    simul = np.shape(errorvec)[1];
    three_sigma_metric = np.full([dx,simul],False,dtype=np.bool);
    three_sigma_bounds = np.zeros([dx,simul],dtype=np.float64);
    for index in range(0,simul):
        three_sigma_bounds[:,index] = 3.*np.sqrt(np.diagonal(covmat[0:dx,0:dx,index])); # edited: 20/03/17
        three_sigma_metric[:,index] = np.less_equal(np.absolute(errorvec[:,index]),three_sigma_bounds[:,index]); # edited: 13/11/17
        """
        Comment: 22.09.17
        np.less_equal([4, 2, 1], [2, 2, 2])
        array([False,  True,  True], dtype=bool)

        meaning return False if errorvec[:,index] not less or equal to three_sigma_bounds[:,index]
        """

    return three_sigma_metric, three_sigma_bounds
    
# --------------------------------------------------------------------------- #
## NES according to Bar Shalom book.
def fnCalc_NES(errorvec,covmat):
    """
    Calculate the Normalized Error Squared
    
    Created: 09 March 2017
    """
    nVars = np.shape(errorvec)[0];
    nes = MathsFn.fn_invSymQuadForm(errorvec,covmat); 
    return nes/float(nVars) 
# --------------------------------------------------------------------------- #
def fnCalculate_MSE(x_error):
    """
    Calculate the means squared error in the estimated error vector
    
    x_error is dx by num_mc by len(timevec)
    Created on: 28/09/17
    """
    dx = np.shape(x_error)[0]; num_MC = np.shape(x_error)[1];l_timevec=np.shape(x_error)[2];
    sqr_error = np.zeros([dx,num_MC,l_timevec],dtype=np.float64);
    means_sqr_error =  np.zeros([dx,l_timevec],dtype=np.float64);
    
    for i_t in range(0,l_timevec):
        for i_mc in range(0,num_MC):
            # Square the error
            sqr_error[:,i_mc,i_t] = np.square(x_error[:,i_mc,i_t]);
        # add all squared errors to reduce by the number of Monte Carlo runs
        means_sqr_error[:,i_t]= np.sum(sqr_error[:,:,i_t],axis=1)/num_MC;
        #divide by number of Monte Carlo runs to averag
        
    return means_sqr_error