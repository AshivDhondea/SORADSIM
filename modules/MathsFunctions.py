# -*- coding: cp1252 -*-
"""
Description:
A collection of functions which implement special maths functions in my Masters dissertation.

Created by: Ashiv Dhondea, RRSG, UCT.
Created on: 21 July 2016
Edited on: 
22 July 2016: added fn_Create_Concatenated_Block_Diag_Matrix, originally from DynamicsFunctions.py
28 July 2016: implemented the function invSymQuadForm from the Tracker Component Library.
6 August 2016: created the function fnInvert_symplectic_STM for inverting STMs for symplectic Hamiltonian systems.
28 September 2016: added comments and cleaned the code.
10 October 2016: from 15 September 2016: added the function fnStack_Block_Diag which is needed for measurement covariance matrices in the GNF framework.
23/01/17: added a fast implementation fn_invSymQuadForm
26/01/17: cleaned up the code.
02/03/17: added the function find_nearest required by PM.fnCalculate_DownlinkTime_Iter

References
1. Statistical Orbit Determination, 2004. Tapley, Born, Schutz.
2. ekfukf toolbox [Online]
3. Tracker Component Library [Github]
"""
# --------------------------------------------------------------------------- #
# Import libraries
import numpy as np
import math
import scipy.linalg
# --------------------------------------------------------------------------- #
def schol(A): # Cholesky decomposition for PSD matrices.
    ## Emulates schol.m of ekfukf toolbox.
    ## Description from schol.m
    ##    %SCHOL  Cholesky factorization for positive semidefinite matrices
    ##% Syntax:
    ##%   [L,def] = schol(A)
    ## % In:
    ##%   A - Symmetric pos.semi.def matrix to be factorized
    ##%
    ##% Out:
    ##%   L   - Lower triangular matrix such that A=L*L' if def>=0.
    ##%   def - Value 1,0,-1 denoting that A was positive definite,
    ##%         positive semidefinite or negative definite, respectively.
    ## % Copyright (C) 2006 Simo Särkkä
    n = np.shape(A)[0];
    L = np.zeros((n,n),dtype=np.float64);
    definite = 1;
    
    for i in range(0,n):
        for j in range(0,i+1):
            s = A[i,j];
            for k in range (0,j):
                s = s - L[i,k]*L[j,k];
            if j < i :
                if L[j,j] > np.finfo(np.float64).eps:
                    L[i,j] = s/L[j,j];
                else:
                    L[i,j] = 0;
            else:
                if (s < - np.finfo(np.float64).eps ):
                    s = 0;
                    definite = -1;
                elif (s < np.finfo(np.float64).eps):
                    s = 0;
                    definite = min(0,definite);
                    
                L[j,j] = np.sqrt(s);
    # if definite < 0, then negative definite
    # if definite == 0, then positive semidefinite
    # if definite == 1, then positive definite
    return L, definite
# ------------------------------------------------------------------- #
def fn_Create_Concatenated_Block_Diag_Matrix(R,stacklen):
    # fn_Create_Concatenated_Block_Diag_Matrix creates a block diagonal matrix of size (stacklen) x (stacklen)
    # whose diagonal blocks are copies of the matrix R.
##    L = [R]; 
##    for index in range (0,stacklen):
##        L.append(R);
##        ryn = scipy.linalg.block_diag(*L);
    ryn =np.kron(np.eye(stacklen+1),R); # Edit: 19/07/2016: probably better idea than using a for loop.
    return ryn
# ------------------------------------------------------------------------- #
def invSymQuadForm(x,M):
    """
    %%INVSYMQUADFORM Compute the quadratic form x'*inv(R)*x in a manner that
    %                should be more numerically stable than directly evaluating
    %                the matrix inverse, where R is a symmetric, positive
    %                definite matrix. The parameter M can either be the matrix
    %                R directly (R=M) (the default), or the lower-triangular
    %                square root of R (R=M*M'). The distance  can be computed
    %                for a single matrix and multiple vectors x. If x consists
    %                of vector differences and R is a covariance matrix, then
    %                this can be viewed as a numerically robust method of
    %                computing Mahalanobis distances.
    %
    %INPUTS: x       An mXN matrix of N vectors whose quadratic forms with the
    %                inverse of R are desired.
    %        M       The mXm real, symmetric, positive definite matrix R, or
    %                the square root of R, as specified by the following
    %                parameter.
    %        matType This optional parameter specified the type of matrix that
    %                M is. Possible values are
    %                0 (the default if omitted) M is the matrix R in the form
    %                  x'*inv(R)*x.
    %                1 M is the lower-triangular square root of the matrix R.
    %
    %OUTPUTS: dist   An NX1 vector of the values of x*inv(R)*x for every vector
    %                in the matrix x.
    %
    %As one can find in many textbooks, solvign A*x=b using matrix inversion is
    %generally a bad idea. This relates to the problem at hand, because one can
    %write
    %x'*inv(R)*x=x'inv(C*C')*x
    %           =x'*inv(C)'*inv(C)*x
    %where C is the lower-triangular Cholesky decomposition of R. Next, say
    %that
    %y=inv(C)*x.
    %Then, we are just computing y'*y.
    %What is a stable way to find y? Well, we can rewrite the equation as
    %C*y=x
    %Since C is lower-triangular, we can find x using forward substitution.
    %This should be the same as one of the many ways that Matlab solves the
    %equation C*y=x when one uses the \ operator. One can explicitely tell
    %Matlab that the matrix is lower triangular when using the linsolve
    %function, thus avoiding the need for loops or for Matlab to check the
    %structure of the matrix on its own.
    %
    %August 2014 David F. Crouse, Naval Research Laboratory, Washington D.C.
    %(UNCLASSIFIED) DISTRIBUTION STATEMENT A. Approved for public release.
    """
    ## 28 July 2016. Emulates the invSymQuadForm.m function from the Tracker Component Library. Removed matType to make things simpler.
    
    # implements x'M x
    C,definiteness = schol(M);
   
    y = np.linalg.solve(C,x);
    dist = np.dot(y.T,y);
    return dist
# --------------------------------------------------------------------------------- #
def fnInvert_symplectic_STM(stm):
    """
    Analytic way of inverting an STM when it is symplectic. This occurs when the acceleration
    can be written as the gradient of a potential function. Refer to eqn 4.2.22 on page 167 in
    Statistical Orbit Determination, Tapley, Schutz, Born. 2004.
    Created: 6/8/16
    """
    stm_inv = np.zeros_like(stm);
    dim = int(0.5*np.shape(stm)[0]); # stm should be (even number x same even number).
    stm_inv[0:dim,0:dim] = stm[dim:,dim:];
    stm_inv[dim:,0:dim] = -stm[dim:,0:dim];
    stm_inv[0:dim,dim:] = -stm[0:dim,dim:];
    stm_inv[dim:,dim:] = stm[0:dim,0:dim];    
    return stm_inv
# --------------------------------------------------------------------------------------- #
def fnStack_Block_Diag(R,stacklen):
    """
    Creates a block diagonal matrix using the first 'stacklen' square matrices  r x r in R.
    'R' is r x r x t. 
    
    This function is needed for the Gauss-Newton filter.
    
    Date: 15 September 2016
    Note that the difference between this function and fn_Create_Concatenated_Block_Diag_Matrix 
    is that R is r x r x t instead of just r x r.
    """
    stackmatrix = [R[:,:,0]]; 
    for index in range (1,stacklen):
        stackmatrix.append(R[:,:,index]);
        ryn = scipy.linalg.block_diag(*stackmatrix);
    return ryn

# --------------------------------------------------------------------------------------------------------- #
def fn_invSymQuadForm(x,M):
    """
    Faster implementation of invSymQuadForm
    Created: 23/01/17
    """
    R = np.linalg.inv(M);
    isqf = np.dot(np.transpose(x),np.dot(R,x));
    return isqf
# ------------------------------------------------------------------------------------------------------------------- #
def find_nearest(array,value):
    """
    Find element from array which is nearest to a certain value, as well as its index in an ordered array.
    
    Date: 02/03/17
    
    http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1],idx-1
    else:
        return array[idx],idx
