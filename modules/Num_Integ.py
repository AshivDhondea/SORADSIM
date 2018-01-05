"""
## Num_Integ.py
# ------------------------- #
# Description:
# A collection of Numerical Methods functions for solving/simulating deterministic and stochastic differential equations.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 20 June 2016
# Edits: 16 July 2016: added fnRK4_moment used to numerically integrate the moment differential equations for the CD-EKF. removed it.
#        27 July 2016: included the stochastic numerical integration schemes in this file. added references and comments.
#        28 July 2016: changed the arguments for fnRK4_vector. Q and L are included to integrate fnMoment_DE from DynFn.
         28 September 2016: cleaned up the code.
"""
# ------------------------- #
import numpy as np
# ------------------------------------------------------------------------------------------------------------------------------------------#
## References
""" 1. Kloeden, Platen: Numerical Solution of Stochastic Differential Equations.
    @book{kloeden2011numerical,
      title={Numerical Solution of Stochastic Differential Equations},
      author={Kloeden, P.E. and Platen, E.},
      series={Stochastic Modelling and Applied Probability},
      year={1992},
      publisher={Springer Berlin Heidelberg}
    }
    2.Crouse(2015) Basic tracking using nonlinear continuous-time dynamic models [Tutorial]
    @article{crouse2015basic,
      title={Basic tracking using nonlinear continuous-time dynamic models [Tutorial]},
      author={Crouse, David},
      journal={Aerospace and Electronic Systems Magazine, IEEE},
      volume={30},
      number={2},
      pages={4--41},
      year={2015},
      publisher={IEEE}
    }
    3. Burden, Faires (2011) Numerical Analysis
    @book{Burden,
     author = {Burden, Richard L. and Faires, J. Douglas},
     title = {Numerical Analysis},
     year = {2011},
     edition = {9th},
     publisher={Brooks/Cole}
    } 
"""
# ------------------------------------------------------------------------------------------------------------------------------------------#
## Deterministic numerical integration
def fnRK4_vector(f, dt, x,t,Q=None,L=None):
    """
    fnRK4_vector implements the Runge-Kutta fourth order method for solving Initial Value Problems.
    f : dynamics function
    dt : fixed stepsize
    x : state vector
    t : current time instant.
    Refer to Burden, Faires (2011) for the RK4 method.
    Edit: 28/7/16: Added Q and L to integrate fnMoment_DE.
    """
    
    if Q is None: # Assuming L is also None.
        # Execute one RK4 integration step
        k1 = dt*  f(t         ,x         );
        k2 = dt*  f(t + 0.5*dt,x + 0.5*k1);
        k3 = dt*  f(t + 0.5*dt,x + 0.5*k2);
        k4 = dt*  f(t +     dt,x +     k3);
    else:
        # Execute one RK4 integration step
        k1 = dt*  f(t         ,x         ,Q,L);
        k2 = dt*  f(t + 0.5*dt,x + 0.5*k1,Q,L);
        k3 = dt*  f(t + 0.5*dt,x + 0.5*k2,Q,L);
        k4 = dt*  f(t +     dt,x +     k3,Q,L);
        
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6.0
# -------------------------------------------------------------------------------------------------------------------------------------------#
## Stochastic numerical integration
def fnEuler_Maruyama(x,fnD,timevec,L,Qd):
    """
    fnEuler_Maruyama implements the 0.5 strong Euler-Maruyama scheme. See Sarkka, Solin (2014).
    x : state vector of dimensions dx by 1.
    fnD : nonlinear dynamics function.
    timevec : time vector for simulation duration.
    L : dispersion matrix of dimensions dx by dw.
    Qd : discretized covariance matrix of dimensions dw by dw.
    """
    dt = timevec[1]-timevec[0];
    dx = np.shape(L)[0]; # dimension of state vector.
    dw = np.shape(L)[1]; # dimension of process noise vector
    x_state = np.zeros([dx,len(timevec)],dtype=np.float64);
    x_state[:,0] = x;
        
    for index in range (1,len(timevec)):
        x_state[:,index] = x_state[:,index-1] + dt*fnD(timevec[index-1],x_state[:,index-1]) + np.dot(L,np.random.multivariate_normal(np.zeros(np.shape(L)[1],dtype=np.float64),Qd));
    return x_state

def fnEuler_Maruyama_test(x,fnD,timevec,L,Qc):
    dt = timevec[1]-timevec[0];
    dx = np.shape(L)[0]; # dimension of state vector.
    dw = np.shape(L)[1]; # dimension of process noise vector
    x_state = np.zeros([dx,len(timevec)],dtype=np.float64);
    x_state[:,0] = x;
        
    for index in range (1,len(timevec)):
        x_state[:,index] = x_state[:,index-1] + dt*fnD(timevec[index-1],x_state[:,index-1]) + np.dot(np.dot(L,Qc),np.random.multivariate_normal(np.zeros(np.shape(L)[1],dtype=np.float64),dt*np.eye(np.shape(L)[1])));
    return x_state

def fnSRK_Crouse(x,fnD,timevec,L,Qd):
    """
    fnSRK_Crouse implements the 1.5 strong Stochastic Runge-Kutta method in Crouse (2015).
    Note that as opposed to Crouse (2015), we have assumed that the dispersion matrix is constant, i.e. not time-varying
    and definitely not state-dependent.
    x : state vector of dimensions dx by 1.
    fnD : nonlinear dynamics function.
    timevec : time vector for simulation duration.
    L : dispersion matrix of dimensions dx by dw.
    Qd : discretized covariance matrix of dimensions dw by dw.
    Note: 27/07/16: explicit order 1.5 strong scheme in section 11.2 in Kloden, Platen (1992) and Crouse(2015).
    """
    dw = np.shape(L)[1]; # dimension of process noise vector
    dx = np.shape(L)[0]; # dimension of state vector.
    
    x_state = np.zeros([dx,len(timevec)],dtype=np.float64);
    x_state[:,0] = x;
    
    dt = timevec[1]-timevec[0];
    
    # Form the covariance matrix for delta_beta and delta_alpha.
    beta_beta = dt*Qd;
    beta_alpha = 0.5*(dt**2)*Qd;
    alpha_alpha = ((dt**3)/3.0)*Qd;
    Qd_aug = np.zeros([dw+dw,dw+dw],dtype=np.float64); # The covariance matrix in eqn 44.
    Qd_aug[0:dw,0:dw] = beta_beta;
    Qd_aug[0:dw,dw:] = beta_alpha;
    Qd_aug[dw:,0:dw] = beta_alpha;
    Qd_aug[dw:,dw:] = alpha_alpha;

    # Generate process noise terms according to eqn 44.
    noisevec = np.zeros([dw+dw],dtype=np.float64); # mean vector is zero.

    y_plus = np.zeros([dx,dw],dtype=np.float64);
    y_minus = np.zeros([dx,dw],dtype=np.float64);
    fy_plus = np.zeros([dx,dw],dtype=np.float64);
    fy_minus = np.zeros([dx,dw],dtype=np.float64);
    f2 = np.zeros([dx,dw],dtype=np.float64); # equal to F2 (eqn 39)
    # F3 == F2 because we assume L is a constant matrix.
   

    for index in range(1,len(timevec)):
        process_noise = np.random.multivariate_normal(noisevec,Qd_aug);
        delta_beta = process_noise[0:dw];
        delta_alpha = process_noise[dw:];
        summ = np.zeros([dx],dtype=np.float64);
        
        for j in range(0,dw):
            # find yj+ and yj-. eqns 42 and 43
            y_plus[:,j]  = x_state[:,index-1] + (dt/float(dw))*fnD(timevec[index-1],x_state[:,index-1]) + np.sqrt(dt)*L[:,j];
            y_minus[:,j] = x_state[:,index-1] + (dt/float(dw))*fnD(timevec[index-1],x_state[:,index-1]) - np.sqrt(dt)*L[:,j];
            # expressions in eqns 40 and 38
            fy_plus[:,j]  = fnD(timevec[index-1],y_plus[:,j]);
            fy_minus[:,j] = fnD(timevec[index-1],y_minus[:,j]);
            f2[:,j] = (1/(2*np.sqrt(dt)))*(fy_plus[:,j] - fy_minus[:,j]); # eqn 40
            # sum term in eqn 38
            summ += fy_plus[:,j] - 2*fnD(timevec[index-1],x_state[:,index-1]) + fy_minus[:,j];

        f1 = x_state[:,index-1] + dt*fnD(timevec[index-1],x_state[:,index-1]) + 0.25*dt*summ; # eqn 38
        x_state[:,index] = f1 + np.dot(L,delta_beta) + np.dot(f2,delta_alpha); # eqn 37    
    
    return x_state
