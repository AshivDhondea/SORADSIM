# -*- coding: cp1252 -*-
"""
## AstroConstants.py
# ------------------------- #
# Description:
# Declare Astrodynamics constants
# According to WGS84.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 20 June 2016
# Edits: 
22.07.2016: edited some constants
02.03.2017: added the speed of light
26.05.2017: added the Boltzmann constant. 
"""
# ------------------------- #
import math
import numpy as np

# ------------------------- #
mu_E = 398600.4418;# Earth’s gravitational parameter [km^3/s^2]

R_E = 6378.1366;# Earth equatorial radius [km] 

theta_dot = 7.29211585530066e-5;# [rad/s]

J_2 = 1.082626925638815e-3;# [] 

R_polar = 6356752.3142e-3;# Earth polar radius [km]
flattening = 0.0033528106718309896;
flattening_inverse = 298.2572229328697;

# ---------------------------------------------------------------------------- #
c = 299792458.*1.e-3; # [km/s]
# --------------------------------------------------------------------------- #
boltzmann_constant = 1.38064852e-23; # [watt sec/K]
