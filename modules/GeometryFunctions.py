"""
GeometryFunctions.py

Several functions from geometry which are useful for modelling radar geometries
Some of these were originally in BistaticFunctions.py

Created by: Ashiv Dhondea
Created on: 26 April 2017
Edited:
28/04/17: Added fnCalculate_Projection_3D_2D
08/05/17: created the function fnCalculate_CircumferencePoints
09/05/17: created the function fnCheck_IsInCircle
09/05/17: added the function fnSmoothe_AngleSeries
12/05/17: fixed a sign bug in fnCalculate_ClosestPointOnLine
12/05/17: debugged the function fnSmoothe_AngleSeries
24/05/17: commented out print statements in the function fnSmoothe_AngleSeries
30/05/17: added the function fnCalculate_BeamFoV

29.08.17: moved the rotation functions from AstFn to this library

Ref: 
1. Coordinate conversion and tracking for very long range radars. Tian, Bar-Shalom

Books on vector analysis may also be useful.
"""
import math
import numpy as np
# ----------------------------------------------------------------------------------------------- #
def fnCalculate_PlaneNormal(p,q,r):
	"""
	Calculate the normal vector to a plane containing three points
	
	Created: 21 April 2017
	Added to BD on 22 April 2017. Validation in testbistaticplane.py
	Moved to GF on 26/04/17
	"""
	pq = q - p;
	pr = r - p;
	normal_vec = np.cross(pq,pr);
	return normal_vec

def fnCalculate_Coords_Plane3D(p,q,r):
	"""
	Calculate equation of plane containing the 3 points p,q,r
	
	matches results from http://keisan.casio.com/exec/system/1223596129
	
	Created: 26 April 2017
	"""
	normal_vec = fnCalculate_PlaneNormal(p,q,r);
	d = -q.dot(normal_vec);
	return normal_vec, d

def fnCalculate_LawOfSines_Angles(a,A,B,C):
	"""
	Implements the Law of Sines given 
	the 3 angles + 1 side
	
	A,B,C = angles in the triangle
	a,b,c = length of the side opposite the angle
	
	Be careful about which angle corresponds to 
	which side of the bistatic triangle.
	
	Created: 22 April 2017
	"""
	ratio = a/math.sin(A);
	b = math.sin(B)*ratio;
	c = math.sin(C)*ratio;
	return b,c
# -----------------------------------------------------------------------------------------------#
def fnCalculate_Coords_Line3D(direction_vec,pos_line,mu):
    """
    Calculates coordinates of a point on a line in 3D.
    direction_vec is the direction vector of the line
    pos_line is the position vector of a point on the line.
    mu is the free parameter which determines the coordinates of the desired
    point on the line.
    
    Created: 26/04/17 in testplanetransformations.py
    """
    pos_new = pos_line + mu*direction_vec;
    return pos_new # validated in testplanetransformation.py on 26/04/17
    
def fnCalculate_ClosestPointOnLine(a,b,p):
	"""
	Based on: https://gamedev.stackexchange.com/questions/72528/how-can-i-project-a-3d-point-onto-a-3d-line
	
	Created: 26 April 2017
	"""
	ap = (p-a);
	ab = (b-a);
	closest_point = a + np.dot(ap,ab) / np.dot(ab,ab) * ab; # bug: 12/5/17: flipped the sign
	return closest_point # validated in testplanetransformations.py
 
 
def fnCalculate_Projection_3D_2D(r_0,e_1,e_2,r_p):
    """
    Calculate the projection of a 3D point onto a 2D plane
    r_0 = normal vector to plane
    e_1 = 3D vector which represents the x-axis of the 2D plane
    e_2 = 3D vector which represents the y-axis of the 2D plane    
    
    http://stackoverflow.com/questions/23472048/projecting-3d-points-to-2d-plane
    
    Created: 28 April 2017
    """
    n = r_0/np.linalg.norm(r_0);
    pvec = r_p - r_0;
    # 2D coordinates of P are
    t_1 = np.dot(e_1,pvec);
    t_2 = np.dot(e_2,pvec);
    # out of plane separation
    s = np.dot(n,pvec);
    return t_1,t_2,s 
# ----------------------------------------------------------------------------------------------------------- #
def fnCalculate_CircumferencePoints(centre,radius,numpoints):
    """
    Calculate the coordinates of points on the circumference of a circle.
	
    	(x-centrex)^2 + (y-centrey)^2 = radius^2
	
    	Created: 08 May 2017
	"""
    circpts = np.zeros([2,numpoints]);
    theta = np.linspace(0,2*math.pi,numpoints,endpoint=True);
    xcomp = radius*np.cos(theta) + centre[0];
    ycomp = radius*np.sin(theta) + centre[1];
    circpts = np.vstack((xcomp,ycomp));
    return circpts # validated in main_007_fengyun_06.py
# ----------------------------------------------------------------------------------------------------------- #
def fnCheck_IsInCircle(centre,radius,testpoint):
    """
    Checks if a testpoint lies within a circle
    Created: 9 May 2017
    """
    if (np.linalg.norm(testpoint-centre) <= radius):
        return True
    else:
        return False # validated in main_007_fengyun_06.py

def fnSmoothe_AngleSeries(y,angle):
    """
    Smoothe a time indexed vector of angles.
    angle is the angle about which wrapping occurs.
    Created: 9 May 2017
    Edited:
    24/05/17: commented out print statements
    """
    leny = len(y)
    #print leny
    az_rx_argmin = np.argmin(y);
    #print az_rx_argmin
    az_rx_jump_before = y[az_rx_argmin] - y[az_rx_argmin-1];
    
    if az_rx_argmin < leny-1: # edited: 12 May 2017: to account for reaching the end of the time series at az_rx_argmin
		az_rx_jump_after = y[az_rx_argmin+1] - y[az_rx_argmin];
    else:
		return y # no need to smoothe anything
	
    if az_rx_jump_before < -angle/2.:
        # compensate after
        az_rx_comp = 1.;
        az_rx_corr = np.hstack((np.zeros([az_rx_argmin],dtype=np.float64),az_rx_comp*angle*np.ones([leny-az_rx_argmin],dtype=np.float64)));
        az_rx_nice = np.add(y,az_rx_corr);
    else:
        if az_rx_jump_after > angle/2.:
            az_rx_comp = -1.;
            az_rx_corr = np.hstack((np.zeros([az_rx_argmin+1],dtype=np.float64),az_rx_comp*angle*np.ones([leny-az_rx_argmin-1],dtype=np.float64)));
            az_rx_nice = np.add(y,az_rx_corr);
        else:
            az_rx_comp = 0;
            az_rx_nice = y;
    return az_rx_nice # validated in testsmooth.py

# ----------------------------------------------------------------------------------------------------- #
def fnCalculate_BeamFoV(a,b):
	"""
	Area covered by beam in degree^2 or radians^2
	
	Created: 30 May 2017
	"""
	area = math.pi*a*b;
	return area
	
# -- rotation functions moved on 29.08.17 ------------------ #
# -- Rotation functions  --------------------------------#
def fnRotate1(alpha_rad):
    T = np.array([[1,                 0   ,          0],
                  [0,  math.cos(alpha_rad)  ,math.sin(alpha_rad)],
                  [0, -math.sin(alpha_rad)  ,math.cos(alpha_rad)]],dtype=np.float64);
    return T # Validated against Vallado's example 2-3. 20/06/16

def fnRotate2(alpha_rad):
    T = np.array([[math.cos(alpha_rad), 0, -math.sin(alpha_rad)], 
                  [                0, 1,                  0],
                  [math.sin(alpha_rad), 0,  math.cos(alpha_rad)]],dtype=np.float64);
    return T

def fnRotate3(alpha_rad):
    T = np.array([[ math.cos(alpha_rad),math.sin(alpha_rad),0], 
                  [-math.sin(alpha_rad),math.cos(alpha_rad),0],
                  [                 0,                0,1]],dtype=np.float64);
    return T # Validated against Vallado's example 2-3. 20/06/16
    # Edited: 28 September 2016: use math instead of np for non-array operations.

# --------------------------------------------------------------------------- #
def fnAngleModular(a,b):
    """
    Modular arithmetic. 
    Handling angle differences when target flies from one quadrant to another 
    and there's a kink in the az angles
    Created: 29 April 2017
    """
    diffabs = abs(b) - abs(a);
    return diffabs # see testmodulararithmetic.py
	
	
# -------------------------------------------------------------------------------------------------------------------------------- #
def fnAngleDiff(ang1,ang2,wrappedangle):
    """
    in [deg] not [rad]
    
    if wrappedangle = 360 [deg], then keep the result in the range [0,360).
                      180 [deg],                                   [0,180)
    
    http://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
    
    13 Nov 2016. used when finding the error in estimated keplerians.
    Edited: 14 Nov 2016: the angle to wrap about.
    """
    diffangle = (wrappedangle/2.) - abs(abs(ang1 - ang2) - (wrappedangle/2.)); 
    return diffangle
	
def fnAngle(v1, v2, acute):
    """
    Calculates the angle between two vectors in 3D in Python.
    from http://stackoverflow.com/questions/39497496/angle-between-two-vectors-3d-python
    12 Nov 2016
    """
    # v1 is your first vector
    # v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        return angle
    else:
        return 2 * np.pi - angle
		
def fnZeroTo2Pi(rotangle):
    """
    Wraps angle to fit in [0,2pi).
    Works in [rad] not [deg]
    Date: 7 October 2016
    """
    wrappedangle = rotangle % (2*math.pi);
    return wrappedangle

def fnZeroToPi(rotangle):
    """
    Wraps angle to fit in [0,pi).
    Works in [rad] not [deg]
    Date: 15 October 2016
    """
    wrappedangle = rotangle % (math.pi);
    return wrappedangle
    
def fnwrapToPi(angle):
    """
    Wrap angle to fit in [-pi,pi)
    Works in [rad] not [deg]
    Created: 16 November 2016
    """
    wrappedangle = (angle + math.pi) % 2*math.pi - math.pi;
    return wrappedangle