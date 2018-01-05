readme_modules.txt

This directory contains modules developed for the SORADSIM project.

Author: Ashiv Rao Dhondea. ashivdhondea5[at]gmail.com
Created: 05 January 2018
# ------------------------------------------------------------------------------ #
Files in this directory:
01. AstroConstants.py 
    Contains constants which are useful in this project. 
    These relate to astrodynamics, radar and geodesy.
02. AstroFunctions.py
    Functions which are related to astrodynamics. All distances are expressed in 
    terms of kilometres, not metres. All angles, unless specificed, are in radians, 
    not degrees. AstroFunctions.py bases itself on theory presented in the following
    book: Fundamentals of Astrodynamics, Vallado (1997, 2013 editions)
    
03. BistaticAndDoppler.py
    Functions which are related to Doppler shift and bistatic radar measurements.
    Technical background: 
    (a) Tracking Filter Engineering, Norman Morrison. 2013.
    (b) Bistatic Radar: Principles & Practice, Cherniakov. 
    (c) Bistatic Radar, Willis.
    (d) Coordinate Conversion and Tracking for Very Long Range Radars, Tian, Bar-Shalom.
     05259184.pdf

04. Coastline.txt
    This txt file contains latitudes and longitudes of the coastline around landmasses
    around the Earth. The data is used in plotting satellite ground tracks.
    
05. DynamicsFunctions.py
    A collection of functions which implement dynamics-related functions for orbiting objects.
    Theoretical background:
    (a) Tracking Filter Engineering, Norman Morrison. 2013
    (b) Estimation with applications to tracking and navigation, Bar Shalom, Li, Kirubarajan. 2001
    
06. GeometryFunctions.py
    Several functions from geometry which are useful for modelling radar geometries.
    Theoretical background:
    (a) Fundamentals of Astrodynamics, Vallado (1997, 2013 editions)
    (b) Coordinate conversion and tracking for very long range radars. Tian, Bar-Shalom
    (c) Statistical Orbit Determination, Tapley, Schutz, Born. 2004
    
 07. KinematicsFunctions.py
     Kinematics functions for Kinematic Models such as CVM, CAM and NCVM, NCAM.
     Theoretical background:
     (a) Tracking Filter Engineering, Norman Morrison, 2013.
     (b) Estimation with applications to tracking and navigation. Bar Shalom, Li, Kirubarajan, 2001.
     
08. MathsFunctions.py
    A collection of functions which implement special maths functions
    Theoretical background:
    (a) Statistical Orbit Determination, 2004. Tapley, Born, Schutz.
    (b) ekfukf toolbox [Online]
    (c) Tracker Component Library [Github]
    
09. Num_integ.py
    A collection of Numerical Methods functions for solving/simulating deterministic differential equations.
    
10. PropagationModels.py
    Module which implements propagation modelling functions.
    Theoretical background:
    Satellite Orbits: Models, Methods, Applications. Montenbruck, Gill. 2000. Section 6.2.2
    
11. RadarSystem.py
     Functions which implement various radar engineering functions relevant to the space debris detection and tracking project.
     
12. SatelliteVisibility.py
     Checks if a satellite will be visible

13. StatsFunctions.py
    Statistics functions to assess the performance of an estimator

14. TimeHandlingFunctions.py 
    Functions to handle time keeping aspects.

15. Unbiased Converted Measurements.py
    Functions to perform unbiased converted measurements with spherical radar measurements.
    Reference:
    1. Bar Shalom, Longbin 1998: Unbiased converted measurements
    2. Javier Areta note on Spherical to Cartesian measurement converion with MATLAB code.
    3. S. Bordonaro thesis 2015 on Converted Measurement Kalman Filtering.

# ------------------------------------------------------------------------------ #
Other libraries required:
01. numpy
02. math
03. pandas
04. matplotlib
05. datetime
06. pytz
07. aniso8601
08. scipy.stats
# ------------------------------------------------------------------------------ #
%% License
% Copyright (c) 2016, 2017, 2018 Ashiv Dhondea

% Permission is hereby granted, free of charge, to any person obtaining a 
% copy of this software and associated documentation files (the "Software")
% , to deal in the Software without restriction, including without 
% limitation the rights to use, copy, modify, merge, publish, distribute, 
% sublicense, and/or sell copies of the Software, and to permit persons to 
% whom the Software is furnished to do so, subject to the following conditions:

% The above copyright notice and this permission notice shall be included 
% in all copies or substantial portions of the Software.

% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
% IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
% CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
% OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR 
% THE USE OR OTHER DEALINGS IN THE SOFTWARE.
