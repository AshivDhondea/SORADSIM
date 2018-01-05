"""
TimeHandlingFunctions.py

Functions to handle time keeping aspects.

Most of these functions were originally in AstroFunctions.py

Created by: Ashiv Dhondea
Created on: 29 August 2017

Edited:
29.08.17: edited the function fnCalculate_DatetimeEpoch

"""
import numpy as np

import datetime as dt
import pytz
import aniso8601

def fn_Calculate_Epoch_Time(epoch):
    """
    Calculates the epoch time of a TLE in terms of year, month, day, hours, minutes, seconds.
    Date: 28 September 2016
    
    originally in AstroFunctions.py
    """
    year = int(epoch[0:2])+2000;
    day = int(epoch[2:5]);
    
    fraction_of_day = 24.*float(epoch[5:]);
    hours = int(fraction_of_day);
    fraction_of_day = 60.*(fraction_of_day - hours);
    minutes = int(fraction_of_day);
    fraction_of_day = 60.*(fraction_of_day - minutes);
    seconds = int(fraction_of_day);
    millisecs = fraction_of_day - seconds;
    return year, day, hours, minutes, seconds, millisecs

def fn_epoch_date(year,today):
    """
    Computes the date of the today'th day of the year 'year'
    
    Requires the date class of datetime
    
    Author: Ashiv Dhondea, RRSG, UCT.
    Date: 28 September 2016
    
    originally in AstroFunctions.py
    """
    now = dt.date(year, 1, 1);
    epochday = today-1;
    difference1 = dt.timedelta(days=epochday);
    todays_date = now + difference1
    return todays_date
    

def fnSeconds_To_Hours(time_period):
    """
    Convert from seconds to hours, minutes and seconds.
    
    Date: 16 October 2016

    originally in AstroFunctions.py
    """
    num_hrs = int(time_period/(60.*60.));
    time_period =time_period - num_hrs*60.*60.;
    num_mins = int(time_period/60.);
    num_secs = time_period - num_mins*60.;
    return num_hrs,num_mins,num_secs # edit: 1/12/16: float division and multiplication

    
def fn_Time_Duration(hrs_utc,mins_utc,secs_utc,duration_h,duration_m,duration_s):
    """
    Find the time after a certain period.
    Date: 29 September 2016
    
    originally in AstroFunctions.py
    """
    s_UTC = secs_utc + duration_s;
    s_min = int(s_UTC/60.0);
    s_UTC = s_UTC - 60.0*s_min;
    
    m_UTC = mins_utc + duration_m + s_min;
    min_hrs = int(m_UTC/60.0);
    m_UTC = m_UTC - 60.0*min_hrs;
    
    h_UTC = hrs_utc + duration_h + min_hrs;
    
    return h_UTC,m_UTC,s_UTC
    
def fn_HMS_to_S(hrs,mins,secs):
    """
    Convert from hours and minutes and seconds to seconds.
    Date: 02 October 2016.
    
    originally in AstroFunctions.py
    """
    epoch_time = (hrs*60*60)+(mins*60.0)+secs;
    return epoch_time    

def fnJulianDate(yr, mo, d, h, m, s):
    """
    Implements Algo 14 in Vallado book: JulianDate
    Date: 05 October 2016
    
    originally in AstroFunctions.py
    """
    JD = 367.0*yr - int((7*(yr+ int((mo+9)/12)))/4.0) + int((275.0*mo)/9.0) + d+ 1721013.5 + ((((s/60.0)+m)/60+h)/24.0);
    return JD # validated with example 3-4 in vallado.

def fn_Calculate_GMST(JD):
    """
    Calculates the Greenwich Mean Sidereal Time according to eqn 3-47 on page 188 in Vallado.
    Date: 05 October 2016
    Edit: 06 October 2016: CAUTION: theta_GMST is output in [degrees] rather than in [radians], 
    unlike most of the angles in this file.
    
    originally in AstroFunctions.py
    """
    T_UT1 = (JD - 2451545.0)/36525.0
    theta_GMST = 67310.54841 + (876600.0*60*60 + 8640184.812866)*T_UT1 + 0.093104 * T_UT1**2 - 6.2e-6 * T_UT1**3;
    
    while theta_GMST > 86400.0:
        theta_GMST = theta_GMST - 86400;

    theta_GMST = theta_GMST/240.0;
    theta_GMST = theta_GMST - 360; # in [deg] not [rad] !!!!!!!!!!!!!!!!!!!!!!!!!!!
    return theta_GMST # validated with example 3-5 in vallado.

def fn_Convert_Datetime_to_GMST(datetime_object):
    """
    Converts a date and time in the datetime object form
    to GMST.
    
    Date: 05 October 2016
    Edited: 8 December 2016 : edited to take microseconds into consideration
    
    originally in AstroFunctions.py
    """
    obj = datetime_object;
    julianday =  fnJulianDate(obj.year,obj.month,obj.day,obj.hour,obj.minute,obj.second+ (1.e-6)*obj.microsecond); #edited: 8 December 2016 to account for microsecond
    theta_GMST =  fn_Calculate_GMST(julianday);
    return theta_GMST # validated with example 3-5 in vallado.
	
	
# ----------------------------------------------------------------------------- #
def fnCalculate_DatetimeEpoch(timevec, index, timestamp):
    """
    Find the datetime object for the time index.
    
    Created: 24 May 2017 in testepochfunc.py
    
    Edited:
    29.08.17: Commented out everything, better to use datetime methods. 
                As set up originally, results would be imprecise due to 
                ignoring time steps which are fractions of a second.
    """
    """
    current_time = timevec[index];
    hrs,mins,secs = fnSeconds_To_Hours(current_time + timestamp.hour*60*60 + timestamp.minute*60 + timestamp.second);
    dys = timestamp.day + int(math.ceil(hrs/24));
    if hrs >= 24:
        hrs = hrs - 24*int(math.ceil(hrs/24)) ;
    ms = secs %1; secs = secs-ms;
    epochindex = dt.datetime(year=timestamp.year,month=timestamp.month,day=int(dys),hour=int(hrs),minute=int(mins),second=int(secs),microsecond=int(1.e6*ms),tzinfo= pytz.utc);
    """
    epochindex = timestamp + dt.timedelta(seconds=timevec[index]); # tested in test_datetime2.py
    
    return epochindex
	
def fnRead_Experiment_Timestamps(experiment_timestamps,index):
    """
    Read experiment timestamps (which are usually read from a txt file)
    
    Created: 28 August 2017
    originally in AstroFunctions.py
    """
    line = experiment_timestamps[index];
    modified_timestring = line[:-1];
    experiment_timestamps_start = aniso8601.parse_datetime(modified_timestring);
    return experiment_timestamps_start