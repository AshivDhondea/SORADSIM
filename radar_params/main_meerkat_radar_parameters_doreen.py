# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 12:07:35 2017

Calculating the radar parameters given by Doreen.
I misunderstood her intended radar design.

Calculate and save meerkat radar parameters
http://public.ska.ac.za/meerkat/meerkat-schedule

Background:
1. Richard Curry: Radar System Performance Modeling.

@author: Ashiv Dhondea
"""
import RadarSystem as RS
import AstroConstants as AstCnst
# --------------------------------------------------------------------------- #
speed_light = AstCnst.c*1e3; # [m/s]
# --------------------------------------------------------------------------- #
# Provided by SKA-SA
meerkat_dish_radius = 0.5*13.5; # [m]

meerkat_sys_temp = 23.0; # [K]
# --------------------------------------------------------------------------- #
# Values provided by Doreen Agaba.

# Tx power
P_Tx = 2.e6; # [W]

# Centre frequency value
centre_frequency = 1.35e9; # [Hz]

wavelength = speed_light/(centre_frequency);
print 'wavelength = %f m' %wavelength

pulse_width = 5.e-6; # [s] aka waveform duration tau in Section 4.4 Curry
prf = 75.e3; # [Hz]
pulse_repetition_time = 1/prf; # [s]

bandwidth = 10.e6; # [Hz]
# --------------------------------------------------------------------------- #
"""
I have only just realized that Doreen must have chosen a phase-coded waveform
as waveform design for her pulse-Doppler radar.

So a number of calculations should be done differently.
"""
time_bandwidth_product = pulse_width*bandwidth; # pulse compression ratio; PCR
num_subpulses = int(time_bandwidth_product) # n_s in Curry 4.4
subpulse_width = pulse_width/num_subpulses; # [s]

print 'time-bandwidth product aka pulse compression ratio = %f ' %time_bandwidth_product
print 'number of sub-pulses employed = %d ' %num_subpulses
print 'waveform duration = %f s' %pulse_width
print 'subpulse duration = % s' %subpulse_width

# Doppler velocity resolution for a monostatic radar employing a phase-coded
# waveform. See section 4.4 in Curry: Radar Perf Modelling.
monostatic_doppler_vel_resolution = wavelength/(2*pulse_width); # [m/s]

# range resolution for a monostatic radar employing a phase-coded waveform
# See section 4.4 in Curry: Radar Perf Modelling.
monostatic_range_resolution = AstCnst.c*subpulse_width/2. # [km]

# --------------------------------------------------------------------------- #
# My values
antenna_efficiency = 0.6; # Value assumed. May be different.

k = 57.3; # ideal case; Doreen assumed 59.05

beamwidth_rx = RS.fnCalculate_AntennaBeamwidth(k,wavelength,meerkat_dish_radius);
print 'HPBW rx = %f deg' %beamwidth_rx

gain_rx = RS.fnCalculate_AntennaGain(meerkat_dish_radius,wavelength,antenna_efficiency);
gain_rx_dB = RS.fn_Power_to_dB(gain_rx);
print 'gain rx = %f dBi' %gain_rx_dB

denel_dish_radius = 5.; # figured this from Doreen's stated Tx beamwidth and gain.
gain_tx = RS.fnCalculate_AntennaGain(denel_dish_radius,wavelength,antenna_efficiency);
gain_tx_dB = RS.fn_Power_to_dB(gain_tx);
print 'gain tx = %f dBi' %gain_tx_dB

beamwidth_tx = RS.fnCalculate_AntennaBeamwidth(k,wavelength,denel_dish_radius);
print 'HPBW tx = %f deg' %beamwidth_tx

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
print 'Writing to file'
fname = 'main_meerkat_radar_parameters_doreen.txt'
f = open(fname,'w');
f.write('centre_frequency in hertz ='+str(centre_frequency)+'\n');
f.write('bandwidth in hertz ='+str(bandwidth)+'\n');
f.write('system temperature in kelvin ='+str(meerkat_sys_temp)+'\n');
f.write('HPBW Rx in deg ='+str(beamwidth_rx)+'\n');
f.write('Gain Rx in dBi ='+str(gain_rx_dB)+'\n');
f.write('HPBW Tx in deg ='+str(beamwidth_tx)+'\n');
f.write('Gain Tx in dBi ='+str(gain_tx_dB)+'\n');
f.write('PRF in Hz='+str(prf)+'\n');
f.write('Pulse width in seconds ='+str(pulse_width)+'\n');
f.write('transmitted power in watts ='+str(P_Tx)+'\n');

f.write('Number of subpulses ='+str(num_subpulses)+'\n');
f.write('Subpulse width ='+str(subpulse_width)+'\n');
f.write('Monostatic range resolution ='+str(monostatic_range_resolution)+'\n');
f.write('Monostatic Doppler velocity resolution ='+str(monostatic_doppler_vel_resolution)+'\n');

f.close();
print 'cool cool cool'

