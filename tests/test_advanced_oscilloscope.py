import json
import numpy as np
from DEIStools.acquire.deischannel import DEISchannel
from DEIStools.acquire.advancedoscilloscope import ZPico5000a
from DEIStools.acquire.multisinegen import MultisineGenerator
from pyeclab.device import BiologicDevice
from pyeclab.channel import Channel, ChannelOptions
import pyeclab.techniques as tech
from TrueFormAWG.trueformawg.trueformawg import TrueFormAWG, VISAdevices, import_awg_txt

# =============================================================================
# %% Set up saving path
# =============================================================================
saving_dir = 'E:/Experimental_data/Federico/2025/python_software_test/'
experiment_name = '2501130949_test_filtering_automation'


# =============================================================================
# %% Set up waveform generator with extended bandwidth multisine
# =============================================================================
awgs = VISAdevices()

trueform_address = awgs.list[1]
# Configure channel 1 with multisine with high frequncy
awg_ch1 = TrueFormAWG(trueform_address,1)
awg_ch1.clear_ch_mem()
multisine_high_path = 'E:/multisine_collection/2412111607multisine_splitted_100kHz-10mHz_8ptd_fgen1MHz_flat_norm_random_phases/low_freqs/waveform.txt'
multisine_high = import_awg_txt(multisine_high_path)
awg_ch1.load_awf('ms_high', multisine_high)
awg_ch1.select_awf('ms_high')
awg_ch1.set_Z_out_infinite()
awg_ch1.set_sample_rate(1000000)

# Configure channel 2 with multisine with high frequncy
awg_ch2 = TrueFormAWG(trueform_address,2)
awg_ch2.clear_ch_mem()
multisine_high_path = 'E:/multisine_collection/2412111607multisine_splitted_100kHz-10mHz_8ptd_fgen1MHz_flat_norm_random_phases/high_freqs/waveform.txt'
multisine_high = import_awg_txt(multisine_high_path)
awg_ch2.load_awf('ms_low', multisine_high)
awg_ch2.select_awf('ms_low')
awg_ch2.set_Z_out_infinite()
awg_ch2.set_sample_rate(1000)

# Combine channels
awg_ch1.combine_channels()
awg_ch1.set_amplitude(0.1)
awg_ch1.set_offset(0)

# Define frequencies for online computation of impedance
json_file = open('E:/multisine_collection/2412111607multisine_splitted_100kHz-10mHz_8ptd_fgen1MHz_flat_norm_random_phases/high_freqs/waveform_metadata.json')
waveform_metadata = json.load(json_file)
multisine_freqs = np.array(waveform_metadata['Frequencies / Hz'])
# Add a second multisine 
json_file = open('E:/multisine_collection/2412111607multisine_splitted_100kHz-10mHz_8ptd_fgen1MHz_flat_norm_random_phases/low_freqs/waveform_metadata.json')
waveform_metadata = json.load(json_file)
multisine_freqs = np.append(multisine_freqs,np.array(waveform_metadata['Frequencies / Hz']))
# Choose high frequencies
high_freqs = multisine_freqs[28:]

# =============================================================================
# %% Set up oscilloscope
# =============================================================================
# Measurment paramters
capture_size =  400000000
samples_total =  400000000
sampling_time =  1
sampling_time_scale = 'PS5000A_US'
# STFFT-EIS parameters
sample_size = int(100/1e-6) # low freq period / sampling rate
irange = 0.0001
filter_order = 5
cutoff = 100/500000
ds_factor = int(1e-4/1e-6) # new time step / original time step
buffer_size = int(1e4 * 60 * 60 * 1) # one hour of aqusition / seconds per spectra
# Connect instrument and perform the acquisiton
pico = ZPico5000a(sample_size,
                  high_freqs,
                  irange,
                  filter_order,
                  cutoff,
                  ds_factor,
                  buffer_size,
                  'PS5000A_DR_14BIT')
saving_path = saving_dir + experiment_name
pico.set_pico(capture_size, samples_total, sampling_time, sampling_time_scale, saving_path)
pico.set_channel('PS5000A_CHANNEL_A', 'PS5000A_500MV')
pico.set_channel('PS5000A_CHANNEL_B', 'PS5000A_1V', irange)
# Activate low pass filter


# =============================================================================
# %% Set up potetiostat
# =============================================================================

# IP address of the instrument
ip_address = '172.28.26.10'
# Path of the SDK from BioLogic installed in the machine
binary_path = "C:/EC-Lab Development Package/EC-Lab Development Package/"
# Istantiate device class
device = BiologicDevice(ip_address, binary_path = binary_path)

# Create CP technique
# E_range        = 0
# bw             = 8
# repeat_count   = 0
# record_dt      = 1
# record_dE      = 10    # Volts
# current        = 0    # Ampers
# duration_CP    = 100        # Seconds (sec * min * hours)
# limit_E_crg    = 0b11111
# E_lim_high     = 5        # Volts
# E_lim_low      = -5         # Volts
# i_range        = 8
# exit_cond      = 1
# xctr           = 0b00001000
# CP_user_params = tech.CPLIM_params(current, 
#                                       duration_CP, 
#                                       False, 
#                                       0, 
#                                       record_dt, 
#                                       record_dE, 
#                                       repeat_count, 
#                                       i_range,
#                                       E_range,
#                                       exit_cond,
#                                       xctr,
#                                       E_lim_high,
#                                       limit_E_crg, 
#                                       bw)
# CP_tech = tech.CPLIM_tech(device, device.is_VMP3, CP_user_params)

# Create CA technique
E_range        = 0
bw             = 8
repeat_count   = 0
record_dt      = 1
record_dI      = 10   # Ampers
voltage        = 0   # Volts
i_range        = 6
duration_CA    = 100     # Seconds (sec * min * hours)
exit_cond      = 10
xctr           = 0b00001000
CA_user_params    = tech.CA_params(voltage, 
                                  duration_CA, 
                                  False, 
                                  0, 
                                  record_dt, 
                                  record_dI, 
                                  repeat_count,
                                  i_range,
                                  E_range,
                                  exit_cond, 
                                  xctr,
                                  bw)
CA_tech = tech.CA_tech(device, device.is_VMP3, CA_user_params)

# Istantiate channel
test_options = ChannelOptions(experiment_name)
deisch1=DEISchannel(device,
                 1,
                 saving_dir,
                 test_options,
                 picoscope= pico,
                 is_live_plotting= True,
                 )

# Make sequence
sequence = [CA_tech]
deisch1.load_sequence(sequence, ask_ok=False)

# =============================================================================
# %% Start the measurement
# =============================================================================
awg_ch1.turn_on()
deisch1.start()


# =============================================================================
# %% End measurement
# =============================================================================
deisch1.stop()
pico.disconnect()
