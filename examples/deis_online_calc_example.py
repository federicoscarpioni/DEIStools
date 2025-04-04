from pathlib import Path
import json

import numpy as np

from pyeclab import BiologicDevice, ChannelConfig, FileWriter, Channel, BANDWIDTH, E_RANGE, I_RANGE
from pyeclab.techniques import ChronoAmperometry
from trueformawg import TrueFormAWG, VISAdevices, import_awg_txt
from pypicostreaming.series5000 import Picoscope5000a
from deistools.processing import MultiFrequencyAnalysis, fermi_dirac_filter, BlockCalculator
from deistools.acquisition import DEISchannel, PicoCalculator


# ===============
# User parameters
# ===============

saving_directory = 'E:/Experimental_data/Federico/2025/python_software_test/'
experiment_name = "2503101126"
saving_path = saving_directory + experiment_name

# Potentiostat
potentiostat_ip = "172.28.26.10"
eclabsdk_binary_path = "C:/EC-Lab Development Package/EC-Lab Development Package/"
potentiostat_channel = 1
ca_voltage=0
ca_duration=10
ca_vs_init=True
ca_nb_steps=0
ca_record_dt=1
ca_record_dI=1
ca_repeat=0
ca_e_range=E_RANGE.E_RANGE_5V
ca_i_range=I_RANGE.I_RANGE_100uA
ca_bandwidth=BANDWIDTH.BW_4

# Waveform generator
awgs = VISAdevices()
trueform_address = awgs.list[0]
awg_channel = 1
multisine_path = "E:/multisine_collection/2409131232multisine_1kHz-100mHz_8ptd_fgen10kHz_random_phases_flat_normalized/"
sample_rate = 10000
amplitude_pp = 0.050

# Digital oscilloscope
capture_size = 100000
samples_total = 100000
sampling_time = 1
sampling_time_scale = 'PS5000A_US'
bandwidth_chA = 'PS5000A_100MV'
bandwidth_chB = 'PS5000A_2V'
current_conversion_factor = 0.001

# Online calculation
frequencies = json.load(multisine_path + "waveform_metadata.json")["Frequencies / Hz"]
sampling_time = 1e-6
time_window = 100

# Decimation
filter_cutoff = 100
filter_order = 8
buffer_size = ca_duration/time_window
ds_factor = 1e6 / 5e2 # Sampling frequency / re-sampling_frequency



# ==========================
# Initialize devices objects
# ==========================

# Initialize potentiostat
device = BiologicDevice(potentiostat_ip, binary_path=eclabsdk_binary_path)
ca = ChronoAmperometry(
    device=device,
    voltage=ca_voltage,
    duration=ca_duration,
    vs_init=ca_vs_init,
    nb_steps=ca_nb_steps,
    record_dt=ca_record_dt,
    record_dI=ca_record_dI,
    repeat=ca_repeat,
    e_range=ca_e_range,
    i_range=ca_i_range,
    bandwidth=ca_bandwidth,
)
ca.make_technique()
sequence = [ca]
writer = FileWriter(
    file_dir=Path(saving_directory),
    experiment_name=experiment_name,
)
channel1 = Channel(
    device,
    potentiostat_channel,
    writer=writer,
    config=ChannelConfig(live_plot=True),
)
channel1.load_sequence(sequence)

# Initialize AWG
awg_ch1 = TrueFormAWG(trueform_address, awg_channel)
awg_ch1.clear_ch_mem()
multisine = import_awg_txt(multisine_path + "waveform.txt")
awg_ch1.load_awf('cstm_multisin', multisine) # Keep the name short or it gives an error
awg_ch1.avalable_memory()
awg_ch1.select_awf('cstm_multisin')
awg_ch1.set_Z_out_infinite()
awg_ch1.set_sample_rate(sample_rate)
awg_ch1.set_amplitude(amplitude_pp) 

# Initialize oscilloscope
pico = Picoscope5000a('PS5000A_DR_14BIT')
pico.set_pico(capture_size, samples_total, sampling_time, sampling_time_scale, saving_path)
pico.set_channel('PS5000A_CHANNEL_A', bandwidth_chA)
pico.set_channel('PS5000A_CHANNEL_B', bandwidth_chB, current_conversion_factor)

# Initialize the method for multi-frequency analysis
block_size =  time_window / sampling_time
high_z_calculator = MultiFrequencyAnalysis(
    frequencies, 
    np.zeroes(block_size),
    np.zeroes(block_size),
    sampling_time,
)
high_z_calculator.compute_freq_axis()

# Initialize the bock calculator

block_calculator = BlockCalculator(
    imput_size = block_size,
    sampling_time = sampling_time,
    high_z_calculator = high_z_calculator,
    lp_filter = fermi_dirac_filter(high_z_calculator.freq_axis, 0, 2 * filter_cutoff, filter_order),
    ds_factor = ds_factor,
    buffer_size =buffer_size,
    saving_dir = saving_path
)

# Inject block calculator into PicoCalculator

pico_calculator = PicoCalculator(
    pico = pico,
    block_calculator= block_calculator,
)

# Inject instrument object into DEISchannel object

deischannel = DEISchannel(
    potentiostat = channel1,
    pico = pico_calculator,
    awg=awg_ch1,
)

# =====================
# Start the measurement
# =====================

deischannel.start()