from pathlib import Path
import json

import numpy as np

from pyeclab import BiologicDevice, ChannelConfig, FileWriter, Channel, BANDWIDTH, E_RANGE, I_RANGE
from pyeclab.techniques import ChronoAmperometry
from trueformawg import TrueFormAWG, VISAdevices, import_awg_txt
from pypicostreaming import Picoscope5000a
from deistools.processing import MultiFrequencyAnalysis, fermi_dirac_filter, BlockCalculator
from deistools.acquisition import DEISchannel, PicoCalculator


# ===============
# User parameters
# ===============

saving_directory = 'E:/Experimental_data/Federico/2025/python_software_test/'
experiment_name = "2504071613_test_DEISchannel_with_dependency_injection_30s_auto_saving"
saving_path = saving_directory + experiment_name 

# Potentiostat
potentiostat_ip = "172.28.26.10"
eclabsdk_binary_path = "C:/EC-Lab Development Package/EC-Lab Development Package/"
potentiostat_channel = 1
ca_voltage=0
ca_duration=30
ca_vs_init=True
ca_nb_steps=0
ca_record_dt=1
ca_record_dI=1
ca_repeat=0
ca_e_range=E_RANGE.E_RANGE_5V
ca_i_range=I_RANGE.I_RANGE_10mA
ca_bandwidth=BANDWIDTH.BW_4

# Waveform generator
trueform_address = 'USB0::0x0957::0x4B07::MY59000581::0::INSTR'
awg_channel = 1
multisine_path = "E:/multisine_collection/2409131232multisine_1kHz-100mHz_8ptd_fgen10kHz_random_phases_flat_normalized/"
sample_rate = 10000
amplitude_pp = 0.050

# Digital oscilloscope
pico_capture_size = 50000000
pico_samples_total = 50000000
pico_sampling_time = 1
pico_sampling_time_scale = 'PS5000A_US'
pico_bandwidth_chA = 'PS5000A_5V'
pico_bandwidth_chB = 'PS5000A_500MV'
pico_current_conversion_factor = 0.01

# Online calculation
frequencies = json.load(open(multisine_path + "waveform_metadata.json"))["Frequencies / Hz"]
frequencies = np.array(frequencies[11:])
sampling_time = 1e-6
time_window = 10

# Decimation
filter_cutoff = 8
filter_order = 8
time_experiment = ca_duration
sampling_frequency = 1e6
resampling_frequency = 5e2
ds_factor = int(sampling_frequency // resampling_frequency )
buffer_size = int(time_experiment * resampling_frequency)


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
awg_ch1.load_awf('multisine', multisine) # Keep the name short or it gives an error
awg_ch1.avalable_memory()
awg_ch1.select_awf('multisine')
awg_ch1.set_Z_out_infinite()
awg_ch1.set_sample_rate(sample_rate)
awg_ch1.set_amplitude(amplitude_pp) 

# Initialize oscilloscope
pico = Picoscope5000a('PS5000A_DR_14BIT')
pico.set_pico(
    pico_capture_size, 
    pico_samples_total, 
    pico_sampling_time, 
    pico_sampling_time_scale, 
    saving_path,
)
pico.set_channel(
    'PS5000A_CHANNEL_A', 
    pico_bandwidth_chA,
)
pico.set_channel(
    'PS5000A_CHANNEL_B', 
    pico_bandwidth_chB, 
    pico_current_conversion_factor,
)

# Initialize the method for multi-frequency analysis
block_size =  int(time_window // sampling_time)
high_z_calculator = MultiFrequencyAnalysis(
    frequencies, 
    np.zeros(block_size),
    np.zeros(block_size),
    sampling_time,
)
high_z_calculator.compute_freq_axis()

# Initialize the bock calculator

block_calculator = BlockCalculator(
    input_size = block_size,
    sampling_time = sampling_time,
    high_z_calculator = high_z_calculator,
    lp_filter = fermi_dirac_filter(high_z_calculator.freq_axis, 0, 2 * filter_cutoff, filter_order),
    ds_factor = ds_factor,
    buffer_size =buffer_size,
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
