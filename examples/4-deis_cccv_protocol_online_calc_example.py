'''
Note: always use duration of techniques longer than the window time of the 
block calculation. Otherwise, the code throughs an error because trys to save
the results when the calculation has not yet happen.
'''
from pathlib import Path
import json

import numpy as np

from pyeclab import BiologicDevice, ChannelConfig, FileWriter, Channel, BANDWIDTH, E_RANGE, I_RANGE
from pyeclab.techniques import ChronoAmperometry, ChronoPotentiometry, OpenCircuitVoltage, Loop
from trueformawg import TrueFormAWG, VISAdevices, import_awg_txt
from pypicostreaming import Picoscope5000a
from deistools.processing import MultiFrequencyAnalysis, fermi_dirac_filter, BlockCalculator
from deistools.acquisition import DEISchannel, PicoCalculator


# ===============
# User parameters
# ===============

saving_directory = 'E:/Experimental_data/Federico/2025/python_software_test/'
experiment_name = "2504101440_test_DEISchannel_with_dependency_injection_cccv_protocol"
saving_path = saving_directory + experiment_name 

# Potentiostat
potentiostat_ip = "172.28.26.10"
eclabsdk_binary_path = "C:/EC-Lab Development Package/EC-Lab Development Package/"
potentiostat_channel = 1
# values for all techniqes
i_range=I_RANGE.I_RANGE_10mA
e_range=E_RANGE.E_RANGE_5V
bandwidth=BANDWIDTH.BW_9
# chrono-potentiometry charge technique
cpc_current=0
cpc_duration=30
cpc_vs_init=False
cpc_nb_steps=0
cpc_record_dt=1
cpc_record_dE=0.1
cpc_repeat=0
# chrono-amperometry technique
ca_voltage=0
ca_duration=30
ca_vs_init=True
ca_nb_steps=0
ca_record_dt=1
ca_record_dI=1
ca_repeat=0
# chrono-potentiometry discharge technique
cpd_current=0
cpd_duration=30
cpd_vs_init=False
cpd_nb_steps=0
cpd_record_dt=1
cpd_record_dE=0.1
cpd_repeat=0
# ocv technique
ocv_duration= 15
ocv_record_dt=1
# loop technique
loop_repeat_N=1 
loop_start=0

# Waveform generator
trueform_address = 'USB0::0x0957::0x4B07::MY59000581::0::INSTR'
awg_channel = 1
multisine_path = "E:/multisine_collection/2409131232multisine_1kHz-100mHz_8ptd_fgen10kHz_random_phases_flat_normalized/"
sample_rate = 10000
amplitude_pp = 0.050 # in V

# Digital oscilloscope
pico_samples_total = 50000000
pico_capture_size = pico_samples_total
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
cpc = ChronoPotentiometry(
    device=device,
    current=cpc_current,
    duration=cpc_duration,
    vs_init=cpc_vs_init,
    nb_steps=cpc_nb_steps,
    record_dt=cpc_record_dt,
    record_dE=cpc_record_dE,
    repeat=cpc_repeat,
    i_range=i_range,
    e_range=e_range,
    bandwidth=bandwidth,
)
cpc.make_technique()
ca = ChronoAmperometry(
    device=device,
    voltage=ca_voltage,
    duration=ca_duration,
    vs_init=ca_vs_init,
    nb_steps=ca_nb_steps,
    record_dt=ca_record_dt,
    record_dI=ca_record_dI,
    repeat=ca_repeat,
    e_range=e_range,
    i_range=i_range,
    bandwidth=bandwidth,
)
ca.make_technique()
cpd = ChronoPotentiometry(
    device=device,
    current=cpd_current,
    duration=cpd_duration,
    vs_init=cpd_vs_init,
    nb_steps=cpd_nb_steps,
    record_dt=cpd_record_dt,
    record_dE=cpd_record_dE,
    repeat=cpd_repeat,
    i_range=i_range,
    e_range=e_range,
    bandwidth=bandwidth,
)
cpd.make_technique()
ocv = OpenCircuitVoltage(
    device=device,
    duration= ocv_duration,
    record_dt=ocv_record_dt,
    e_range=e_range,
    bandwidth=bandwidth,
)
ocv.make_technique()
loop = Loop(
    device=device, 
    repeat_N=loop_repeat_N, 
    loop_start=loop_start,
)
loop.make_technique()
sequence = [
    cpc,
    ca,
    ocv,
    cpd,
    ocv,
    loop,
]
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
