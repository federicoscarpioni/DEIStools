from pathlib import Path
import numpy as np

from deistools.acquisition import DEISchannel,ZPico5000a
from pyeclab import BANDWIDTH, E_RANGE, I_RANGE, BiologicDevice, Channel, ChannelConfig, FileWriter
from pyeclab.techniques import ChronoPotentiometry, Loop, generate_xctr_param

from trueformawg import import_awg_txt

# User parameters
# Result saves
saving_dir = 'E:/Experimental_data/Federico/2025/python_software_test'
experiment_name = '2503181527_deischannel_ca_with_advancedpico'
# Measurment paramters
capture_size = 300000000
samples_total = 300000000
sampling_time = 1
sampling_time_scale = 'PS5000A_US'
# STFFT-EIS parameters
sample_size = int(1e6*10) # sampling frequency * analysis time
frequencies = np.array([1000,100000])
irange = 0.1
# Downsampling parameters
filter_order = 8
cutoff = 90
ds_factor = int(2e-3/1e-6) # new time step / original time step
buffer_size = int(capture_size/ds_factor) # sampling time / acquisition time  (in s)
# Potentiostat and EC-lab SDK
ip_address = '172.28.26.10'
binary_path = "C:/EC-Lab Development Package/EC-Lab Development Package/"
device = BiologicDevice(ip_address, binary_path = binary_path)
config = ChannelConfig(
    record_ece=False,
    record_charge=False,
    live_plot=True,
    print_values=False,
    external_control=True,
    record_analog_In1=False,
    record_analog_In2=False,
)
# Create experimental technique
cp = ChronoPotentiometry(
    device=device,
    current=0,
    duration=30,
    vs_init=False,
    nb_steps=0,
    record_dt=1,
    record_dE=0.1,
    repeat=0,
    i_range=I_RANGE.I_RANGE_10uA,
    e_range=E_RANGE.E_RANGE_5V,
    bandwidth=BANDWIDTH.BW_4,
    xctr = generate_xctr_param(config)
)
cp.make_technique()
loop = Loop(
    device=device,
    repeat_N=0,
    loop_start=0
)
loop.make_technique()
sequence= [cp,loop]
# Set up oscilloscope
pico = ZPico5000a(sample_size,
                  frequencies,
                  irange,
                  filter_order,
                  cutoff,
                  ds_factor,
                  buffer_size,
                  'PS5000A_DR_14BIT')
saving_path = saving_dir +  '/' +  experiment_name
pico.set_pico(capture_size, samples_total, sampling_time, sampling_time_scale, saving_path)
pico.set_channel('PS5000A_CHANNEL_A', 'PS5000A_5V')
pico.set_channel('PS5000A_CHANNEL_B', 'PS5000A_200MV',irange=1e-5)
# Set up potentiostat
writer = FileWriter(
    file_dir=Path(saving_dir),
    experiment_name=experiment_name,
)
channel1=DEISchannel(
    device,
    1,
    writer=writer,
    config=config,
    picoscope = pico,
)
channel1.load_sequence(sequence, ask_ok=False, )
# Start devices separately
channel1.start()


