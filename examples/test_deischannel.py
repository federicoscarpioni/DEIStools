from pathlib import Path

from pyeclab import BANDWIDTH, E_RANGE, I_RANGE, BiologicDevice, ChannelConfig, FileWriter
from pyeclab.techniques import OpenCircuitVoltage
from deistools.acquisition import DEISchannel
from pypicostreaming import Picoscope5000a

# User parameters
# Result saves
saving_dir = 'E:/Experimental_data/Federico/2025/python_software_test'
experiment_name = '2503181005_deischannel_ocv_3sec_with_pico'
# Potentiostat and EC-lab SDK
ip_address = '172.28.26.10'
binary_path = "C:/EC-Lab Development Package/EC-Lab Development Package/"


# Setting up piscoscope
# Measurment paramters
capture_size = 1500000
samples_total = 1500000 
sampling_time = 2
sampling_time_scale = 'PS5000A_US'

# # Connect i nstrument and perform the acquisiton
pico = Picoscope5000a('PS5000A_DR_14BIT')
saving_path = f'{saving_dir}/{experiment_name}'
pico.set_pico(capture_size, samples_total, sampling_time, sampling_time_scale, saving_path)
pico.set_channel('PS5000A_CHANNEL_A', 'PS5000A_5V')
# pico.set_channel('PS5000A_CHANNEL_B', 'PS5000A_500MV', 0.1)



# Istantiate device class
device = BiologicDevice(ip_address, binary_path = binary_path)

# Create experimental technique
ocv = OpenCircuitVoltage(
    device=device,
    duration= 10,
    record_dt=1,
    e_range=E_RANGE.E_RANGE_5V,
    bandwidth=BANDWIDTH.BW_4,
)
ocv.make_technique()
sequence = [ocv]


# Istantiate channel
writer = FileWriter(
    file_dir=Path(saving_dir),
    experiment_name=experiment_name,
)
config = ChannelConfig(
    record_ece=False,
    record_charge=False,
    live_plot=True,
    print_values=False,
    external_control=False,
    record_analog_In1=False,
    record_analog_In2=False,
)
channel1=DEISchannel(
    device,
    1,
    writer=writer,
    config=config,
    picoscope=pico,
)
channel1.load_sequence(sequence, ask_ok=False, )

# %%
channel1.start()
