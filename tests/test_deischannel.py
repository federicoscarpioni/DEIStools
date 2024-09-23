from acquire.deischannel import DEISchannel
from pyeclab.device import BiologicDevice
from pyeclab.channel import Channel, ChannelOptions
import pyeclab.techniques as tech
from pypicostreaming.pypicostreaming.series5000.series5000 import Picoscope5000a


# Saving 
saving_dir = 'E:/Experimental_data/Federico/2024/python_software_test'
experiment_name = '2409161415_test_DEISchannel_class'

# Setting up piscoscope
# Measurment paramters
capture_size = 10000
samples_total = 10000000 
sampling_time = 2
sampling_time_scale = 'PS5000A_US'

# Connect i nstrument and perform the acquisiton
pico = Picoscope5000a('PS5000A_DR_14BIT')
saving_path = f'{saving_dir}/{experiment_name}'
pico.set_pico(capture_size, samples_total, sampling_time, sampling_time_scale, saving_path)
pico.set_channel('PS5000A_CHANNEL_A', 'PS5000A_500MV')
pico.set_channel('PS5000A_CHANNEL_B', 'PS5000A_2V', 0.1)
pico.set_channel('PS5000A_CHANNEL_C', 'PS5000A_500MV')

# IP address of the instrument
ip_address = '172.28.26.11'
# Path of the SDK from BioLogic installed in the machine
binary_path = "C:/EC-Lab Development Package/EC-Lab Development Package/"
# Istantiate device class
device = BiologicDevice(ip_address, binary_path = binary_path)

# Create CP technique
repeat_count   = 0
record_dt      = 1
record_dE      = 1    # Volts
current        = 0    # Ampers
duration_CP    = 10        # Seconds (sec * min * hours)
limit_E_crg    = 0b11111
E_lim_high     = 5        # Volts
E_lim_low      = -5         # Volts
i_range        = 9
exit_cond      = 1
E_range      = 0
bw           = 4
xctr           = 0b00001000
CP_user_params = tech.CPLIM_params(current, 
                                     duration_CP, 
                                     False, 
                                     0, 
                                     record_dt, 
                                     record_dE, 
                                     repeat_count, 
                                     i_range,
                                     E_range,
                                     exit_cond,
                                     xctr,
                                     E_lim_high,
                                     limit_E_crg, 
                                     bw)
CP_tech = tech.CPLIM_tech(device, device.is_VMP3, CP_user_params)

# Create loop technique
number_repetition  = 1
tech_index_start   = 0
LOOP_user_params = tech.LOOP_params(number_repetition, tech_index_start)
LOOP_tech    = tech.loop_tech(device, device.is_VMP3, LOOP_user_params)
# Make sequence
sequence = [CP_tech,LOOP_tech]
# Istantiate channel
test_options = ChannelOptions(experiment_name)
channel1=Channel(device,
                 1,
                 saving_dir,
                 test_options,
                 is_live_plotting= False
                 )
channel1.load_sequence(sequence, ask_ok=False)

deisch1 = DEISchannel(channel1, pico)


# %%
deisch1.run()

# %%
deisch1.stop()