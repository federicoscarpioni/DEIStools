from DEIStools.acquire.deischannel import DEISchannel
from DEIStools.acquire.multisinegen import MultisineGenerator
from pyeclab.device import BiologicDevice
from pyeclab.channel import Channel, ChannelOptions
import pyeclab.techniques as tech
from TrueFormAWG.trueformawg.trueformawg import TrueFormAWG, VISAdevices, import_awg_txt

awgs = VISAdevices()

trueform_address = awgs.list[1]
awg_ch1 = TrueFormAWG(trueform_address,1)
awg_ch1.clear_ch_mem()
sin1hz_path = 'E:/multisine_collection/2412121310_signle_sine_1Hz/waveform.txt'
sin1hz = import_awg_txt(sin1hz_path)
awg_ch1.load_awf('sin1hz', sin1hz)
awg_ch1.select_awf('sin1hz')
awg_ch1.set_Z_out_infinite()
awg_ch1.set_sample_rate(10)

sin1hz_path = 'E:/multisine_collection/2412121310_signle_sine_100Hz/waveform.txt'
sin1hz = import_awg_txt(sin1hz_path)
awg_ch1.load_awf('sin100hz', sin1hz)
awg_ch1.select_awf('sin100hz')
awg_ch1.set_Z_out_infinite()
awg_ch1.set_sample_rate(10)


# Set up waveform sequence
awg_seq = ['sin1hz','sin100hz','sin1hz']
awg_ampli = [0.05,0.1,0.05]
msgen = MultisineGenerator(trueform_address,1, awg_seq, awg_ampli)

# Saving 
saving_dir = 'E:/Experimental_data/Federico/2024/python_software_test'
experiment_name = '2412141007_test_switching_waveform'

# IP address of the instrument
ip_address = '172.28.26.11'
# Path of the SDK from BioLogic installed in the machine
binary_path = "C:/EC-Lab Development Package/EC-Lab Development Package/"
# Istantiate device class
device = BiologicDevice(ip_address, binary_path = binary_path)

# Create CA technique
E_range        = 0
bw             = 8
repeat_count   = 0
record_dt      = 1
record_dI      = 1   # Ampers
voltage        = 0   # Volts
i_range        = 8
duration_CA    = 10     # Seconds (sec * min * hours)
exit_cond      = 8
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

# Create CP technique
E_range        = 0
bw             = 8
repeat_count   = 0
record_dt      = 1
record_dE      = 1    # Volts
current        = 0    # Ampers
duration_CP    = 10        # Seconds (sec * min * hours)
limit_E_crg    = 0b11111
E_lim_high     = 5        # Volts
E_lim_low      = -5         # Volts
i_range        = 8
exit_cond      = 1
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

# Istantiate channel
test_options = ChannelOptions(experiment_name)
deisch2=DEISchannel(device,
                 2,
                 saving_dir,
                 test_options,
                 trueformawg = msgen,
                 is_live_plotting= True,
                 )

# Make sequence
sequence = [CP_tech,CA_tech,CP_tech]
deisch2.load_sequence(sequence, ask_ok=False, )

deisch2.start()
