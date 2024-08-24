from pybiologic import Channel

class DEISchannel(Channel):
    def __init__(self, 
                 bio_device : BiologicDevice, 
                 channel_num : int, 
                 saving_dir : str,
                 channel_options : namedtuple,
                 do_live_plot : bool = True, # ? Deside which naming convention to use for booleans
                 do_record_Ece : bool = False,
                 do_record_analog_in1 : bool = False,
                 do_record_analog_in2 : bool = False,
                 do_print_values : bool  = False,
                 picoscope = None, 
                 awg = None):

        super().__init__(bio_device, 
                         channel_num,
                         saving_dir,
                         channel_options,
                         do_live_plot,
                         do_record_Ece,
                         do_record_analog_in1,
                         do_record_analog_in2,
                         )
                         
        self.pico = piscoscope,
        self.awg = awg

    def run(self):
        self.start()
        pico.run_streaming_non_blocking()