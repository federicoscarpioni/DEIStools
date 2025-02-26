from pyeclab.channel import Channel
from pyeclab.device import BiologicDevice
from pypicostreaming import Picoscope5000a  
from collections import namedtuple  
from threading import Thread


class DEISchannel(Channel):
    def __init__(self,
                 bio_device: BiologicDevice,
                 channel_num: int,
                 saving_dir: str,
                 channel_options: namedtuple,
                 picoscope: Picoscope5000a = None,
                 trueformawg = None,
                 is_live_plotting: bool = True,  # ? Deside which naming convention to use for booleans
                 is_recording_Ece: bool = False,
                 is_external_controlled: bool = False,
                 is_recording_analog_In1: bool = False,
                 is_recording_analog_In2: bool = False,
                 is_charge_recorded: bool = False,
                 is_printing_values: bool = False,
                 callbacks=[],
                 
                 ):
        super().__init__(bio_device,
                 channel_num,
                 saving_dir,
                 channel_options,
                 is_live_plotting,  
                 is_recording_Ece,
                 is_external_controlled,
                 is_recording_analog_In1,
                 is_recording_analog_In2,
                 is_charge_recorded,
                 is_printing_values,
                 callbacks=[])
        self.pico = picoscope
        self.awg = trueformawg

    def end_technique(self):
        if self.pico is not None:
            save_intermediate_pico  = Thread(target=self.pico.save_intermediate_signals, args=(f'/loop_{self.current_loop}/technique_{self.current_techn_index}',))
            save_intermediate_pico.start()
        super().end_technique()
        

    
    def _update_sequence_trackers(self):
        if self.pico is not None:
            save_intermediate_pico  = Thread(target=self.pico.save_intermediate_signals, args=(f'/loop_{self.current_loop}/technique_{self.current_tech_index}',))
            save_intermediate_pico.start()
        super()._update_sequence_trackers()
        if self.awg is not None: self.awg.update(self.current_tech_index)

        
    def _final_actions(self):
        super()._final_actions()
        if self.pico is not None:
            self.pico.save_intermediate_signals(f'/loop_{self.current_loop}/technique_{self.current_tech_index}')
            self.pico.stop()
        if self.awg is not None: self.awg.turn_off()
        
    
    def start(self):
        if self.awg is not None: 
            self.awg.update(self.current_tech_index)
            self.awg.turn_on()
        if self.pico is not None: self.pico.run_streaming_non_blocking(autoStop=False)
        super().start()
        
    
    def stop(self):
        super().stop()
        if self.pico is not None: self.pico.stop()
        if self.awg is not None: self.awg.turn_off()