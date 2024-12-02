from pyeclab.channel import Channel
from pypicostreaming.pypicostreaming.series5000.series5000 import Picoscope5000a    


class DEISchannel():
    def __init__(self, 
                 potentiostat_channel : Channel,
                 picoscope : Picoscope5000a,
                 trueform_awg = None):
        self.pot = potentiostat_channel
        self.pico = picoscope
        number_techniques = len([item for item in self.pot.sequence if item != "LOOP_tech"])
        self.sequence_counter = SequenceCounter(number_techniques)


    def run(self):
        self.pico.run_streaming_non_blocking()
        self.pot.callbacks.append(self.sequence_counter.update())
        self.pot.callbacks.append(self.save_pico_intermediate())
        self.pot.start()

    # def _control_loop(self):
    #     while self.pot.is_running():
    #         if self.pot.current_loop != self.pot.data_info.loop:
                
        

    def save_pico_intermediate(self):
        self.pico.save_intermediate_signals(f'/cycle_{self.sequence_counter.loop}/sequence_{self.sequence_counter.technique_index}')



    def stop(self):
            self.pot.stop()


class SequenceCounter:
    '''
    This class counts the current techniques index and number of loops of the technique in the potentiostat. It is
    created because it is not possible the get the correct number of loops and technique index from the Channel class
    that represents the instrument due to the concurrency of the processes (multi-threading).
    '''
    def __init__(self, techniques_in_sequence):
        self.loop = 0
        self.techniques_in_sequence = techniques_in_sequence
        self.technique_index = 0

    def update(self):
        self.technique_index = + 1
        if self.technique_index == self.techniques_in_sequence:
            self.loop =+ 1
            self.technique_index = 0

