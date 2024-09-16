from pyeclab.channel import Channel
from pypicostreaming.pypicostreaming.series5000.series5000 import Picoscope5000a    


class DEISchannel():
    def __init__(self, 
                 potentiostat_channel : Channel,
                 picoscope : Picoscope5000a,
                 trueform_awg = None):
        self.pot = potentiostat_channel
        self.pico = picoscope


    def run(self):
        self.pico.run_streaming_non_blocking()
        self.pot.start()
        self.pot.callbacks.append(self.save_pico_intermediate)


    def save_pico_intermediate(self):
        self.pico.save_intermediate_signals(f'/cycle_{self.pot.current_loop-1}/sequence_{self.pot.current_tech_index}')


def stop(self):
        self.pot.stop()
        self.pico.stop()
        self.save_pico_intermediate()
