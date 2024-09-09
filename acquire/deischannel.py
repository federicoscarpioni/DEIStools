class DEISchannel():
    def __init__(self, 
                 potentiostat_channel : Channel,
                 picoscope,
                 trueform_awg=None):
        self.pot = potentiostat_channel
        self.pico = picoscope,
        self.pot.callbacks.append(self.save_pico_intermediate())
        if trueform_awg != None:
            self.awg = trueform_awg
            # !!! Add the callback to change wave in the awg


    def run(self):
        self.awg.turn_on()
        self.pot.start()
        self.pico.run_streaming_non_blocking()

    def save_pico_intermediate(self):
        self.pico.convert_all_channels()
        self.pico.save_signals(f'cycle{self.pot.data_info.loop - 1}/sequence_{self.pot.current_tech_index}_tech_{self.pot.current_tech_id}')
        self.pico.reset_buffer()
