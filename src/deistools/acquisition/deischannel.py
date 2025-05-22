import numpy as np
from time import sleep
from threading import Thread
from dataclasses import dataclass, field

from pyeclab import Channel
from deistools.acquisition.utils import ConditionAverage, check_software_limits, WaveFormSequence
from deistools.acquisition.picocalculator import PicoCalculator
from trueformawg import TrueFormAWG


@dataclass
class DEISchannel:
    potentiostat : Channel
    pico : PicoCalculator
    awg : TrueFormAWG = field(default=None)
    waveforms_sequence : WaveFormSequence = field(default=None)
    conditions: list[ConditionAverage] = field(default_factory=list)
    running : bool = field(default=False)

    def __post_init__(self):
        self.run_thread = Thread(target=self._run)
        self.potentiostat.function = self._execute_on_technique_termination
    
    def start(self):
        if self.awg:
            self._update_waveform()
            if self.awg: self.awg.turn_on()
        self.potentiostat.start()
        self.pico.start()
        self.running = True
        self.run_thread.start()
        

    def stop(self):
        self.running = False

    def _run(self):
        while self.running:
            if check_software_limits(self):
                print("Software limit met!")
                self.potentiostat.end_technique()
            self.running = self.potentiostat.running
            sleep(1)
        self.stop_instruments()
        print('DEIS measurement completed')

    def stop_instruments(self):
        if self.potentiostat.running:
            self.potentiostat.stop()
        else:
            self._execute_on_technique_termination()
        for tentative in range(10):
            try:
                self.pico.stop()
                break
            except :
                print('Failed to communicate function to pico, device busy. New tentative..')
                sleep(1)
            print('Failed to stop and disconnect picoscope device.')
        if self.awg: self.awg.turn_off()

    def skip(self):
        self.potentiostat.end_technique()

    def _update_waveform(self):
        if self.running:
            if any(index == self.potentiostat.new_tech_index for index in self.waveforms_sequence.indexes):
                self.awg.turn_on()
            else :
                self.awg.turn_off()
        else :
            self.set_multisine(0)

    def set_multisine(self, list_index):
        self.awg.select_awf(self.waveforms_sequence.names[list_index])
        self.awg.set_sample_rate(self.waveforms_sequence.sample_rates[list_index])
        self.awg.set_amplitude(self.waveforms_sequence.amplitudes[list_index])

    def _execute_on_technique_termination(self):
        print('Program should now execute saving')
        self.pico.save_block_calculation(f'/cycle_{self.potentiostat.current_loop}_sequence_{self.potentiostat.current_tech_index}')
        if self.awg: self._update_waveform()