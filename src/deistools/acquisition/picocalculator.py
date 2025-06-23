from pathlib import Path
import json
from threading import Thread
from time import sleep
from dataclasses import dataclass, field
from typing import Union

from pypicostreaming import Picoscope4000, Picoscope5000a

from deistools.processing import BlockCalculator

@dataclass
class PicoCalculator:
    pico : Union[Picoscope4000, Picoscope5000a]
    block_calculator : BlockCalculator
    running : bool = field(default=False)
    computation_thread : Thread = field(init=False)

    def __post_init__ (self):
        self.run_thread = Thread(target=self._run)
        self.block_calculator.running = True

    def start(self):
        self.pico.run_streaming_non_blocking(autoStop = False)
        self.running = True
        self.run_thread.start()
    
    def stop(self):
        self.pico.stop()
        self.pico.disconnect()
        self.running = False
    
    def save_block_calculation(self, subfolder_name):
        if hasattr(self, 'computation_thread') and self.computation_thread.is_alive():
            self.computation_thread.join()
        saving_file_path = self.pico.saving_dir + subfolder_name
        Path(saving_file_path).mkdir(parents=True, exist_ok=True)
        self.block_calculator.save_results(saving_file_path)
        self.pico.empty_buffers()

    def _run(self):
        while self.running:
            data_length = self.pico.channels['A'].buffer_total.get_length()
            if data_length > self.block_calculator.input_size:
                voltage_block = self.pico.convert_ADC_numbers(
                    self.pico.channels['A'].buffer_total.pop(self.block_calculator.input_size), 
                    self.pico.channels['A'].vrange,
                    self.pico.channels['A'].conv_factor
                )
                current_block = self.pico.convert_ADC_numbers(
                    self.pico.channels['B'].buffer_total.pop(self.block_calculator.input_size),
                    self.pico.channels['B'].vrange,
                    self.pico.channels['B'].conv_factor
                )
                self.computation_thread = Thread(target=self.block_calculator.calculate, args=(voltage_block, current_block,))
                self.computation_thread.start()
            sleep(0.1)