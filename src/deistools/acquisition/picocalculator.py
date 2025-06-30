from pathlib import Path
import numpy as np
from threading import Thread
from time import sleep
from dataclasses import dataclass, field
from typing import Union

from pypicostreaming import Picoscope4000, Picoscope5000a

from deistools.processing import BlockCalculator

from pyeclab import Channel


@dataclass(frozen = False)
class ConditionAverageScope:
    """
    Quantity should be "current" or "voltage". Operator instead can only be ">" 
    or "<".
    """
    technique_index : int
    quantity : str
    operator : str
    threshold : float

@dataclass
class PicoCalculator:
    pico : Union[Picoscope4000, Picoscope5000a]
    block_calculator : BlockCalculator
    potentiostat : Channel
    running : bool = field(default=False)
    computation_thread : Thread = field(init=False)
    conditions: list[ConditionAverageScope] = field(default_factory=list)

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
                self.voltage_block = self.pico.convert_ADC_numbers(
                    self.pico.channels['A'].buffer_total.pop(self.block_calculator.input_size), 
                    self.pico.channels['A'].vrange,
                    self.pico.channels['A'].conv_factor
                )
                self.current_block = self.pico.convert_ADC_numbers(
                    self.pico.channels['B'].buffer_total.pop(self.block_calculator.input_size),
                    self.pico.channels['B'].vrange,
                    self.pico.channels['B'].conv_factor
                )
                self.computation_thread = Thread(target=self.block_calculator.calculate, args=(self.voltage_block, self.current_block,))
                self.computation_thread.start()
                if check_software_limits_scope(
                    self.conditions, 
                    self.potentiostat,
                    self.voltage_block,
                    self.current_block):
                        print("Software limit scope met!")
                        self.potentiostat.end_technique()
            sleep(0.5)


def check_software_limits_scope(conditions, potentiostat, voltage_block, current_block):
    """
    Check if a certain averege condition (< or > of a treshold value) is met for a
    value of the sampled data during the window of the online computation.
    """
    for condition in conditions:
        if potentiostat.data_info.TechniqueIndex == condition.technique_index:
            if condition.quantity == 'voltage':
                value_avarage = np.mean(voltage_block)
            if condition.quantity == 'current':
                value_avarage = np.mean(current_block)
            if condition.operator == ">" and value_avarage >= condition.threshold:
                print(f'{condition.quantity} > {condition.threshold}')
                return True
            elif condition.operator == "<" and value_avarage <= condition.threshold:
                print(f'{condition.quantity} < {condition.threshold}')
                return True
    return False
