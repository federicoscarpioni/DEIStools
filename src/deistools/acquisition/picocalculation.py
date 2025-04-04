from threading import Thread
from dataclasses import dataclass, field
from deistools.processing import BlockCalculator

@dataclass
class PicoCalculator:
    pico : ...
    block_calculator : BlockCalculator
    running : bool = field(default=False)

    def __post_init__ (self):
        self.run_thread = Thread(target=self._run)
        self.block_calculator.running = True

    def start(self):
        self.pico.run_streaming_non_blocking(autoStop = False)
        self.run_thread.start()
        self.running = True
    
    def stop(self):
        self.pico.stop()
        self.pico.disconnect()
        self.running = False
    
    def _run(self):
        while self.running:
            data_length = self.pico.channels['A'].buffer_total.get_lenght()
            if data_length > self.block_calculator.input_size:
                voltage_block = self.pico.convert_ADC_numbers(
                    self.pico.channels['A'].buffer_total.pop(self.block_calculator.input_size), 
                    self.channels['A'].vrange,
                    self.channels['A'].irange
                )
                current_block = self.convert_ADC_numbers(
                    self.channels['B'].buffer_total.pop(self.block_calculator.input_size),
                    self.channels['B'].vrange,
                    self.channels['B'].irange
                )
                self.computation_thread = Thread(target=self.block_calculator, args=(voltage_block, current_block,))
                self.computation_thread.start()