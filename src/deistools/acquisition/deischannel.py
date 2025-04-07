import numpy as np
from time import sleep
from threading import Thread
from dataclasses import dataclass, field

from pyeclab import Channel
from deistools.acquisition.picocalculator import PicoCalculator
from trueformawg import TrueFormAWG


@dataclass
class DEISchannel:
    potentiostat : Channel
    pico : PicoCalculator # I have no general class for pico yet
    awg : TrueFormAWG
    running : bool = field(default=False)

    def __post_init__(self):
        self.run_thread = Thread(target=self._run)
        self.potentiostat.function = self._execute_on_technique_termination
    
    def start(self):
        self.awg.turn_on()
        self.potentiostat.start()
        self.pico.start()
        self.running = True
        self.run_thread.start()
        

    def stop(self):
        self.running = False

    def _run(self):
        while self.running:
            self.running = self.potentiostat.running 
            sleep(1)
        self._stop_instruments()

    def _stop_instruments(self):
        if self.potentiostat.running:
            self.potentiostat.stop()
        self.pico.stop()
        self.awg.turn_off()

    def _execute_on_technique_termination(self):
        self.pico.block_calculator.finish_technique()
        self.awg.update(self.potentiostat.current_tech_index)

        

# class DEISchannel(Channel):
#     def __init__(
#         self,
#         bio_device: BiologicDevice,
#         channel_num: int,
#         writer: FileWriter,
#         config: ChannelConfig,
#         picoscope = None,
#         trueformawg = None,
#         callbacks: list | None = None,
#     ):
#         super().__init__(
#             bio_device,
#             channel_num,
#             writer,
#             config,
#             callbacks,
#         )
#         self.pico = picoscope
#         self.awg = trueformawg

#     def start(self):
#         if self.awg is not None: 
#             self.awg.update(self.current_tech_index)
#             self.awg.turn_on()
#         if self.pico is not None: self.pico.run_streaming_non_blocking(autoStop=False)
#         super().start()
        
    
#     def stop(self):
#         super().stop()
#         if self.pico is not None: self.pico.stop()
#         if self.awg is not None: self.awg.turn_off()

#     def end_technique(self):
#         if self.pico is not None:
#             save_intermediate_pico  = Thread(target=self.pico.save_intermediate_signals, args=(f'/loop_{self.current_loop}/technique_{self.current_techn_index}',))
#             save_intermediate_pico.start()
#         super().end_technique()

#     def set_condition(self, technique_index: int, quantity: str, operator: str, threshold: float, num_samples:int):
#         buffer = NumpyCircularBuffer(num_samples, np.float32)
#         self.conditions.append((technique_index, quantity, operator, threshold, buffer))

#     def _check_software_limits(self):
#         """
#         Check if a certain condition (< or > of a trashold value) is met for a
#         value of the sampled data over a certain number of points.
#         """
#         for (
#             technique_index,
#             quantity,
#             operator,
#             threshold,
#             buffer
#         ) in (
#             self.conditions
#         ):  # ? Can I manually add other attributes to current_values for the quantities that are missing?
#             if self.data_info.TechniqueIndex == technique_index:
#                 quantity_value = getattr(
#                     self.current_values, quantity, None
#                 )  # ! It works only for attributes of current_data. I need onther trick to make it work also for capacity or power
#                 if quantity_value is None:
#                     continue
#                 else:
#                     buffer.push(quantity_value)
#                 if operator == ">" and np.mean(buffer.get_data()) >= threshold:
#                     return True
#                 elif operator == "<" and np.mean(buffer.get_data()) <= threshold:
#                     return True
#         return False

        
#     def _update_sequence_trackers(self):
#         if self.pico is not None:
#             save_intermediate_pico  = Thread(target=self.pico.save_intermediate_signals, args=(f'/loop_{self.current_loop}/technique_{self.current_tech_index}',))
#             save_intermediate_pico.start()
#         super()._update_sequence_trackers()
#         if self.awg is not None: self.awg.update(self.current_tech_index)

        
#     def _final_actions(self):
#         super()._final_actions()
#         if self.pico is not None:
#             self.pico.save_intermediate_signals(f'/loop_{self.current_loop}/technique_{self.current_tech_index}')
#             self.pico.stop()
#         if self.awg is not None: self.awg.turn_off()
        
    
