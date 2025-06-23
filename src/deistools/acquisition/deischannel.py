import numpy as np
from time import sleep
import json
from threading import Thread
from dataclasses import dataclass, field, asdict
from typing import Union

from pyeclab import Channel
from deistools.acquisition.utils import ConditionAverage, check_software_limits, WaveFormSequence, condition_avarage_serialization_factory
from deistools.acquisition.picocalculator import PicoCalculator
from deistools.acquisition.multisinegen import MultisineGenerator, MultisineGeneratorCombined, custom_serialization_factory


@dataclass
class DEISchannel:
    potentiostat : Channel
    pico : PicoCalculator # This should be also a normal pico
    frequencies : np.array
    awg : Union[MultisineGenerator, MultisineGeneratorCombined] = field(default=None)
    conditions: list[ConditionAverage] = field(default_factory=list)
    running : bool = field(default=False)


    def __post_init__(self):
        self.run_thread = Thread(target=self._run)
        self.potentiostat.function = self._execute_on_technique_termination
    

    def start(self):
        self.save_metadata()
        if self.awg:
            self._update_waveform()
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
            self.awg.update(self.potentiostat.new_tech_index)
        else :
            self.awg.update(0)


    def _execute_on_technique_termination(self):
        print('Program should now execute saving')
        self.pico.save_block_calculation(f'/cycle_{self.potentiostat.current_loop}_sequence_{self.potentiostat.current_tech_index}')
        if self.awg: self._update_waveform()


    def save_metadata(self):
        metadata_dict = {
            'Software limits avarage' : [asdict(instance, dict_factory=condition_avarage_serialization_factory) for instance in self.conditions],
            'AWG controlled' : 'No' if self.awg == None else 'Yes',
            'Frequencies multisine (Hz)': self.frequencies.tolist(),
        }
        if type(self.awg) == MultisineGenerator :
            metadata_dict.update(
                {
                    'Channel number' : self.awg.channel.channel_num,
                    'Multisines names sequence': self.awg.waveforms_names,
                    'Sequence indexes': self.awg.sequence_indexes,
                    'Waveform alias sequence': self.awg.names,
                    'Sample rates sequence': self.awg.sample_rates,
                    'Amplitudes sequence' : self.awg.amplitudes,
                }
            )
        if type(self.awg) == MultisineGeneratorCombined :
            metadata_dict.update(
                {
                    'Channel 1' : {
                        'Multisines names sequence': self.awg.channel1.waveforms_names,
                        'Sequence indexes': self.awg.channel1.sequence_indexes,
                        'Waveform alias sequence': self.awg.channel1.names,
                        'Sample rates sequence': self.awg.channel1.sample_rates,
                        'Amplitudes sequence' : self.awg.channel1.amplitudes,
                    },
                    'Channel 2' : {
                        'Multisines names sequence': self.awg.channel2.waveforms_names,
                        'Sequence indexes': self.awg.channel2.sequence_indexes,
                        'Waveform alias sequence': self.awg.channel2.names,
                        'Sample rates sequence': self.awg.channel2.sample_rates,
                        'Amplitudes sequence' : self.awg.channel2.amplitudes,
                    },
                }
            )
        if type(self.pico) == PicoCalculator:
            index_remaining_frequencies = self.frequencies.size - self.pico.block_calculator.high_z_calculator.frequencies.size
            metadata_dict.update({
                'Window length (Sa)' : self.pico.block_calculator.input_size,
                'STFFT-EIS frequencies (Hz)' : self.pico.block_calculator.high_z_calculator.frequencies.tolist(),
                'Low-pass filter' : type(self.pico.block_calculator.lp_filter).__name__,
                'Low-pass filter order' : self.pico.block_calculator.lp_filter.order,
                'Low-pass cut-off frequency': self.pico.block_calculator.lp_filter.cutoff,
                'Frequencies remaining (Hz)': self.frequencies[:index_remaining_frequencies].tolist(),
            }
            )
        with open(str(self.potentiostat.writer.file_dir / self.potentiostat.writer.experiment_name) +'/metadata_deis_exp.json', 'w') as fp:
            json.dump(metadata_dict, fp)