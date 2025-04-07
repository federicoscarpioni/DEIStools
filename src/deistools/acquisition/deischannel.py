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
        print('Program should now execute saving')
        self.pico.save_block_calculation(f'/cycle_{self.potentiostat.current_loop}/sequence_{self.potentiostat.current_tech_index}')
        # self.awg.update(self.potentiostat.current_tech_index)