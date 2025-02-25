from TrueFormAWG.trueformawg.trueformawg import TrueFormAWG

class MultisineGenerator(TrueFormAWG):
    def __init__(self,
                 address,
                 channel,
                 sequence:list[str],
                 amplitudes:list[str]):
        self.sequence = sequence
        self.amplitudes = amplitudes
        super().__init__(address, channel)

    
    def update(self, index):
        if self.sequence[index] is None:
            self.turn_off()
        else:
            self.select_awf(self.sequence[index])
            self.turn_on()
        self.set_amplitude(self.amplitudes[index])
        print('AWG updated')
