from trueformawg.trueformawg import TrueFormAWG

class MultisineGenerator:
    def __init__(self,
                 awg:TrueFormAWG,
                 sequence:list[str],
                 amplitudes:list[str]):
        self.sequence = sequence
        self.amplitudes = amplitudes
        # Import all attributes from the awg instance
        for attr, value in vars(awg).items():
            setattr(self, attr, value)

    def update(self, index):
        if self.sequence[index] is None:
            self.turn_off()
        else:
            self.select_awf(self.sequence[index])
            self.turn_on()
        self.set_amplitude(self.amplitudes[index])
