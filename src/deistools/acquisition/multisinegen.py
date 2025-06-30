from trueformawg import TrueFormAWG
from dataclasses import dataclass

@dataclass
class WaveFormSequence:
    indexes : list[int]
    names : list[str]
    sample_rates : list[int]
    amplitudes : list[float]
    
@dataclass
class MultisineGenerator:
    channel : TrueFormAWG
    waveforms_names : list[str]
    sequence_indexes : list[int]
    names : list[str]
    sample_rates : list[int]
    amplitudes: list[float]

    def turn_on(self):
        self.channel.turn_on()

    def turn_off(self):
        self.channel.turn_off()
    
    def combine_channels(self):
        self.channel.combine_channels()
    
    def set_amplitude(self, value):
        self.channel.set_amplitude(value)
    
    def set_waveform(self, name):
        self.channel.select_awf(name)

    def set_sample_rate(self, sample_rate):
        self.channel.set_sample_rate(sample_rate)

    def update(self, tech_index):
        if any(index == tech_index for index in self.sequence_indexes):
            list_index = self.sequence_indexes.index(tech_index)
            self.set_waveform(self.names[list_index])
            self.set_amplitude(self.amplitudes[list_index])
            self.set_sample_rate(self.sample_rates[list_index])
            self.turn_on()
        else:
            self.turn_off()


@dataclass
class MultisineGeneratorCombined:
    channel1 : MultisineGenerator
    channel2 : MultisineGenerator
    waveforms_names : list[str]

    def __post_init__(self):
        self.channel1.combine_channels()

    def turn_on(self):
        self.channel1.turn_on()
        self.channel2.turn_on()

    def turn_off(self):
        self.channel1.turn_off()
        self.channel2.turn_off()
    
    def update(self, tech_index):
        self.channel1.update(tech_index)
        self.channel2.update(tech_index)


def custom_serialization_factory(data):
    result_dict = {}
    for field_name, filed_value in data:
        if field_name == 'channel' :
            continue
        result_dict[field_name] = filed_value
    return result_dict