import numpy as np
from pyeclab.api.kbio_tech import ECC_parm, make_ecc_parm, make_ecc_parms
from pyeclab.techniques.functions import reset_duration, set_duration_to_1s
from dataclasses import dataclass, field
from npbuffer import NumpyCircularBuffer

@dataclass(frozen = False)
class ConditionAverage:
    """
    Quantity should have the format of Channel.current_values for example: "Ewe",
    "Ece", "I" or "ElapsedTime". Operator instead can only be ">" or "<".
    Note that using the hardware limits of EC-Lab SDK Ece is not implemented.
    """
    technique_index : int
    quantity : str
    operator : str
    threshold : float
    num_elements : int
    buffer : NumpyCircularBuffer = field(init = False)

    def __post_init__(self):
        self.buffer = NumpyCircularBuffer(self.num_elements, dtype=np.float16)


def check_software_limits(deischannel):
    """
    Check if a certain averege condition (< or > of a treshold value) is met for a
    value of the sampled data over a certain number of points.
    """
    for condition in deischannel.conditions:
        if deischannel.potentiostat.data_info.TechniqueIndex == condition.technique_index:
          quantity_value = getattr(
               deischannel.potentiostat.current_values,
               condition.quantity,
          ) 
          condition.buffer.push(np.array(quantity_value)) 
          quantity_avarage = np.mean(condition.buffer.get_data())
          if condition.operator == ">" and quantity_avarage >= condition.threshold:
              condition.buffer.empty()
              return True
          elif condition.operator == "<" and quantity_avarage <= condition.threshold:
              condition.buffer.empty()
              return True
    return False

@dataclass
class WaveFormSequence:
    indexes : list[int]
    names : list[str]
    sample_rates : list[int]
    amplitudes : list[float]
