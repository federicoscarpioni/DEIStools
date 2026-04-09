import numpy as np

# def compute_baseline(signal):
    
#     xstart = 0
#     ystart = signal[0]
#     xfinish = signal.size
#     yfinish = signal[-1]
    
#     m = (yfinish - ystart) / (xfinish - xstart)
#     q = yfinish - (m * xfinish)
    
#     line = np.arange(xstart, xfinish, 1) * m + q 
    
#     return m, q,


# def remove_baseline(signal, m, q):
#     line = np.arange(signal[0], signal[-1], 1) * m + q 
#     return signal - line


# def redo_baseline(signal, m, q):
#     line = np.arange(signal[0], signal[-1], 1) * m + q   
#     return signal + line

#------------------------------------------------------------------------------#

def remove_baseline(signal, sampling_time):
    'Tested on 22.03 working preperly'
    coordinates = {'xstart'  : 0,
                   'ystart'  : signal[0],
                   'xfinish' : signal.size * sampling_time,
                   'yfinish' : signal[-1]
                   }    
    m = (coordinates['yfinish'] - coordinates['ystart']) / (coordinates['xfinish'] - coordinates['xstart'])
    q = coordinates['yfinish'] - (m * coordinates['xfinish'])
    line = np.arange(coordinates['xstart'], coordinates['xfinish'], sampling_time) * m + q 
    signal = signal - line   
    return signal, coordinates

#------------------------------------------------------------------------------#

def redo_baseline(signal, coordinates): 
    'Tested on 22.03 working preperly'
    m = (coordinates['yfinish'] - coordinates['ystart']) / (coordinates['xfinish'] - coordinates['xstart'])
    q = coordinates['yfinish'] - (m * coordinates['xfinish'])
    line = np.linspace(coordinates['xstart'], coordinates['xfinish'], signal.size) * m + q  
    signal = signal + line  
    return signal