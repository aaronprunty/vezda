import numpy as np

# velocity of the constant background medium
# (only used if --medium=constant)
velocity = 1
peakFreq = 4
peakTime = 3

# definition of the time-dependent pulse function
def pulse(t):
    peakFreq = 4
    return np.sin(peakFreq * t) * np.exp(-1.6 * (t - peakTime)**2)
