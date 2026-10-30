
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

fs = 8192

# Build a periodic tone. amps[m] is the amplitude of the (m+1)-th harmonic
def additive(t, f0, amps, fs):

    x = np.zeros_like(t)
    for m in range(len(amps)):
        k = m + 1
        if k*f0 < fs/2:
            x = x + amps[m] * np.sin(2 * np.pi * k * f0 * t) # Complete: add the k-th harmonic, see Eq. (1) of the manual
    return x

def envelope(t, t_a=0.01, tau_d=0.15):
    # returns t/t_a during the rise (t < t_a), exp decay afterwards
    return np.where(t < t_a, t/t_a, np.exp(-(t-t_a)/tau_d))

freqs = {"C4":261.63, "D4":293.66, "E4":329.63, "F4":349.23, "G4":392.00, "A4":440.00}

song = [("C4",0.5),("C4",0.5),("G4",0.5),("G4",0.5),("A4",0.5),("A4",0.5),("G4",1.0),("F4",0.5),("F4",0.5),("E4",0.5),("E4",0.5),("D4",0.5),("D4",0.5),("C4",1.0)]
my_amps = [1, 1.3, 1, 0.9, 0, 0.65, 0.6]


clip = []
for name, duration in song:
    tn = np.arange(0, duration, 1.0/fs)
    note = additive(tn, freqs[name], my_amps, fs)

    clip.append(note * envelope(tn))

clip = np.concatenate(clip)
clip = clip/np.max(np.abs(clip))
clip_i16 = np.int16(clip * 32767) # converts to 16 bit int for wav format
wavfile.write("clip.wav", round(fs), clip_i16) # writes the wav file to disk