import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

fs = 8192
f0 = 220.0
t = np.arange(0, 2.0, 1.0/fs)

# Build a periodic tone. amps[m] is the amplitude of the (m+1)-th harmonic
def additive(t, f0, amps, fs):

    x = np.zeros_like(t)
    for m in range(len(amps)):
        k = m + 1
        if k*f0 < fs/2:
            x = x + amps[m] * np.sin(2 * np.pi * k * f0 * t) # Complete: add the k-th harmonic, see Eq. (1) of the manual
    return x

# only the fundamental is nonzero
pure = np.zeros(20)
pure[0] = 1

# odd harmonics with amplitude 1/k, even harmonics 0
odd_1k = np.zeros(20)
for i in range(len(odd_1k)):
    k = i + 1
    if k % 2 != 0:
        odd_1k[i] = float(1/k)

# all harmonics with amplitude 1/k 
all_1k = np.zeros(20)
for i in range(len(all_1k)):
    k = i + 1
    all_1k[i] = float(1/k)

# all harmonics with amplitude 1/k**2
all_1k2 = np.zeros(20)
for i in range(len(all_1k2)):
    k = i + 1
    all_1k2[i] = float((1/k) ** 2)

def envelope(t, t_a=0.01, tau_d=0.15):
    # returns t/t_a during the rise (t < t_a), exp decay afterwards
    return np.where(t < t_a, t/t_a, np.exp(-(t-t_a)/tau_d))

tone_pure = additive(t, f0, pure, fs) 
tone_pure_envelope = tone_pure / np.max(np.abs(tone_pure)) * envelope(t) #normalize to [-1, 1], multiply by envelope
tone_pure_click = tone_pure / np.max(np.abs(tone_pure)) * envelope(t, t_a=0) #normalize to [-1, 1], multiply by envelope with no rise

tone_odd_1k = additive(t, f0, odd_1k, fs)
tone_odd_1k_envelope = tone_odd_1k / np.max(np.abs(tone_odd_1k)) * envelope(t) #normalize to [-1, 1], multiply by envelope
tone_odd_1k_click = tone_odd_1k / np.max(np.abs(tone_odd_1k)) * envelope(t, t_a=0) #normalize to [-1, 1], envelope with no rise

tone_all_1k = additive(t, f0, all_1k, fs)
tone_all_1k_envelope = tone_all_1k / np.max(np.abs(tone_all_1k)) * envelope(t) #normalize to [-1, 1], multiply by envelope
tone_all_1k_click = tone_all_1k / np.max(np.abs(tone_all_1k)) * envelope(t, t_a=0) #normalize to [-1, 1], envelope with no rise

tone_all_1k2 = additive(t, f0, all_1k2, fs)
tone_all_1k2_envelope = tone_all_1k2 / np.max(np.abs(tone_all_1k2)) * envelope(t) #normalize to [-1, 1], multiply by envelope
tone_all_1k2_click = tone_all_1k2 / np.max(np.abs(tone_all_1k2)) * envelope(t, t_a=0) #normalize to [-1, 1], envelope with no rise

#Write tones to files to playback and compare
tones = {
    "pure_envelope":    tone_pure_envelope,
    "pure_click":       tone_pure_click,
    "odd_1k_envelope":  tone_odd_1k_envelope,
    "odd_1k_click":     tone_odd_1k_click,
    "all_1k_envelope":  tone_all_1k_envelope,
    "all_1k_click":     tone_all_1k_click,
    "all_1k2_envelope": tone_all_1k2_envelope,
    "all_1k2_click":    tone_all_1k2_click,
}

for name, tone in tones.items():
    tone_i16 = np.int16(tone * 32767) # converts to 16 bit int for wav format
    wavfile.write(f"{name}.wav", round(fs), tone_i16) # writes the wav file to disk