import numpy as np
from scipy.io import wavfile

fs = 44100
f0 = 220.0
A = 1.0
L = 0.5*A
t = np.arange(0, 2.0, 1.0/fs)

def envelope(t, t_a=0.01, tau_d=0.15):
    #rise then decay
    return np.where(t < t_a, t/t_a, np.exp(-(t-t_a)/tau_d))

#make the sine
tone_clean = A*np.sin(2*np.pi*f0*t)

#clip the sine
tone_clipped = np.clip(tone_clean, -L, L)

#normalize and add the envelope
tone_clean_envelope = tone_clean / np.max(np.abs(tone_clean)) * envelope(t, tau_d=0.5)
tone_clipped_envelope = tone_clipped / np.max(np.abs(tone_clipped)) * envelope(t, tau_d=0.5)

#save the clean tone
clean_i16 = np.int16(tone_clean_envelope * 32767)
wavfile.write("clean.wav", round(fs), clean_i16)

#save the clipped tone
clipped_i16 = np.int16(tone_clipped_envelope * 32767)
wavfile.write("clipped.wav", round(fs), clipped_i16)