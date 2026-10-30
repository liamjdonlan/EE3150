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

tone_pure = additive(t, f0, pure, fs) 
tone_pure = tone_pure / np.max(np.abs(tone_pure)) #normalize to [-1, 1]

tone_odd_1k = additive(t, f0, odd_1k, fs)
tone_odd_1k = tone_odd_1k / np.max(np.abs(tone_odd_1k)) #normalize to [-1, 1]

tone_all_1k = additive(t, f0, all_1k, fs)
tone_all_1k = tone_all_1k / np.max(np.abs(tone_all_1k)) #normalize to [-1, 1]

tone_all_1k2 = additive(t, f0, all_1k2, fs)
tone_all_1k2 = tone_all_1k2 / np.max(np.abs(tone_all_1k2)) #normalize to [-1, 1]

#Write tones to files to playback and compare
pure_i16 = np.int16(tone_pure / np.max(np.abs(tone_pure)) * 32767) # converts to 16 bit int for wav format
fname = "pure.wav"
wavfile.write(fname, round(fs), pure_i16) # writes the wav file to disk

odd_1k_i16 = np.int16(tone_odd_1k / np.max(np.abs(tone_odd_1k)) * 32767) # converts to 16 bit int for wav format
fname = "odd_1k.wav"
wavfile.write(fname, round(fs), odd_1k_i16) # writes the wav file to disk

all_1k_i16 = np.int16(tone_all_1k / np.max(np.abs(tone_all_1k)) * 32767) # converts to 16 bit int for wav format
fname = "all_1k.wav"
wavfile.write(fname, round(fs), all_1k_i16) # writes the wav file to disk

all_1k2_i16 = np.int16(tone_all_1k2 / np.max(np.abs(tone_all_1k2)) * 32767) # converts to 16 bit int for wav format
fname = "all_1k2.wav"
wavfile.write(fname, round(fs), all_1k2_i16) # writes the wav file to disk

# Create k-axis to plot magnitudes against
k = np.arange(1, 21)

# plot A_k as a function of k for each tone
fig1, ax1 = plt.subplots(2, 2, sharex=True, sharey=True, figsize = (10, 7), label='A_k as a function of k for each tone')

ax1[0, 0].stem(k, pure)
ax1[0, 0].set_title('Pure Tone Coefficients')
ax1[0, 1].stem(k, odd_1k)
ax1[0, 1].set_title(r'Odd Coefficient Magnitudes = $1/k$')
ax1[1, 0].stem(k, all_1k)
ax1[1, 0].set_title(r'All Coefficient Magnitudes = $1/k$')
ax1[1, 1].stem(k, all_1k2)
ax1[1, 1].set_title(r'All Coefficient Magnitudes = $1/k^2$')
fig1.supxlabel('k')
fig1.supylabel('magnitude')
fig1.tight_layout

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(t, tone_pure, label='Pure')
ax2.plot(t, tone_odd_1k, label=r'Odd = $1/k$')
ax2.plot(t, tone_all_1k, label=r'All = $1/k$')
ax2.plot(t, tone_all_1k2, label=r'All = $1/k^2$')

ax2.set_xlim(0, 1/f0)
ax2.set_title('Tones (1 period, normalized)')
ax2.set_xlabel('t (s)')
ax2.set_ylabel('amplitude')
ax2.legend()

fig2.tight_layout()

plt.show()