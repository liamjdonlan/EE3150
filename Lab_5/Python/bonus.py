import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve

fs = 44100
W = 2000
fc = 6000
mu = 0.7


def plot_spectrum(x, fs, fmax, title):
    N = len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(N)))
    f = np.fft.rfftfreq(N, 1.0 / fs)
    X = X / np.max(X)
    keep = f <= fmax
    plt.figure(figsize=(7, 3))
    plt.plot(f[keep], X[keep], lw=0.8)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("|X| (normalized)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title.replace(" ", "_") + ".png")
    plt.close()

def ideal_lowpass(x, fs, cutoff):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    X[f > cutoff] = 0.0
    return np.fft.irfft(X, n=len(x))

def save_audio(filename, y):
    y = y / np.max(np.abs(y))
    y_i16 = np.int16(np.clip(y, -1, 1) * 32767)
    wavfile.write(filename, fs, y_i16)

sr1, m1raw = wavfile.read("clip1.wav")
assert sr1 == fs
m1raw = m1raw.astype(float)
if m1raw.ndim > 1:
    m1raw = m1raw.mean(axis=1)
N = len(m1raw)
m1raw = m1raw[:N]
t = np.arange(N) / fs

m1 = ideal_lowpass(m1raw, fs, W)
m1 = m1 / np.max(np.abs(m1))

env = 1 + mu*m1 #envelope
s = env * np.cos(2 * np.pi * fc * t) #transmitted signal

n = int(fs * 0.012)

plt.figure(figsize=(7, 3))

i0 = np.argmax(np.abs(m1))      
seg = slice(i0, i0 + n)
plt.plot(t[seg] * 1000, env[seg], label="envelope", color="tab:blue")
plt.plot(t[seg] * 1000, -env[seg], color="tab:blue")
plt.plot(t[seg] * 1000, s[seg], label="s(t)", lw=0.6, color="tab:red")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("12 ms of env and s(t)")
plt.legend()
plt.tight_layout()

plot_spectrum(s, fs, 10000, "Transmitted Signal s(t) Spectrum")

rect = np.abs(s)  #rectify
plot_spectrum(rect, fs, 20000, "Rectified Signal Spectrum")

rect = ideal_lowpass(rect, fs, W) #apply lowpass to remove unwanted frequencies
rect = rect - 2/np.pi #remove offset
rect = rect/np.max(np.abs(rect)) #normalize

plot_spectrum(rect, fs, 20000, "Recovered Rectified Signal Spectrum")

save_audio("recovered_rectified.wav", rect)