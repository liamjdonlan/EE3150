import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve

fs = 44100
W = 2000
f1 = 6000
f2 = 12000

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
sr2, m2raw = wavfile.read("clip2.wav")
assert sr1 == sr2 == fs
m1raw = m1raw.astype(float)
m2raw = m2raw.astype(float)
if m1raw.ndim > 1:
    m1raw = m1raw.mean(axis=1)
if m2raw.ndim > 1:
    m2raw = m2raw.mean(axis=1)
N = min(len(m1raw), len(m2raw))
m1raw, m2raw = m1raw[:N], m2raw[:N]
t = np.arange(N) / fs

plot_spectrum(m1raw, fs, 8000, "Clip 1 Raw Waveform")
plot_spectrum(m2raw, fs, 8000, "Clip 2 Raw Waveform")

m1 = ideal_lowpass(m1raw, fs, W)
m1 = m1 / np.max(np.abs(m1))
m2 = ideal_lowpass(m2raw, fs, W)
m2 = m2 / np.max(np.abs(m2))

plot_spectrum(m1, fs, 8000, "Clip 1 Band-Limited Waveform")
plot_spectrum(m2, fs, 8000, "Clip 2 Band-Limited Waveform")
save_audio("band-limited_clip1.wav", m1)
save_audio("band-limited_clip2.wav", m2)

c1 = np.cos(2 * np.pi * f1 * t)
c2 = np.cos(2 * np.pi * f2 * t)
x = m1 * c1 + m2 * c2
plot_spectrum(x, fs, 20000, "Transmitted Signal x(t) Waveform")