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

r1 = x * np.cos(2 * np.pi * f1 * t)
plot_spectrum(r1, fs, 20000, "Demodulated Channel 1 Spectrum")

r2 = x * np.cos(2 * np.pi * f2 * t)
m1hat = 2 * ideal_lowpass(r1, fs, W)
m2hat = 2 * ideal_lowpass(r2, fs, W)
save_audio("reconstructed_clip1.wav", m1hat)
save_audio("reconstructed_clip2.wav", m2hat)

n = int(fs * 0.020)
plt.figure(figsize=(7, 3))
plt.plot(t[:n] * 1000, m1[:n], label="m1")
plt.plot(t[:n] * 1000, m1hat[:n], label="m1hat")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("First 20 ms of m1 and m1hat")
plt.legend()
plt.tight_layout()
plt.savefig("m1_and_m1hat_Waveforms.png")
plt.close()

rms_m1 = float(np.sqrt(np.mean(m1 ** 2)))
phase_rows = []
for phi in [0, 30, 60, 90]:
    phi_rad = np.deg2rad(phi)
    mixed = x * np.cos(2 * np.pi * f1 * t + phi_rad)
    y1hat = 2 * ideal_lowpass(mixed, fs, W)
    rms_y1 = float(np.sqrt(np.mean(y1hat ** 2)))
    predicted = rms_m1 * abs(np.cos(phi_rad))
    phase_rows.append((phi, np.cos(phi_rad), rms_y1, predicted))
    print(phi, rms_y1, predicted)

R1, C1 = 10e3, 10e-9
R2, C2 = 4.7e3, 10e-9
tau1 = R1 * C1
tau2 = R2 * C2
G = 1

f = np.linspace(0, 8000, 4001)
Hrc = G / ((1 + 1j * 2 * np.pi * f * tau1)
           * (1 + 1j * 2 * np.pi * f * tau2))
Hideal = (f <= W).astype(float)
plt.figure(figsize=(7, 3))
plt.plot(f, np.abs(Hrc), label="Two-stage RC")
plt.plot(f, Hideal, label="Ideal low-pass")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("RC and Ideal Frequency Response")
plt.legend()
plt.tight_layout()
plt.savefig("RC_and_Ideal_Frequency_Response.png")
plt.close()

def h(t):
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t)
    keep = t >= 0
    y[keep] = G / (tau1 - tau2) * (
        np.exp(-t[keep] / tau1) - np.exp(-t[keep] / tau2))
    return y

sr1, raw1 = wavfile.read("clip1.wav")
sr2, raw2 = wavfile.read("clip2.wav")
assert sr1 == sr2 == fs
raw1, raw2 = raw1.astype(float), raw2.astype(float)
if raw1.ndim > 1:
    raw1 = raw1.mean(axis=1)
if raw2.ndim > 1:
    raw2 = raw2.mean(axis=1)
N = min(len(raw1), len(raw2))
raw1, raw2 = raw1[:N], raw2[:N]
t = np.arange(N) / fs
kernel = h(t) / fs

m1RC = fftconvolve(raw1, kernel)[:N]
m2RC = fftconvolve(raw2, kernel)[:N]
m1RC = m1RC / np.max(np.abs(m1RC))
m2RC = m2RC / np.max(np.abs(m2RC))

xRC = (m1RC * np.cos(2 * np.pi * f1 * t)
       + m2RC * np.cos(2 * np.pi * f2 * t))
r1RC = xRC * np.cos(2 * np.pi * f1 * t)
r2RC = xRC * np.cos(2 * np.pi * f2 * t)
m1RChat = 2 * fftconvolve(r1RC, kernel)[:N]
m2RChat = 2 * fftconvolve(r2RC, kernel)[:N]

save_audio("RC_reconstructed_clip1.wav", m1RChat)
save_audio("RC_reconstructed_clip2.wav", m2RChat)

plot_spectrum(m1hat, fs, 8000, "Clip 1 Reconstructed Waveform")
plot_spectrum(m2hat, fs, 8000, "Clip 2 Reconstructed Waveform")
plot_spectrum(m1RChat, fs, 8000, "Clip 1 RC Reconstructed Waveform")
plot_spectrum(m2RChat, fs, 8000, "Clip 2 RC Reconstructed Waveform")
