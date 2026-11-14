import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve

# Define constants
fs = 44100
W = 2000
f1 = 6000
f2 = 12000

def plot_spectrum(x, fs, fmax, title):
    # Magnitude spectrum of x versus frequency in Hz, up to fmax.
    N = len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(N)))
    f = np.fft.rfftfreq(N, 1.0 / fs)
    X = X / np.max(X)
    keep = f <= fmax
    plt.figure(figsize=(7, 3))
    plt.plot(f[keep], X[keep], lw=0.8)
    plt.xlabel("frequency (Hz)"); plt.ylabel("|X| (normalized)")
    plt.title(title); plt.tight_layout(); plt.savefig(title.replace(" ", "_")) # Line modified to save fig instead of plotting it

def ideal_lowpass(x, fs, cutoff):
    # Ideal low-pass that removes every frequency component above 'cutoff' in Hz
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    X[f > cutoff] = 0.0
    return np.fft.irfft(X, n=len(x))

# Task 1.1
# Read .wav files
sr1, m1raw = wavfile.read("clip1.wav"); m1raw = m1raw.astype(float);
if m1raw.ndim > 1: m1raw = m1raw.mean(axis=1)
sr2, m2raw = wavfile.read("clip2.wav"); m2raw = m2raw.astype(float);
if m2raw.ndim > 1: m2raw = m2raw.mean(axis=1)
N = min(len(m1raw), len(m2raw)); m1raw, m2raw = m1raw[:N], m2raw[:N]
t = np.arange(N) / fs

# Show raw waveform graphs
plot_spectrum(m1raw, fs, 8000, "Clip 1 Raw Waveform")
plot_spectrum(m2raw, fs, 8000, "Clip 2 Raw Waveform")

# Band-limit each message to [0, W] using ideal_lowpass and normalize each by its peak magnitude
m1 = ideal_lowpass(m1raw, fs, W)
m1 = m1 / np.max(np.abs(m1))

m2 = ideal_lowpass(m2raw, fs, W)
m2 = m2 / np.max(np.abs(m2))

# Show each spectrum after band-limiting
plot_spectrum(m1, fs, 8000, "Clip 1 Band-Limited Waveform")
plot_spectrum(m2, fs, 8000, "Clip 2 Band-Limited Waveform")

# Save Band-Limited Waveforms
m1_i16 = np.int16(m1 * 32767)
wavfile.write("band-limited_clip1.wav", round(fs), m1_i16)
m2_i16 = np.int16(m2 * 32767)
wavfile.write("band-limited_clip2.wav", round(fs), m2_i16)

# Task 1.2
# Define carriers
t = np.arange(len(m1)) / fs
c1 = m1 * np.cos(2 * np.pi * f1 * t)

t = np.arange(len(m2)) / fs
c2 = m2 * np.cos(2 * np.pi * f2 * t)

# Plot x(t)
x = c1 + c2
plot_spectrum(x, fs, 20000, "Transmitted Signal x(t) Waveform")

# Task 2.1
# Define r1 and r2
t = np.arange(len(x)) / fs
r1 = x * np.cos(2 * np.pi * f1 * t)
r2 = x * np.cos(2 * np.pi * f2 * t)

# Plot r1 and r2
plot_spectrum(r1, fs, 20000, "Received Signal r1(t) Waveform")
plot_spectrum(r2, fs, 20000, "Received Signal r1(t) Waveform")

# Task 3.1
# Define m1hat and m2hat
m1hat = 2 * ideal_lowpass(r1, fs, W)
m2hat = 2 * ideal_lowpass(r2, fs, W)

# Save m1hat and m2hat
m1hat_i16 = np.int16(m1hat * 32767)
wavfile.write("reconstructed_clip1.wav", round(fs), m1hat_i16)
m2hat_i16 = np.int16(m2hat * 32767)
wavfile.write("reconstructed_clip2.wav", round(fs), m2hat_i16)

# Plot first 20ms of m1hat and m1
t = np.arange(len(m1)) / fs
ms = 20
n = int(fs * ms / 1000)
plt.figure(figsize=(7, 3))
plt.plot(t[:n] * 1000, m1[:n], label = "m1")
plt.plot(t[:n] * 1000, m1hat[:n], label = "m1hat")
plt.legend()
plt.title("First 20 ms of m1 and m1hat")
plt.xlabel("Time (ms)")
plt.ylabel("Volume")
plt.tight_layout()
plt.savefig("m1_and_m1hat_Waveforms")

# Task 3.2
# Calculate RMS for modulations of x1 and cos(wt) for different phase shifts
for phi in [0, 30, 60, 90]:
    y1 = x * np.cos(2 * np.pi * f1 * t + phi * (np.pi / 180))
    print(f"For phi = {phi} degrees, RMS of y1 = {float(np.sqrt(np.mean(y1 ** 2)))}")

# Task 4.1
# Define RC circuit
R1, C1 = 10e3, 10e-9; tau1 = R1*C1
R2, C2 = 4.7e3, 10e-9; tau2 = R2*C2
R4, R5 = 1e3, 4.7e3; G = 1

def h(t): 
    t = np.asarray(t, float)
    y = np.zeros_like(t)
    y = G/(tau1-tau2)*(np.exp(-t/tau1) - np.exp(-t/tau2))
    y[t<=0] = 0.0
    return y

# Apply RC filter to messages
t = np.arange(len(m1raw)) / fs
m1RC = fftconvolve(m1raw, h(t) / fs)[:len(m1raw)]
m1RC = m1RC / np.max(np.abs(m1RC))

t = np.arange(len(m2raw)) / fs
m2RC = fftconvolve(m2raw, h(t) / fs)[:len(m2raw)]
m2RC = m2RC / np.max(np.abs(m2RC))

# Module messages
t = np.arange(len(m1RC)) / fs
c1RC = m1RC * np.cos(2 * np.pi * f1 * t)

t = np.arange(len(m2RC)) / fs
c2RC = m2RC * np.cos(2 * np.pi * f2 * t)

# Combine modulated signals
xRC = c1RC + c2RC

# Define received messages 
r1RC = xRC * np.cos(2 * np.pi * f1 * t)
r2RC = xRC * np.cos(2 * np.pi * f2 * t)

# Demodulate messages
t = np.arange(len(r1RC)) / fs
m1RChat = 2 * fftconvolve(r1RC, h(t) / fs)[:len(r1RC)]

t = np.arange(len(r2RC)) / fs
m2RChat = 2 * fftconvolve(r2RC, h(t) / fs)[:len(r2RC)]

# Save demodulated messages
m1RChat_i16 = np.int16(m1RChat * 32767)
wavfile.write("RC_reconstructed_clip1.wav", round(fs), m1RChat_i16)
m2RChat_i16 = np.int16(m2RChat * 32767)
wavfile.write("RC_reconstructed_clip2.wav", round(fs), m2RChat_i16)

# Plot demodulated messages
plot_spectrum(m1hat, fs, 8000, "Clip 1 Reconstructed Waveform")
plot_spectrum(m2hat, fs, 8000, "Clip 2 Reconstructed Waveform")
plot_spectrum(m1RChat, fs, 8000, "Clip 1 RC Reconstructed Waveform")
plot_spectrum(m2RChat, fs, 8000, "Clip 2 RC Reconstructed Waveform")
