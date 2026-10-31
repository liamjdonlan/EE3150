import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


A = 1.0
f0 = 10.0
w0 = 2*np.pi*f0
T0 = 1.0/f0

def square_wave(t, T):
    sq = np.zeros_like(t)
    for i in range(len(t)):
        rem = t[i] % T
        if rem < T/2:
            sq[i] = A
        else: 
            sq[i] = -A

    return sq


tau1, tau2, G = 10e-3, 4.7e-3, 5.7

def H(f):
    jw = 1j*2*np.pi*f
    return G / ((1 + jw*tau1) * (1 + jw*tau2))

def y_fs(t, f0, N=99):
    w0 = 2*np.pi*f0
    y = np.zeros_like(t)
    for k in range(1, 2*N, 2):
        Hk = H(k*f0)
        y = y + (4*A/(np.pi*k)) * np.abs(Hk) * np.sin(k*w0*t + np.angle(Hk))
    return y

def y_conv(f0, dt=5e-5):
    nper = 6
    t = np.arange(0, nper/f0, dt)
    x = np.sign(np.sin(2*np.pi*f0*t)); x[x == 0] = 1
    th = np.arange(0, 12*tau1, dt)
    h = G/(tau1 - tau2) * (np.exp(-th/tau1) - np.exp(-th/tau2))
    yc = np.convolve(x, h)[:len(t)] * dt
    return t, yc

# f0 = 10 Hz
tc, yc = y_conv(10.0)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(tc, yc, label='Convolution')
ax2.plot(tc, y_fs(tc, 10.0), '--', label='Fourier series')
ax2.set_title('Filtered Square Wave, f0 = 10 Hz')
ax2.set_xlabel('t (s)')
ax2.set_ylabel('amplitude')
ax2.legend()
fig2.tight_layout()

#  f0 plots
for f in [2, 10, 40]:
    tt = np.arange(0, 2/f, 1/f/10000)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tt, square_wave(tt, 1/f), label='Input square wave')
    ax.plot(tt, y_fs(tt, f), label='Filtered output')
    ax.set_title(f'f0 = {f} Hz')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('amplitude')
    ax.legend()
    fig.tight_layout()

# values
for f in [2, 10, 40]:
    print(f'f0 = {f} Hz:', [round(abs(H(k*f)), 3) for k in [1, 3, 5, 7, 9]])

plt.show()