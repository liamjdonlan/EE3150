import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

A = 1.0
f0 = 10.0
w0 = 2*np.pi*f0
T0 = 1.0/f0

#N-term partial sum of x
def xtilde(t, N):
    x = np.zeros_like(t)
    for n in range(1, N+1):
        k = 2*n- 1
        x = x + 1/k * np.sin(k * w0 * t)
    return x * (4 * A)/np.pi

#ideal square wave
def square_wave(t, T):
    sq = np.zeros_like(t)
    for i in range(len(t)):
        rem = t[i] % T
        if rem < T/2:
            sq[i] = A
        else: 
            sq[i] = -A
    return sq

t = np.arange(-0.25*T0, 1.25*T0, T0/10000)

#Array of different partial sum lengths
N = [1, 3, 7, 49, 199]

fig1, ax1 = plt.subplots(figsize=(10, 5))

#Plot partial sum approximations for differnet partial sum lengths
for n in N:
    ax1.plot(t, xtilde(t, n), label=f'xtilde with N = {n}')

peak_N49 = np.max(xtilde(t, 49))
ax1.axhline(peak_N49, linestyle='--', color='gray', label=f'N=49 peak = {peak_N49:.4f}')
peak_N199 = np.max(xtilde(t, 199))
ax1.axhline(peak_N199, linestyle='--', color='blue', label=f'N=199 peak = {peak_N199:.4f}')

ax1.plot(t, square_wave(t, T0), label='Ideal square wave')

ax1.set_xlim(-0.25 * T0, 1.25 * T0)
ax1.set_title('Truncated Square Wave Plot, with Approximations')
ax1.set_xlabel('t (s)')
ax1.set_ylabel('amplitude')
ax1.legend()

fig1.tight_layout()
plt.show()
