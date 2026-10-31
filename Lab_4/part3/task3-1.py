import numpy as np
import matplotlib.pyplot as plt

A = 1.0
f0 = 220.0
fs = 44100
w0 = 2*np.pi*f0
T0 = 1/f0

t = np.arange(0, T0, 1/fs)
x = A*np.sin(w0*t)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t*1000, x, 'k--', label='Unclipped sine')
for ratio in [0.9, 0.5, 0.2]:
    L = ratio*A
    ax.plot(t*1000, np.clip(x, -L, L), label=f'L/A = {ratio}')
ax.set_title('Clipped Sine, f0 = 220 Hz')
ax.set_xlabel('t (ms)')
ax.set_ylabel('amplitude')
ax.legend()
fig.tight_layout()
plt.show()