import numpy as np
import matplotlib.pyplot as plt

A = 1.0

def clip_Ck(k, A, L):
    theta_c = np.arcsin(L/A) #get the angle

    if k % 2 == 0: #even k
        return 0.0

    if k == 1: #when k = 1
        return (2*A/np.pi) * (
            theta_c + np.sin(theta_c)*np.cos(theta_c)
        )

    first = np.sin((k-1)*theta_c)/(k-1) #first part
    second = np.sin((k+1)*theta_c)/(k+1) #second part

    #when k is odd and not 1
    return (2*A/np.pi)*(first - second) + \
           (4*L/(np.pi*k))*np.cos(k*theta_c)


#odd k up to 13
ks = np.arange(1, 14, 2)

#make the plot
fig, ax = plt.subplots(figsize=(10, 5))

for ratio in [0.9, 0.5, 0.2]: #each clip level
    L = ratio*A
    coefficients = []

    for k in ks: #get each coefficient
        coefficients.append(clip_Ck(k, A, L))

    #plot the sizes
    ax.plot(ks, np.abs(coefficients), 'o-', label=f'L/A = {ratio}')

#plot labels
ax.set_title('Effect of Clipping on Harmonic Amplitudes')
ax.set_xlabel('Harmonic number k')
ax.set_ylabel('|Ck|')
ax.set_xticks(ks)
ax.legend()
fig.tight_layout()


#L/A close to 1
L = 0.999*A

print('L/A = 0.999: coefficients approach an unclipped sine')

for k in ks: #print each coefficient
    print('k =', k, 'Ck =', round(clip_Ck(k, A, L), 6))


#L/A close to 0
L = 0.001*A

print('\nL/A = 0.001: normalized coefficients approach a square wave')

for k in ks:
    normalized = clip_Ck(k, A, L)/L #divide by L
    expected = 4/(np.pi*k) #square wave value

    #compare the values
    print('k =', k,
          'Ck/L =', round(normalized, 6),
          '4/(pi*k) =', round(expected, 6))

plt.show() 