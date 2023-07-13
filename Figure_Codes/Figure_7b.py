import numpy
import matplotlib.pyplot as pyplot
import OBE_Tools as OBE

## Calculate the probe absorption as a function of probe detuning for different
## parameters, both with and without including dephasing.

Omegas = [0.1,0] #2pi MHz
Deltas = [0,0] #2pi MHz
Gammas = [1,0.1] #2pi MHz

Omega_23s = [0, 1, 5]
gammas = [0.2,0.2] #2pi MHz

Delta_12 = numpy.linspace(-5.8,5.8,400) #2pi MHz
probe_abs = numpy.zeros((2, len(Omega_23s), len(Delta_12)))

# for each value of Omega_23, find the probe absorption first without and then with dephasing
for j, O in enumerate(Omega_23s):
    Omegas[1] = O
    for i, p in enumerate(Delta_12):
        Deltas[0] = p
        solution = OBE.steady_state_soln(Omegas, Deltas, Gammas)
        probe_abs[0,j,i] = numpy.imag(solution[1])
        solution = OBE.steady_state_soln(Omegas, Deltas, Gammas, gammas = gammas)
        probe_abs[1,j,i] = numpy.imag(solution[1])

fig, axs = pyplot.subplots(1,2,figsize = (5,2.5), sharey = True)
for i in range(len(Omega_23s)-1):
    ax = axs[i]
    ax.plot(Delta_12, probe_abs[0,0,:], label = r'$\Omega_{12} = 0$ MHz', 
    c = 'C0', ls = 'dashed')
    ax.plot(Delta_12, probe_abs[0,i+1,:], label = r'$\Omega_{12} = 0$ MHz', 
    c = 'C1')
    ax.plot(Delta_12, probe_abs[1,i+1, :], c = 'C2', ls='dotted')
    ax.set_xlabel('Probe detuning $\Delta_{12}$ (MHz)')
    ax.set_xlim(min(Delta_12), max(Delta_12))

height = numpy.max(probe_abs[0,2,:])
scale = 1.07
axs[1].plot((-Omega_23s[2]/2, Omega_23s[2]/2),(height*scale, height*scale), 
marker = '|', c = 'black', markeredgewidth = 1.5, lw = 1.5)
axs[1].text(Omega_23s[2]/2 + 0.5, height*scale, r'$\Omega_{23}$', va = 'center')
axs[0].set_ylabel(r'Probe absorption ($-\Im[\rho_{21}]$)')

pyplot.subplots_adjust(
top=0.9,
bottom=0.2,
left=0.12,
right=0.98,
hspace=0.2,
wspace=0.05)

pyplot.show()