import numpy
import matplotlib
import matplotlib.pyplot as pyplot
from matplotlib.lines import Line2D
import OBE_Tools as OBE

## Calculate state populations in the steady-state as a function of probe 
## detuning for different values of probe Rabi frequency.

Omegas = [0.1,4] #2pi MHz
Deltas = [0,0] #2pi MHz
Gammas = [0.5,0.2] #2pi MHz

Omega_12s = numpy.linspace(0.1,2.5,10)
gammas = [0.1,0.1] #2pi MHz

Delta_12s = numpy.linspace(-5.8,5.8,200) #2pi MHz
pops = numpy.zeros((3, len(Omega_12s), len(Delta_12s)))

alphas = numpy.linspace(1,0.05,len(Omega_12s)) #line opacity for plot

# for each value of Omega_12 and Delta_12, find the steady-state population and store in array
for j, O in enumerate(Omega_12s):
    Omegas[0] = O
    for i, p in enumerate(Delta_12s):
        Deltas[0] = p
        solution = OBE.steady_state_soln(Omegas, Deltas, Gammas, gammas = gammas)
        for p in range(3):
            pops[p,j,i] = numpy.real(solution[4*p])

colours = ['C0', 'C2', 'C3']
lines = ['solid', 'dashed', 'dotted']
fig = pyplot.figure(figsize = (7,3))
ax1 = pyplot.subplot2grid((1,1), (0,0))

for i in range(len(Omega_12s)):
    for p in range(3):
        ax1.plot(Delta_12s, pops[p,i,:], c = colours[p], alpha = alphas[i], ls = lines[p])

ax1.set_xlabel('Probe detuning $\Delta_{12}$ (MHz)')

ax1.set_xlim(min(Delta_12s), max(Delta_12s))
ax1.set_ylabel(r'State population')

custom_lines = [Line2D([0], [0], color='C0', lw=2, ls = 'solid'),
                Line2D([0], [0], color='C2', lw=2, ls = 'dashed'),
                Line2D([0], [0], color='C3', lw=2, ls = 'dotted')]

ax1.legend(custom_lines, [r'$|1\rangle$', r'$|2\rangle$', r'$|3\rangle$'], loc = 0)

pyplot.subplots_adjust(
top=0.95,
bottom=0.16,
left=0.1,
right=0.98,
hspace=0.32,
wspace=0.05)
pyplot.show()

