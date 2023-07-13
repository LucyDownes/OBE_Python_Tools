import numpy
import matplotlib
import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import OBE_Tools as OBE

## Evaluate probe absorption as a function of probe detuning for varying values 
## of detuning of the second field in a 3-level system.

Omegas = [0.1,5] #2pi MHz
Deltas = [0,0] #2pi MHz
Gammas = [1,1] #2pi MHz

gammas = [0.1,0.1] #2pi MHz

Delta_12s = numpy.linspace(-9.8,9.8,600) #2pi MHz
Delta_23s = numpy.linspace(0,10,12)
probe_abs = numpy.zeros((len(Delta_23s), len(Delta_12s)))

alphas = 1 - (Delta_23s/(numpy.max(Delta_23s)+0.5))

for j, c in enumerate(Delta_23s):
    Deltas[1] = c
    for i, p in enumerate(Delta_12s):
        Deltas[0] = p
        solution = OBE.steady_state_soln(Omegas, Deltas, Gammas, gammas = gammas)
        probe_abs[j,i] = numpy.imag(solution[1])

fig = pyplot.figure(figsize = (7,3))
ax1 = pyplot.subplot2grid((1,1), (0,0))

for i in range(len(Delta_23s)):
    ax1.plot(Delta_12s, probe_abs[i,:], c = 'C0', alpha = alphas[i])

ax1.set_xlabel('Probe detuning $\Delta_{12}$ (MHz)')

ax1.set_xlim(min(Delta_12s), max(Delta_12s))
ax1.set_ylabel(r'Probe absorption ($-\Im[\rho_{21}]$)')

in_ax = inset_axes(ax1, width = '30%', height = "15%", loc = 'upper right')
colors = ["C0", "white"]
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap1),
             cax=in_ax, orientation='horizontal', label=r'$\Delta_{23}$ (MHz)',
              ticks = [2,4,6,8])

pyplot.subplots_adjust(
top=0.95,
bottom=0.16,
left=0.1,
right=0.98,
hspace=0.32,
wspace=0.05)
pyplot.show()