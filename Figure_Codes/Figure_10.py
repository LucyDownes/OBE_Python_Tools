import numpy
import matplotlib.pyplot as pyplot
import OBE_Tools as OBE

## For a 3-level system, compare the results of using the numerical and 
## analytical solutions for different values of Omega_12

Omegas_all = numpy.asarray([[0.1,10], [5,10]]) #2pi MHz
Deltas = [0,0] #2pi MHz
Gammas = [5,1] #2pi MHz
gammas = [0.1,0.1] #2pi MHz

Delta_12s = numpy.linspace(-20,20,200) #2pi MHz
absorb = numpy.zeros((2,2, len(Delta_12s)))

# first consider weak-probe case, then repeat for outside weak-probe
for n in range(2):
    Omegas = Omegas_all[n]
    for i, p in enumerate(Delta_12s):
        Deltas[0] = p
        solution = OBE.steady_state_soln(Omegas, Deltas, Gammas, gammas = gammas)
        probe_abs = numpy.imag(solution[1])
        absorb[n,0,i] = probe_abs
        solution = OBE.fast_3_level(Omegas, Deltas, Gammas, gammas)
        probe_abs = -numpy.imag(solution)
        absorb[n,1,i] = probe_abs

fig, [ax1, ax2] = pyplot.subplots(1,2,figsize = (7,2.5))
ax1.plot(Delta_12s, absorb[0,0,:], label = 'Matrix')
ax1.plot(Delta_12s, absorb[0,1,:], label = 'Analytic', ls = 'dashed')
ax1.set_xlabel('Probe detuning $\Delta_{12}$ (MHz)')
ax1.set_ylabel(r'Probe absorption' '\n' r'($-\Im[\rho_{21}]$)')
ax1.set_xlim(numpy.min(Delta_12s), numpy.max(Delta_12s))

ax2.plot(Delta_12s, absorb[1,0,:], label = 'Matrix')
ax2.plot(Delta_12s, absorb[1,1,:], label = 'Analytic', ls = 'dashed')
ax2.set_xlabel('Probe detuning $\Delta_{12}$ (MHz)')
ax2.set_xlim(numpy.min(Delta_12s), numpy.max(Delta_12s))

pyplot.subplots_adjust(
top=0.9,
bottom=0.2,
left=0.12,
right=0.98,
hspace=0.2,
wspace=0.12)
pyplot.show()
