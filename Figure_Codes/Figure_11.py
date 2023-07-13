import numpy
import matplotlib.pyplot as pyplot
import OBE_Tools as OBE

## For a 4-level atomic system, evaluate probe absorption as a function of 
## probe detuning for varying values of Omega_34

Omegas = [1,10,0] #2pi MHz
Deltas = [0,0,0] #2pi MHz
gammas = [0.5,0.5,0.5] #2pi MHz
Gammas = [2,2,2] #2pi MHz
Omega_34s = [0,5,10] #2pi MHz

Delta_12s = numpy.linspace(-20,20,400) #2pi MHz
probe_abs = numpy.zeros((len(Omega_34s), len(Delta_12s)))
for j, O34 in enumerate(Omega_34s):
    Omegas[2] = O34
    for i, p in enumerate(Delta_12s):
        Deltas[0] = p
        solution = OBE.steady_state_soln(Omegas, Deltas, Gammas, gammas = gammas)
        probe_abs[j,i] = numpy.imag(solution[1])

pyplot.figure(figsize = (7,4))
pyplot.plot(Delta_12s, probe_abs[0,:], c = 'C1', label = r'$\Omega_{34} = 0$ MHz')
pyplot.plot(Delta_12s, probe_abs[1,:], c = 'C0', ls = 'dashed', label = r'$\Omega_{34} = 5$ MHz')
pyplot.plot(Delta_12s, probe_abs[2,:], c = 'C2', ls = 'dotted', label = r'$\Omega_{34} = 10$ MHz')
pyplot.xlabel('Probe detuning $\Delta_{12}$ (MHz)')
pyplot.ylabel(r'Probe absorption (-$\Im[\rho_{21}]$)')
pyplot.legend(loc = 0)
pyplot.xlim(min(Delta_12s), max(Delta_12s))

pyplot.show()