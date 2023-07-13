import numpy
import matplotlib.pyplot as pyplot
import OBE_Tools as OBE

## Plot the population of each state in a 2-level atomic system as a function 
## of time, both with and without considering decay.


times = numpy.linspace(0,8*numpy.pi,500)
Omegas = [1]
Deltas = [0]
Gammas = [0,0.25]

rho_0 = numpy.transpose(numpy.asarray([1,1,1,0])) #state vector at t=0
pops = numpy.zeros((2,len(times),2)) #empty array to store populations

for j,G in enumerate(Gammas):
    M = OBE.time_dep_matrix(Omegas, Deltas, [G])
    for i, t in enumerate(times):
        sol = OBE.time_evolve(M, t, rho_0) # perform the time evolution
        #extract the populations from the solution array
        rho_11, rho_22 = numpy.real(sol[0]), numpy.real(sol[3]) 
        factor = rho_11 + rho_22
        popn = [rho_11, rho_22]
        pops[j,i,:] = popn/factor # store normalised populations in array

fig, [ax1, ax2] = pyplot.subplots(1,2,figsize = (7,2.5))
ax1.plot(times, numpy.real(pops[0,:,0]))
ax1.plot(times, numpy.real(pops[0,:,1]))
ax1.set_xlabel(r'Time ($1/\Omega$)')
ax1.set_ylabel(r'Population')
ax1.set_xlim(min(times), max(times))

ax2.plot(times, numpy.real(pops[1,:,0]), label = r'$|1\rangle$')
ax2.plot(times, numpy.real(pops[1,:,1]), label = r'$|2\rangle$')
ax2.set_xlabel(r'Time ($1/\Omega$)')
ax2.set_xlim(min(times), max(times))
ax2.legend(loc=0)
pyplot.subplots_adjust(
top=0.9,
bottom=0.2,
left=0.12,
right=0.98,
hspace=0.2,
wspace=0.12)
pyplot.show()