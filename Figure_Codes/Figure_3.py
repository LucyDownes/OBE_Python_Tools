import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import OBE_Tools as OBE

## Evaluate the ratio of population in the ground and excited state in the 
## steady state for different ratios of Gamma and Omega. 
## Also look at the time exolution of the ground-state population.

times = numpy.linspace(0,15*2*numpy.pi,1000)
Omegas = [1]
Deltas = [0]
Gammas = numpy.linspace(0.05,2,20) # range of values of Gamma
norm_Gammas = Gammas/numpy.max(Gammas) #used tp set line opacity in plot

rho_0 = numpy.transpose(numpy.asarray([1,1,1,0])) #state vector at t=0
pops = numpy.zeros((len(Gammas),len(times),2)) #empty array to store populations

# For each value of Gamma, look at the time evolution of the ground state 
# population

for j,G in enumerate(Gammas):
    M = OBE.time_dep_matrix(Omegas, Deltas, [G])
    for i, t in enumerate(times):
        sol = OBE.time_evolve(M, t, rho_0) # perform the time evolution
        #extract the populations from the solution array
        rho_11, rho_22 = numpy.real(sol[0]), numpy.real(sol[3]) 
        factor = rho_11 + rho_22
        popn = [rho_11, rho_22]
        pops[j,i,:] = popn/factor # store normalised populations in array

# For a wider range of Gamma values, find the population at the steady state 
# (done by evaluating at a large value of t)
more_Gammas = numpy.linspace(0.05,5.1,100)
end_pops = numpy.zeros((len(more_Gammas),2), dtype = 'complex')
t = 1000

for j,G in enumerate(more_Gammas):
    M = OBE.time_dep_matrix(Omegas, Deltas, [G])
    sol = OBE.time_evolve(M, t, rho_0) # perform the time evolution
    rho_11, rho_22 = numpy.real(sol[0]), numpy.real(sol[3])
    factor = rho_11 + rho_22
    popn = [rho_11, rho_22]
    end_pops[j,:] = popn/factor # store normalised populations in array

# Calculate the ratio of the populations
ratios = numpy.real(end_pops[:,1]/end_pops[:,0])

fig, ax = pyplot.subplots(1,1,figsize = (7,3.5))

ax.plot(more_Gammas, ratios)
ax.set_xlabel(r'$\Gamma/\Omega$')
ax.set_ylabel(r'$\rho_{22}/\rho_{11}$')
ax.set_xlim(numpy.min(more_Gammas), numpy.max(more_Gammas))

# Plot time evolution in an inset
in_ax = inset_axes(ax, width = 4, height = 1.5, loc=1)
for i in range(len(Gammas)):
    in_ax.plot(times/(2*numpy.pi), numpy.real(pops[i,:,0]), 
    alpha = 1-norm_Gammas[i], c = 'C0')
in_ax.set_xlabel(r'Time ($1/\Omega$)')
in_ax.set_ylabel(r'$\rho_{11}$')
in_ax.set_xlim(min(times/(2*numpy.pi)), max(times/(2*numpy.pi)))

pyplot.subplots_adjust(
top=0.9,
bottom=0.2,
left=0.08,
right=0.98,
hspace=0.2,
wspace=0.22)
pyplot.show()