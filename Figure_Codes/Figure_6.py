import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import OBE_Tools as OBE
import scipy.stats as stats
from scipy.optimize import curve_fit

## Plot the time taken for each solution method to complete a calculation for 
## a varying number of atomic levels.

n_levels = 13
levels = numpy.arange(n_levels)+3

#Import pre-calculated data (calculated using the Computation_Times.py file)
matrix_times = numpy.genfromtxt('Matrix_Times.txt', delimiter = '\t')
lsq_times = numpy.genfromtxt('Leastsq_Times.txt', delimiter = '\t')
iterative_times = numpy.genfromtxt('Iterative_Times.txt', delimiter = '\t')

# Find the average of 5 runs and the standard error on the mean for each method
matrix_avg = numpy.mean(matrix_times, axis = 0)
matrix_errs = stats.sem(matrix_times, axis = 0)

iterative_avg = numpy.mean(iterative_times, axis = 0)
iterative_errs = stats.sem(iterative_times, axis = 0)

lsq_avg = numpy.mean(lsq_times, axis = 0)
lsq_errs = stats.sem(lsq_times, axis = 0)

def line(x, m, c):
    return m*x + c

def power(x, a, b, c):
    return c + a*x**b

# Fit a line to the analytic times and a power law to the matrix times
power_guess = (0.05, 3.2,0)
l_opt, l_cov = curve_fit(line, levels, iterative_avg)
p_opt, p_cov = curve_fit(power, levels, matrix_avg, p0 = power_guess)
p2_opt, p2_cov = curve_fit(power, levels, lsq_avg, p0 = power_guess)

l_fit = line(levels, *l_opt)
p_fit = power(levels, *p_opt)
p2_fit = power(levels, *p2_opt)

fig, ax = pyplot.subplots(1,1,figsize = (7,3.5))
ax.errorbar(levels, iterative_avg, yerr = iterative_errs, fmt = '.', capsize = 5)
ax.errorbar(levels, matrix_avg, yerr = matrix_errs, fmt = '.', capsize = 5)
ax.errorbar(levels, lsq_avg, yerr = lsq_errs, fmt = '.', capsize = 5, 
zorder = 0, mfc = 'white')
ax.plot(levels, l_fit, c = 'C0', ls = 'dotted')
ax.plot(levels, p_fit, c = 'C1', ls = 'dashed')
#ax.plot(levels, p2_fit, c = 'C2', ls = (0,(1,10)), zorder = -1, alpha = 0.5)
ax.set_xlabel(r'Number of levels, $n$')
ax.set_ylabel('Time to solution (s)')
ax.set_xlim(2.75,15.25)
ax.set_xticks(levels)

in_ax = inset_axes(ax, width = 3.1, height = 1.2, loc=2)
in_ax.errorbar(levels, iterative_avg, yerr = iterative_errs, fmt = '.', capsize = 5)
in_ax.plot(levels, l_fit, c = 'C0', ls = 'dotted')
in_ax.set_ylim(0, 0.022)
in_ax.yaxis.tick_right()
in_ax.set_xlabel(r'Number of levels, $n$')
in_ax.set_ylabel(r'Time (s)')
in_ax.yaxis.set_label_position('right')
pyplot.show()

#print('Least squares ({} levels): {:.2f} s'.format(levels[-1], lsq_avg[-1]))
#print('Matrix ({} levels): {:.2f} s'.format(levels[-1], matrix_avg[-1]))
