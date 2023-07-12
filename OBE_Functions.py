'''Series of functions to be used to solve the Optical Bloch equations for atomic systems with any number of levels.
For examples and more details see the accompanying Jupyter Notebooks 'General Symbolic OBE Solver' and '1D Doppler Averaging'. 

Author: Lucy Downes (lucy.downes@durham.ac.uk)
Date: 1/3/23'''

import numpy
import sympy
import scipy.constants as const
import scipy.stats as stats
import scipy.linalg as linalg
import scipy.integrate as integ

def create_rho_list(levels = 3):
    """
    Create a list of Sympy symbolic objects to represent the elements of the
    density matrix (rho_ij) for a given number of levels

    Args:
        levels (int, optional): Number of levels in the atomic scheme. 
            Defaults to 3.

    Returns:
        List: list containing Sympy symbolic objects.
            Length n^2 for n atomic levels.
    """    
    rho_list = []
    for i in range(levels):
        for j in range(levels):
            globals()['rho_'+str(i+1)+str(j+1)] = sympy.Symbol(
                'rho_'+str(i+1)+str(j+1))
            rho_list.append(globals()['rho_'+str(i+1)+str(j+1)])
    return rho_list

def create_rho_matrix(levels = 3):
    """Create density matrix where elements are Sympy symbolic objects.

    Args:
        levels (int, optional): Number of levels in the atomic scheme. 
            Defaults to 3.

    Returns:
        numpy.matrix: matrix of Sympy symbolic objects. 
            Matrix size will be nxn for an n-level system. 
    """    
    rho_matrix = numpy.empty((levels, levels), dtype = 'object')
    for i in range(levels):
        for j in range(levels):
            globals()['rho_'+str(i+1)+str(j+1)] = sympy.Symbol(
                'rho_'+str(i+1)+str(j+1))
            rho_matrix[i,j] = globals()['rho_'+str(i+1)+str(j+1)]
    return numpy.matrix(rho_matrix)

def Hamiltonian(Omegas, Deltas):
    """
    Given lists of Rabi frequencies and detunings, construct interaction 
    Hamiltonian (assuming RWA and dipole approximation) as per equation 14. 
    h_bar = 1 for simplicity.
    Both lists should be in ascending order (Omega_12, Omega_23 etc)

    Args:
        Omegas (list of floats): List of Rabi frequencies for all fields 
            (n-1 fields for n atomic levels)
        Deltas (list of floats): List of detunings for all fields 
            (n-1 fields for n atomic levels)

    Returns:
        numpy.matrix: the total interaction Hamiltonian in matrix form. 
            Will be of shape nxn for an n-level system.
    """    
    levels = len(Omegas)+1
    H = numpy.zeros((levels,levels))
    for i in range(levels):
        for j in range(levels):
            if numpy.logical_and(i==j, i!=0):
                H[i,j] = -2*(numpy.sum(Deltas[:i]))
            elif numpy.abs(i-j) == 1:
                H[i,j] = Omegas[numpy.min([i,j])]
    return numpy.matrix(H/2)

def L_decay(Gammas):
    """
    Given a list of linewidths for each atomic level, 
    construct atomic decay operator in matrix form for a ladder system.
    Assumes that there is no decay from the lowest level (\Gamma_1 = 0).

    Args:
        Gammas (list of floats): Linewidth of the atomic states considered. 
            Assumes no decay from lowest energy atomic state (\Gamma_1 = 0).
            Values should be given in order of lowest to highest energy level 
            (\Gamma_2, \Gamma_3, ... , \Gamma_n for n levels). 
            n-1 values for an n-level system.

    Returns:
        numpy.matrix: Decay operator as a matrix containing multiples of Sympy 
            symbolic objects. Will be of size nxn for an n-level system.
    """    
    levels = len(Gammas)+1
    rhos = create_rho_matrix(levels = levels)
    Gammas_all = [0] + Gammas
    decay_matrix = numpy.zeros((levels, levels), dtype = 'object')
    for i in range(levels):
        for j in range(levels):
            if i != j:
                decay_matrix[i,j] = -0.5*(
                    Gammas_all[i]+Gammas_all[j])*rhos[i,j]
            elif i != levels - 1:
                into = Gammas_all[i+1]*rhos[1+i, j+1]
                outof = Gammas_all[i]*rhos[i, j]
                decay_matrix[i,j] = into - outof
            else:
                outof = Gammas_all[i]*rhos[i, j]
                decay_matrix[i,j] = - outof
    return numpy.matrix(decay_matrix)

def L_dephasing(gammas):
    """
    Given list of laser linewidths, create dephasing operator in matrix form. 

    Args:
        gammas (list): Linewidths of the field coupling each pair of states 
            in the ladder, from lowest to highest energy states 
            (\gamma_{12}, ..., \gamma{n-1,n} for n levels).

    Returns:
        numpy.matrix: Dephasing operator as a matrix populated by expressions 
            containing Sympy symbolic objects.
            Will be size nxn for an n-level system.
    """    
    levels = len(gammas)+1
    rhos = create_rho_matrix(levels = levels)
    deph_matrix = numpy.zeros((levels, levels), dtype = 'object')
    for i in range(levels):
        for j in range(levels):
            if i != j:
                deph_matrix[i,j] = -(numpy.sum(gammas[numpy.min(
                    [i,j]):numpy.max([i,j])]))*rhos[i,j]
    return numpy.matrix(deph_matrix)

def Master_eqn(H_tot, L):
    """
    Return an expression for the right hand side of the Master equation 
    (as in Eqn 18) for a given Hamiltonian and Lindblad term. 
    Assumes that the number of atomic levels is set by the shape of the 
    Hamiltonian matrix (nxn for n levels).

    Args:
        H_tot (matrix): The total Hamiltonian in the interaction picture.
        L (matrix): The Lindblad superoperator in matrix form. 

    Returns:
        numpy.matrix: Right hand side of the Master equation (eqn 18) 
        as a matrix containing expressions in terms of Sympy symbolic objects.
        Will be size nxn for an n-level system.
    """    
    levels = H_tot.shape[0]
    dens_mat = create_rho_matrix(levels = levels)
    return -1j*(H_tot*dens_mat - dens_mat*H_tot) + L

def OBE_matrix(Master_matrix):
    """
    Take the right hand side of the Master equation (-i[H,\rho] + L) 
    expressed as an array of multiples of Sympy symbolic objects and output 
    an ndarray of coefficients M such that d rho_vect/dt = M*rho_vect 
    where rho_vect is the vector of density matrix elements.

    Args:
        Master_matrix (matrix): The right hand side of the Master equation 
            as a matrix of multiples of Sympy symbolic objects

    Returns:
        numpy.ndarray: An array of (complex) coefficients. 
            Will be size n^2 x n^2 for an n-level system.
    """    
    levels = Master_matrix.shape[0]
    rho_vector = create_rho_list(levels = levels)
    coeff_matrix = numpy.zeros((levels**2, levels**2), dtype = 'complex')
    count = 0
    for i in range(levels):
        for j in range(levels):
            entry = Master_matrix[i,j]
            expanded = sympy.expand(entry)
            #use Sympy coeff to extract coefficient of each element in rho_vect
            for n,r in enumerate(rho_vector):
                coeff_matrix[count, n] = complex(expanded.coeff(r)) 
            count += 1
    return coeff_matrix

def SVD(coeff_matrix):
    """
    Perform singular value decomposition (SVD) on matrix of coefficients 
    using the numpy.linalg.svd function.
    SVD returns US(V)*^T where S is the diagonal matrix of singular values.
    The solution of the system of equations is then the column of V 
    corresponding to the zero singular value.
    If there is no zero singular value (within tolerance, allowing for 
    floating point precision) then return a zero array.

    Args:
        coeff_matrix (ndarray): array of complex coefficients M that satisfies 
            the expression d rho_vect/dt = M*rho_vect where rho_vect is the 
            vector of density matrix elements.

    Returns:
        ndarray: 1D array of complex floats corresponding to the steady-state 
            value of each element of the density matrix in the order given by 
            the rho_list function. Will be length n for an n-level system.
    """    
    levels = int(numpy.sqrt(coeff_matrix.shape[0]))
    u,sig,v = numpy.linalg.svd(coeff_matrix)
    abs_sig = numpy.abs(sig)
    minval = numpy.min(abs_sig)
    if minval>1e-12:
        print('ERROR - Matrix is non-singular')
        return numpy.zeros((levels**2))
    index = abs_sig.argmin()
    rho = numpy.conjugate(v[index,:]) 
    #SVD returns the conjugate transpose of v
    pops = numpy.zeros((levels))
    for l in range(levels):
        pops[l] = (numpy.real(rho[(l*levels)+l]))
    t = 1/(numpy.sum(pops)) #Normalise so the sum of the populations is one
    rho_norm = rho*t.item()
    return rho_norm



def steady_state_soln(Omegas, Deltas, Gammas, gammas = []):
    """
    Given lists of parameters (all in order of lowers to highest energy level), 
    construct and solve for steady state of density matrix. 
    Returns ALL elements of density matrix in order of rho_list 
    (rho_11, rho_12, ... rho_i1, rho_i2, ..., \rho_{n, n-1}, rho_nn).

    Args:
        Omegas (list of floats): Rabi frequencies of fields coupling each pair 
            of states in the ladder, from lowest to highest energy states 
            (\Omega_{12}, ..., \Omega{n-1,n} for n levels).
        Deltas (list of floats): Detuning of fields coupling each pair 
            of states in the ladder, from lowest to highest energy states 
            (\Delta_{12}, ..., \Delta{n-1,n} for n levels).
        Gammas (list of floats): Linewidth of the atomic states considered. 
            Assumes no decay from lowest energy atomic state (\Gamma_1 = 0).
            Values should be given in order of lowest to highest energy level 
            (\Gamma_2, \Gamma_3, ... , \Gamma_n for n levels). 
            n-1 values for n levels.
        gammas (list of floats, optional): Linewidths of the fields coupling 
            each pair of states in the ladder, from lowest to highest energy 
            (\gamma_{12}, ..., \gamma{n-1,n} for n levels). 
            Defaults to [] (meaning all \gamma_ij = 0).

    Returns:
        ndarray: 1D array containing values for each element of the density 
            matrix in the steady state, in the order returned by the rho_list 
            function (\rho_11, \rho_12, ..., \rho_{n, n-1}, \rho_nn).
            Will be length n for an n-level system.
    """    
    L_atom = L_decay(Gammas)
    if len(gammas) != 0: 
        L_laser = L_dephasing(gammas)
        L_tot = L_atom + L_laser
    else:
        L_tot = L_atom
    H = Hamiltonian(Omegas, Deltas)
    Master = Master_eqn(H, L_tot)
    rho_coeffs = OBE_matrix(Master)
    soln = SVD(rho_coeffs)
    return soln

def fast_3_level(Omegas, Deltas, Gammas, gammas = []):
    """
    Calculate the analytic solution of the steady-state probe coherence 
    (\rho_{21}) in the weak probe limit for a 3-level ladder system.

    Args:
        Omegas (list of floats): Rabi frequencies of fields coupling each pair 
            of states in the ladder, from lowest to highest energy states 
            (\Omega_{12}, ..., \Omega{n-1,n} for n levels).
        Deltas (list of floats): Detuning of fields coupling each pair 
            of states in the ladder, from lowest to highest energy states 
            (\Delta_{12}, ..., \Delta{n-1,n} for n levels).
        Gammas (list of floats): Linewidth of the atomic states considered. 
            Assumes no decay from lowest energy atomic state (\Gamma_1 = 0).
            Values should be given in order of lowest to highest energy level 
            (\Gamma_2, \Gamma_3, ... , \Gamma_n for n levels). 
            n-1 values for n levels.
        gammas (list of floats, optional): Linewidths of the field coupling 
            each pair of states in the ladder, from lowest to highest energy 
            (\gamma_{12}, ..., \gamma{n-1,n} for n levels). 
            Defaults to [] (meaning all \gamma_ij = 0).

    Returns:
        complex: steady-state value of the probe coherence (\rho_{21})
    """    
    Delta_12, Delta_23 = Deltas[:]
    Omega_12, Omega_23 = Omegas[:]
    Gamma_2, Gamma_3 = Gammas[:]
    if len(gammas) != 0:
        gamma_12, gamma_23 = gammas[:]
    else:
        gamma_12, gamma_23 = 0, 0
    expression = (Omega_23**2/4)/(1j*(Delta_12 + Delta_23) + (Gamma_3/2) \
    + gamma_12 + gamma_23)
    bottom = (1j*Delta_12) + (Gamma_2/2) + gamma_12 + expression
    rho = (1j*Omega_12)/(2*bottom)
    return numpy.conjugate(rho)

def fast_4_level(Omegas, Deltas, Gammas, gammas = []):
    """
    Analytic solution of the steady-state probe coherence (\rho_{21}) 
    in the weak probe limit for a 4-level ladder system.

    Args:
        Omegas (list of floats): Rabi frequencies of fields coupling each pair 
            of states in the ladder, from lowest to highest energy states 
            (\Omega_{12}, ..., \Omega{n-1,n} for n levels).
        Deltas (list of floats): Detuning of fields coupling each pair 
            of states in the ladder, from lowest to highest energy states 
            (\Delta_{12}, ..., \Delta{n-1,n} for n levels).
        Gammas (list of floats): Linewidth of the atomic states considered. 
            Assumes no decay from lowest energy atomic state (\Gamma_1 = 0).
            Values should be given in order of lowest to highest energy level 
            (\Gamma_2, \Gamma_3, ... , \Gamma_n for n levels). 
            n-1 values for n levels.
        gammas (list of floats, optional): Linewidths of the field coupling 
            each pair of states in the ladder, from lowest to highest energy  
            (\gamma_{12}, ..., \gamma{n-1,n} for n levels). 
            Defaults to [] (meaning all \gamma_ij = 0).

    Returns:
        complex: steady-state value of the probe coherence (\rho_{21})
    """    
    Omega_12, Omega_23, Omega_34 = Omegas[:]
    Delta_12, Delta_23, Delta_34 = Deltas[:]
    Gamma_2, Gamma_3, Gamma_4 = Gammas[:]
    if len(gammas) != 0:
        gamma_12, gamma_23, gamma_34 = gammas[:]
    else:
        gamma_12, gamma_23, gamma_34 = 0,0,0
    bracket_1 = 1j*(Delta_12 + Delta_23 + Delta_34) - gamma_12 - gamma_23 - \
    gamma_34 - (Gamma_4/2)
    bracket_2 = 1j*(Delta_12 + Delta_23) - (Gamma_3/2) - gamma_12 - gamma_23 +\
    (Omega_34**2)/(4*bracket_1)
    bracket_3 = 1j*Delta_12 - (Gamma_2/2) - gamma_12 + (
        Omega_23**2)/(4*bracket_2)
    return (1j*Omega_12)/(2*bracket_3)

def term_n(n, Deltas, Gammas, gammas = []):
    """
    Generate the nth term in the iterative expansion method for calculating 
    the probe coherence (\rho_{21}) for an arbitrary number of levels in the 
    weak-probe limit.

    Args:
        n (int): Index (for n>0)
        Deltas (list of floats): Detuning of fields coupling each pair 
            of states in the ladder, from lowest to highest energy states 
            (\Delta_{12}, ..., \Delta{n-1,n} for n levels).
        Gammas (list of floats): Linewidth of the atomic states considered. 
            Assumes no decay from lowest energy atomic state (\Gamma_1 = 0).
            Values should be given in order of lowest to highest energy level 
            (\Gamma_2, \Gamma_3, ... , \Gamma_n for n levels). 
            n-1 values for n levels.
        gammas (list of floats, optional): Linewidths of the fields coupling 
            each pair of states in the ladder, from lowest to highest energy 
            (\gamma_{12}, ..., \gamma{n-1,n} for n levels).
            Defaults to [] (meaning all \gamma_ij = 0).

    Returns:
        complex float: value of the probe coherence (\rho_{21}) 
            in the steady-state
    """  
    if len(gammas) == 0:
        gammas = numpy.zeros((len(Deltas)))  
    # n>0
    return 1j*(numpy.sum(Deltas[:n+1])) - (Gammas[n]/2) - numpy.sum(
        gammas[:n+1])

def fast_n_level(Omegas, Deltas, Gammas, gammas = []):
    """
    Return the steady-state probe coherence (\rho_{21}) in the weak-probe limit
    for a ladder system with an arbitrary number of levels. 
    Calls the term_n function.

    Args:
        Omegas (list of floats): Rabi frequencies of fields coupling each pair 
            of states in the ladder, from lowest to highest energy states 
            (\Omega_{12}, ..., \Omega{n-1,n} for n levels).
        Deltas (list of floats): Detuning of fields coupling each pair 
            of states in the ladder, from lowest to highest energy states 
            (\Delta_{12}, ..., \Delta{n-1,n} for n levels).
        Gammas (list of floats): Linewidth of the atomic states considered. 
            Assumes no decay from lowest energy atomic state (\Gamma_1 = 0).
            Values should be given in order of lowest to highest energy level 
            (\Gamma_2, \Gamma_3, ... , \Gamma_n for n levels). 
            n-1 values for n levels.
        gammas (list of floats, optional): Linewidths of the fields coupling 
            each pair of states in the ladder, from lowest to highest energy 
            (\gamma_{12}, ..., \gamma{n-1,n} for n levels).
            Defaults to [] (meaning all \gamma_ij = 0).            

    Returns:
        complex float: value of the probe coherence (\rho_{21}) 
            in the steady-state
    """  
    if len(gammas) == 0:
        gammas = numpy.zeros((len(Omegas)))  
    n_terms = len(Omegas)
    term_counter = numpy.arange(n_terms)[::-1]
    func = 0
    for n in term_counter:
        prefact = (Omegas[n]/2)**2
        term = term_n(n, Deltas, Gammas, gammas)
        func = prefact/(term+func)
    return (2j/Omegas[0])*func

def MBdist(velocities, temp = 20, mass = 132.90545):
    """
    Method for calculating the relative abundances of given velocity classes 
    assuming a Maxwell-Boltzmann distribution of velocities for a given 
    temperature and atomic mass using built-in Scipy methods.
    Defaults to Caesium (the best atom) at room temperature (20 degrees C). 

    Args:
        velocities (1D array): atomic velocities in units of m/s
        temp (int, optional): Temperature of the atomic ensemble 
            in degrees Celsius. Defaults to 20 deg C (room temp).
        mass (float, optional): Mass number of the atomic species used 
            in atomic mass units. Defaults to 132.90545 amu (Caesium).

    Returns:
        numpy.ndarray: relative abundances of the specified velocities 
            according to the Maxwell-Boltzmann distribution. 
            Will be the same shape as velocities.
    """    
    m = const.m_u*mass #kg
    kb = const.k #m**2 kg s**-2 K**-1
    temp_k = temp + 273.15 #K
    sigma = numpy.sqrt((kb*temp_k)/m)
    return stats.norm(0, sigma).pdf(velocities)

def MBdist2(velocities, temp = 20, mass = 132.90545):
    """
    Method for calculating the relative abundances of given velocity classes 
    assuming a Maxwell-Boltzmann distribution of velocities for a given 
    temperature and atomic mass explicitly (without using Scipy).
    Defaults to Caesium (the best atom) at room temperature. 

    Args:
        velocities (1D array): atomic velocities in units of m/s
        temp (int, optional): Temperature of the atomic ensemble 
            in degrees Celsius. Defaults to 20 deg C (room temp).
        mass (float, optional): Mass number of the atomic species used 
            in atomic mass units. Defaults to 132.90545 amu (Caesium).

    Returns:
        numpy.ndarray: relative abundances of the specified velocities 
            according to the Maxwell-Boltzmann distribution. 
            will be the same shape as velocities.
    """
    kb = 1.38e-23 #m**2 kg s**-2 K**-1
    m = mass*1.66e-27 #kg
    T = temp+273.15 #Kelvin
    f = numpy.sqrt(m/(2*numpy.pi*kb*T))*numpy.exp((-m*velocities**2)/(2*kb*T))
    return f

def time_op(operator, t):
    """Creates expresion for the time evolution operator. 

    Args:
        operator (matrix): Operator describing the time evolution of a system 
            in matrix form
        t (float): Time at which the expression is to be evaluated

    Returns:
        numpy.matrix: matrix form of the time evolution operator 
            exp^{operator*t}
    """    
    exponent = operator*t
    return linalg.expm(exponent) #linalg.expm does matrix exponentiation

def time_evolve(operator, t, psi_0):
    """
    Evaluate the state of a system at a time t, given an operator 
    describing its time evolution and the state of the system at t=0.

    Args:
        operator (matrix): matrix representation of operator describing 
            time evolution of the system.
        t (float): Time at which the state of the system is to be evaluated.
        psi_0 (1D array): Vector describing the initial state of the system 
            (at t=0).

    Returns:
        1D array: the state of the system at time t
    """    
    return numpy.matmul(time_op(operator, t), psi_0)

def time_dep_matrix(Omegas, Deltas, Gammas, gammas = []):
    """
    Given lists of parameters (all in order of lowers to highest energy level), 
    construct matrix of coefficients for time evolution of 
    the density matrix vector.

    Args:
        Omegas (list of floats): Rabi frequencies of fields coupling each pair 
            of states in the ladder, from lowest to highest energy states 
            (\Omega_{12}, ..., \Omega{n-1,n} for n levels).
        Deltas (list of floats): Detuning of fields coupling each pair 
            of states in the ladder, from lowest to highest energy states 
            (\Delta_{12}, ..., \Delta{n-1,n} for n levels).
        Gammas (list of floats): Linewidth of the atomic states considered. 
            Assumes no decay from lowest energy atomic state (\Gamma_1 = 0).
            Values should be given in order of lowest to highest energy level 
            (\Gamma_2, \Gamma_3, ... , \Gamma_n for n levels). 
            n-1 values for n levels.
        gammas (list of floats, optional): Linewidths of the fields coupling 
            each pair of states in the ladder, from lowest to highest energy 
            (\gamma_{12}, ..., \gamma{n-1,n} for n levels).
            Defaults to [] (meaning all \gamma_ij = 0).  
    Returns:
        ndarray: n^2 x n^2 array (for an n-level system) of coefficients M 
            which satisfies the equation d\rho_{vect}/dt = M\rho_{vect} 
            where \rho_{vect} is the vector representation of the 
            density matrix.
    """    
    # first create decay/dephasing operators
    L_atom = L_decay(Gammas)
    if len(gammas) != 0: 
        L_laser = L_dephasing(gammas)
        L_tot = L_atom + L_laser
    else:
        L_tot = L_atom
    H = Hamiltonian(Omegas, Deltas) #create the total Hamiltonian
    Master = Master_eqn(H, L_tot) 
    rho_coeffs = OBE_matrix(Master) #create matrix of coefficients
    return rho_coeffs