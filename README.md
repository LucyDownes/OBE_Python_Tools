# OBE (Optical Bloch Equation) Python Tools

OBE Tools is a collection of functions written in Python that provide solution methods for the optical Bloch equations (OBEs). Alongside the functions available in the `OBE_Tools.py` module are two Jupyer notebooks containing more information on how the functions are set up (`Functions.ipynb`) and examples of how they can be used to model atom-light systems (`Examples.ipynb`). For more information please see the publication _Simple Python tools for modelling few-level atom-light interactions_.

### Prerequisites
- Numpy
- Scipy (constants, stats, linalg sub-packages)
- Sympy 
- Matplotlib (for plotting in the `Examples` notebook)

## Background

The optical Bloch equations (OBEs) arise from the Master equation which describes the time dependence of the density matrix through

$$\frac{\partial \hat{\rho}}{\partial t} = -\frac{i}{\hbar}\left[\hat{H}, \hat{\rho}\right] + \hat{\mathcal{L}}(\hat{\rho})$$

where $\hat{H}$ is the Hamiltonian of the atom-light system, $\hat{\rho}$ is the density matrix and $\hat{\mathcal{L}}$ describes the decay and dephasing in the system.

For systems with fewer than 5 levels, analytic solutions can be found, but these require using approximations and so are only valid within certain parameter ranges. The functions in the `OBE_Tools` module allows the OBEs to be set up and solved numerically for any parameter regime. It uses `Sympy` to derive the OBEs from an interaction Hamiltonian and decay/dephasing operator, then expresses these as matrix equations and uses a singular value decomposition (SVD) to solve them. It also provides functions for the analytic weak-probe solutions.

### Notes

- $\hbar = 1$
- Parameters are in units of $2\pi\,\rm{MHz}$ (the $2\pi$ is omitted from plots for clarity)
- Examples are purely illustrative and do not represent any particular physical experiment
- The functions are in no way optimised for speed, they are intended to be as transparent as possible to aid insight. 
- Variable names have been chosen to align with the notation used in the paper. Parameters (and hence variables) referring to a field coupling two levels $i$ and $j$ with have the subscript $ij$, whereas parameters that are relevant to a single atomic level $n$ will have the single subscript $n$. Variable names with upper/lower-case letters denote upper/lower-case Greek symbols respectively, for example `Gamma` denotes $\Gamma$ while `gamma` denotes $\gamma$.

The code makes a number of assumptions:
- That the number of atomic levels is equal to 1 plus the number of fields. This is consistent with a ladder system in which each level is coupled to the ones above and below it (except for the top and bottom levels).
- That each level only decays to the level below it, and there is no decay out of the bottom level. 
- That laser linewidths do not matter (unless they are explicitly specified).
