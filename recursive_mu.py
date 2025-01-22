# Import necessary modules
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Import functions from sphericaltf.py (or define them in the notebook if already present)
from sphericaltffunctions import *

def find_mu_recursive(N_mu_func, V_func, tf_radii_func, s, delta, omega, U0, dim, N_target, dN, hw):
    """
    Recursively find the chemical potential mu for a given target particle number N_target.

    Parameters:
        N_mu_func (function): Function to compute N(mu).
        V_func (function): Function to compute the potential V(r).
        tf_radii_func (function): Function to compute the TF radii.
        s (float): Length scaling factor.
        delta (float): Trap parameter.
        omega (float): Trap parameter.
        U0 (float): Interaction scaling constant.
        dim (int): Dimensionality (1, 2, or 3).
        N_target (float): Target particle number (default: 50000).
        dN (float): Tolerance for the difference between N(mu) and N_target (default: 0.01).
        hw (float): Energy scale parameter.

    Returns:
        float: The chemical potential mu that satisfies the target particle number.
    """
    # Step 1: Define mu_crit, mu_min, and mu_max
    mu_crit = V_func(0, s, delta, omega, hw)  # V(r=0)
    mu_min = tf_radii_func(mu_crit, s, delta, omega, hw)[0] or mu_crit  # Ensure valid range
    mu_max = 2 * mu_crit
    
    # Ensure mu_max is large enough to contain N_target
    while N_mu_func(mu_max, s, delta, omega, U0, dim, hw) < N_target:
        mu_max *= 2

    # Step 2: While loop to refine mu
    while abs(N_mu_func(mu_crit, s, delta, omega, U0, dim, hw) - N_target) > dN:
        N_current = N_mu_func(mu_crit, s, delta, omega, U0, dim, hw)
        print(f"N_current = {N_current}, mu = {mu_crit}")

        if N_current > N_target:
            # If N(mu) is too large, set mu_max to mu
            mu_max = mu_crit
        else:
            # If N(mu) is too small, set mu_min to mu
            mu_min = mu_crit

        # Update mu as the midpoint of the new range
        mu_crit = (mu_min + mu_max) / 2

    # Return the final mu value
    return mu_crit