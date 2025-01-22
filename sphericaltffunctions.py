import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from recursive_mu import *

# Constants
U0 = 1.0  # Scaling constant
hw = 1.0  # Constant involving hbar and frequency
s = 1.0  # Length scaling factor
delta = 0.5  # Trap parameter
omega = 0.3  # Trap parameter
dim = 2  # Number of dimensions in the BEC, for metric factor in integration

# Metric factor dependent on dimension
def metric_factor(r, dim):
    if dim == 1:
        return 2
    elif dim == 2:
        return 2 * np.pi * r
    elif dim == 3:
        return 4 * np.pi * r**2
    else:
        return 0.0
    

# Potential function V(r)
def V(r, s, delta, omega, hw):
    arg = ((r / s)**2 - delta)**2 + (2 * omega)**2
    return hw * np.sqrt(arg)

# Compute the Thomas-Fermi radii
def tf_radii(mu, s, delta, omega, hw):
    nu = mu / hw
    if nu**2 <= (2 * omega)**2:
        return None, None  # No valid radii
    inner_root = np.sqrt(nu**2 - (2 * omega)**2)
    r_TF_minus = s * np.sqrt(delta - inner_root) if delta - inner_root >= 0 else 0
    r_TF_plus = s * np.sqrt(delta + inner_root) if delta + inner_root >= 0 else 0
    return r_TF_minus, r_TF_plus


# Radial density integrand
def n_r(r, mu, s, delta, omega, U0, dim, hw):
    V_r = V(r, s, delta, omega, hw)
    if mu > V_r:
        return (mu - V_r) / U0 * metric_factor(r, dim)
    else:
        return 0.0
    
    
# Integral for total number of particles N(mu)
def N_mu(mu, s, delta, omega, U0, dim, hw):
    r_min, r_max = tf_radii(mu, s, delta, omega, hw)
    if r_min is None or r_max is None:
        return 0.0  # No valid BEC region

    # Perform the integration over the valid range
    integral, _ = quad(n_r, r_min, r_max, args=(mu, s, delta, omega, U0, dim, hw))
    return integral


# Function to compute the minimum value of V(r)
def V_min(s, delta, omega, r_max, num_points, hw):
    """
    Compute the minimum value of the potential V(r) over a specified range.

    Parameters:
        s (float): Length scaling factor.
        delta (float): Trap parameter.
        omega (float): Trap parameter.
        r_max (float): Maximum r value to evaluate.
        num_points (int): Number of points to sample.

    Returns:
        float: Minimum value of V(r).
    """
    r_values = np.linspace(0, r_max, num_points)  # Range of r values
    V_values = [V(r, s, delta, omega, hw) for r in r_values]  # Compute V(r)
    return min(V_values)


# Function to generate mu values and compare to V_min
def compare_mu_to_V_min(mu_values, V_min_value):
    """
    Compare a list of mu values to a given V_min and print a warning if mu < V_min.

    Parameters:
        mu_values (list): List of mu values to compare.
        V_min_value (float): Precomputed minimum value of V(r).

    Returns:
        None
    """
    print(f"Minimum value of V(r): {V_min_value:.2f}")

    # Compare each mu to V_min
    for mu in mu_values:
        if mu < V_min_value:
            print(f"Warning: mu = {mu:.2f} is less than V_min = {V_min_value:.2f}!")

            
# Example: Function to search text file and interpolate mu for a given N
def interpolate_from_file(filename, N_target):
    """
    Interpolate mu for a given N_target by searching the text file.

    Parameters:
        filename (str): Path to the text file containing N(mu) and mu data.
        N_target (float): The target N value.

    Returns:
        float: Interpolated mu value.
    """
    # Load data from the file
    data = np.loadtxt(filename, skiprows=1)
    N_values, mu_values = data[:, 0], data[:, 1]

    # Handle edge cases: Clamp N_target to the range of N values
    if N_target <= N_values[0]:
        return mu_values[0]  # Return the smallest mu
    elif N_target >= N_values[-1]:
        return mu_values[-1]  # Return the largest mu

    # Find the two closest N values
    idx = np.searchsorted(N_values, N_target)
    idx_low = max(0, idx - 1)
    idx_high = min(len(N_values) - 1, idx)

    # Linear interpolation
    N_low, N_high = N_values[idx_low], N_values[idx_high]
    mu_low, mu_high = mu_values[idx_low], mu_values[idx_high]
    if N_high == N_low:  # Prevent division by zero
        return mu_low
    mu_interpolated = mu_low + (mu_high - mu_low) * (N_target - N_low) / (N_high - N_low)

    return mu_interpolated