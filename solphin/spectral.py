import numpy as np
from scipy.integrate import simpson
import scipy.constants as sc

h = sc.h        # Planck's constant (J·s)
c = sc.c        # Speed of light (m/s)
k = sc.k        # Boltzmann constant (J/K)
q = sc.e        # Elementary charge (Coulombs)

def load_absorption(abs_file):

    abs_data = np.loadtxt(abs_file, skiprows=2)

    abs_energy_eV = abs_data[:, 0]
    abs_coeff = abs_data[:, 1]

    return abs_energy_eV, abs_coeff

def wavelength_conv(abs_energy_eV):

    wavelength_m = (h * c) / (abs_energy_eV * q)
    abs_wavelength_nm = wavelength_m * 1e9

    return abs_wavelength_nm

def extract_int_limits(E_gap):

    wavelength_min = 300

    Eg_wavelength = (( h * c ) / (E_gap * q)) * 1e9
    
    return wavelength_min, Eg_wavelength

# Make the truncated spectras

def truncate_abs_spectra(E_gap, abs_energy_eV, abs_coeff):

     wavelength_min, Eg_wavelength = extract_int_limits(E_gap)

     abs_wavelength_nm = wavelength_conv(abs_energy_eV)

     filtered_pairs = [(wl, val) for wl, val in zip(abs_wavelength_nm, abs_coeff) if wavelength_min <= wl <= Eg_wavelength]

     filtered_wavelengths_abs, filtered_abs_coff = zip(*filtered_pairs) if filtered_pairs else ([], [])

     return filtered_wavelengths_abs, filtered_abs_coff

def truncate_light_spectra(spectrum, E_gap):

     wavelength_min, Eg_wavelength = extract_int_limits(E_gap)

     spectrum = np.copy(spectrum)

     wavelength = spectrum[:, 0]
     irradiance = spectrum[:, 1]  # irradiance (W/m2/nm)

     filtered_pairs_spec = [(wl, val) for wl, val in zip(wavelength, irradiance) if wavelength_min <= wl <= Eg_wavelength]

     filtered_wavelengths_spec, filtered_irradiance_spec = zip(*filtered_pairs_spec) if filtered_pairs_spec else ([], [])

     return filtered_wavelengths_spec, filtered_irradiance_spec  

def match_wavelengths(filtered_wavelengths_abs, filtered_wavelengths_spec, filtered_irradiance_spec):
    matched_values = []

    for target_wl in filtered_wavelengths_abs:
        # Calculate absolute differences from target wavelength
        diffs = [abs(spec_wl - target_wl) for spec_wl in filtered_wavelengths_spec]
        min_diff = min(diffs)  # Find the smallest difference
        
        # Find indices where the difference is equal to min_diff
        close_indices = [i for i, d in enumerate(diffs) if d == min_diff]

        # Get the irradiance values at those indices and take the average
        closest_values = [filtered_irradiance_spec[i] for i in close_indices]
        matched_values.append(sum(closest_values) / len(closest_values))  # Average

    # matched_values = np.array(matched_values, dtype = float)

    return matched_values

def calculate_spectral_dispersion(filtered_abs_coff, matched_irradiance, filtered_wavelengths_abs):
    # Convert to log scale
    log_alpha = [np.log(a) for a in filtered_abs_coff]

    # Compute weighted mean log(α)
    numerator_mean = sum(irr * log_a for irr, log_a in zip(matched_irradiance, log_alpha)) #not mean?
    denominator_mean = sum(matched_irradiance)
    log_alpha_bar = numerator_mean / denominator_mean  # Weighted mean log(alpha)

    # Compute numerator (variance term)

    numerator_x = list(irr * (log_a - log_alpha_bar) ** 2 for irr, log_a in zip(matched_irradiance, log_alpha))
    numerator_variance = simpson(numerator_x)

    # Compute denominator (normalization term)
    denominator_variance = simpson(matched_irradiance)

    # Spectral density
    spectral_dispersion = np.sqrt(numerator_variance / denominator_variance)
    
    return spectral_dispersion

# Function to calculate spectral average
def calculate_spectral_average(filtered_abs_coff, matched_irradiance, filtered_wavelengths_abs):
    # Compute numerator (weighted sum of alpha)
    numerator_1 = list(alpha * irr for alpha, irr in zip(filtered_abs_coff, matched_irradiance))

    numerator = simpson(numerator_1)

    # Compute denominator (total irradiance)
    denominator = simpson(matched_irradiance)

    # Spectral average
    spectral_average = numerator / denominator if denominator != 0 else 0
    
    return spectral_average


def generate_spectral_parameters(optics_directory, spectrum, E_gap):

    abs_file = f'{optics_directory}/absorption.dat'

    abs_energy_eV, abs_coeff = load_absorption(abs_file)
    filtered_wavelengths_abs, filtered_abs_coff = truncate_abs_spectra(E_gap, abs_energy_eV, abs_coeff)
    filtered_wavelengths_spec, filtered_irradiance_spec  = truncate_light_spectra(spectrum, E_gap)
    matched_irradiance = match_wavelengths(filtered_wavelengths_abs, filtered_wavelengths_spec, filtered_irradiance_spec)

    spectral_average = calculate_spectral_average(filtered_abs_coff, matched_irradiance, filtered_wavelengths_abs)
    spectral_dispersion = calculate_spectral_dispersion(filtered_abs_coff, matched_irradiance, filtered_wavelengths_abs)

    return spectral_average, spectral_dispersion

