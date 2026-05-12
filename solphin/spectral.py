import numpy as np
from scipy.integrate import simpson
import scipy.constants as sc

h = sc.h        # Planck's constant (J·s)
c = sc.c        # Speed of light (m/s)
k = sc.k        # Boltzmann constant (J/K)
q = sc.e        # Elementary charge (Coulombs)

def load_absorption(abs_file):
    """
    Loads absorption coefficient data from a file.

    This function reads a text file containing absorption data and extracts
    the energy and absorption coefficient columns for further optical analysis.

    Parameters:
        abs_file(string or Path): path to the absorption data file.
            Expected format is a whitespace-delimited text file with at least
            two columns, where:
            - column 0 is energy in eV
            - column 1 is absorption coefficient

    Returns:
        abs_energy_eV(np.array): energy values in eV.
        abs_coeff(np.array): absorption coefficient corresponding to each energy value.
    """

    abs_data = np.loadtxt(abs_file, skiprows=2)

    abs_energy_eV = abs_data[:, 0]
    abs_coeff = abs_data[:, 1]

    return abs_energy_eV, abs_coeff

def wavelength_conv(abs_energy_eV):
    """
    Converts photon energy values to wavelength in nanometers.

    This function transforms energy values (in eV) into corresponding photon
    wavelengths using the standard relation between energy and wavelength.

    Parameters:
        abs_energy_eV(np.array): photon energies in eV.

    Returns:
        abs_wavelength_nm(np.array): corresponding wavelengths in nanometers.
    """

    wavelength_m = (h * c) / (abs_energy_eV * q)
    abs_wavelength_nm = wavelength_m * 1e9

    return abs_wavelength_nm

def extract_int_limits(E_gap):
    """
    Determines wavelength integration limits based on a material band gap.

    This function computes the lower and upper wavelength bounds used for spectral
    integration in photovoltaic or optical calculations. The upper bound is set by
    the band gap energy, converted from eV to wavelength.

    Parameters:
        E_gap(float): band gap energy in eV.

    Returns:
        wavelength_min(float): lower wavelength integration limit in nm (fixed at 300 nm).
        Eg_wavelength(float): wavelength corresponding to the band gap energy in nm.
    """

    wavelength_min = 300

    Eg_wavelength = (( h * c ) / (E_gap * q)) * 1e9
    
    return wavelength_min, Eg_wavelength

# Make the truncated spectras

def truncate_abs_spectra(E_gap, abs_energy_eV, abs_coeff):
     """
    Filters absorption spectra to include only wavelengths within a band-gap-defined range.

    This function converts absorption energies into wavelengths, then truncates the dataset
    to retain only values between a fixed lower wavelength limit and the wavelength
    corresponding to the material band gap. This is commonly used to isolate the
    relevant portion of the spectrum for photovoltaic absorption analysis.

    Parameters:
        E_gap(float): band gap energy in eV, used to determine the upper wavelength cutoff.
        abs_energy_eV(np.array): photon energies in eV corresponding to the absorption data.
        abs_coeff(np.array): absorption coefficient values corresponding to each energy point.

    Returns:
        filtered_wavelengths_abs(tuple or list): wavelengths (nm) within the valid range.
        filtered_abs_coff(tuple or list): absorption coefficients corresponding to the
            filtered wavelength range.
    """

     wavelength_min, Eg_wavelength = extract_int_limits(E_gap)

     abs_wavelength_nm = wavelength_conv(abs_energy_eV)

     filtered_pairs = [(wl, val) for wl, val in zip(abs_wavelength_nm, abs_coeff) if wavelength_min <= wl <= Eg_wavelength]

     filtered_wavelengths_abs, filtered_abs_coff = zip(*filtered_pairs) if filtered_pairs else ([], [])

     return filtered_wavelengths_abs, filtered_abs_coff

def truncate_light_spectra(spectrum, E_gap):
     """
    Truncates a light spectrum to a wavelength range defined by a material band gap.

    This function filters a spectral irradiance dataset so that only wavelengths
    between a fixed lower cutoff and the band-gap-dependent upper cutoff are retained.
    This is typically used to restrict solar or illumination spectra to the
    energetically relevant range for absorption or efficiency calculations.

    Parameters:
        spectrum(np.array): 2D array where:
            - column 0 is wavelength in nm
            - column 1 is spectral irradiance (W m^-2 nm^-1)
        E_gap(float): band gap energy in eV, used to determine the upper wavelength cutoff.

    Returns:
        filtered_wavelengths_spec(tuple or list): wavelength values (nm) within the valid range.
        filtered_irradiance_spec(tuple or list): corresponding spectral irradiance values.
    """

     wavelength_min, Eg_wavelength = extract_int_limits(E_gap)

     spectrum = np.copy(spectrum)

     wavelength = spectrum[:, 0]
     irradiance = spectrum[:, 1]  # irradiance (W/m2/nm)

     filtered_pairs_spec = [(wl, val) for wl, val in zip(wavelength, irradiance) if wavelength_min <= wl <= Eg_wavelength]

     filtered_wavelengths_spec, filtered_irradiance_spec = zip(*filtered_pairs_spec) if filtered_pairs_spec else ([], [])

     return filtered_wavelengths_spec, filtered_irradiance_spec  

def match_wavelengths(filtered_wavelengths_abs, filtered_wavelengths_spec, filtered_irradiance_spec):
    """
    Matches absorption wavelengths to the closest wavelengths in a light spectrum
    and returns corresponding irradiance values.

    This function performs a nearest-neighbour mapping between two wavelength
    grids (typically absorption and illumination spectra). For each wavelength in
    the absorption dataset, it finds the closest wavelength(s) in the light spectrum
    and assigns the corresponding irradiance value(s), averaging in case of ties.

    Parameters:
        filtered_wavelengths_abs(iterable): wavelengths (nm) from the absorption dataset.
        filtered_wavelengths_spec(iterable): wavelengths (nm) from the light spectrum.
        filtered_irradiance_spec(iterable): spectral irradiance values corresponding
            to filtered_wavelengths_spec (W m^-2 nm^-1).

    Returns:
        matched_values(list): irradiance values mapped onto the absorption wavelength
            grid using nearest-neighbour matching.
    """
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
    """
    Calculates the spectral dispersion of an absorption spectrum weighted by incident irradiance.

    This function computes a weighted statistical measure of the spread of the logarithm
    of the absorption coefficient. It is analogous to a weighted standard deviation of
    log(α), where weights are given by the matched spectral irradiance.

    Parameters:
        filtered_abs_coff(iterable): absorption coefficients (α) corresponding to each wavelength.
        matched_irradiance(iterable): spectral irradiance values matched to the absorption grid
            (used as weighting factors).
        filtered_wavelengths_abs(iterable): wavelength grid (nm) for absorption data.
            (Note: included for completeness but not directly used in the calculation.)

    Returns:
        spectral_dispersion(float): weighted spectral dispersion of log(absorption coefficient).
    """
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


def calculate_spectral_average(filtered_abs_coff, matched_irradiance, filtered_wavelengths_abs):
    """
    Calculates the irradiance-weighted spectral average of the absorption coefficient.

    This function computes the mean absorption coefficient weighted by the incident
    spectral irradiance. Integration is performed numerically using Simpson’s rule.

    Parameters:
        filtered_abs_coff(iterable): absorption coefficients (α) at each wavelength.
        matched_irradiance(iterable): spectral irradiance values matched to the absorption grid.
        filtered_wavelengths_abs(iterable): wavelength grid (nm) for absorption data.
            (Included for interface consistency; not directly used.)

    Returns:
        spectral_average(float): irradiance-weighted average absorption coefficient.
    """
    # Compute numerator (weighted sum of alpha)
    numerator_1 = list(alpha * irr for alpha, irr in zip(filtered_abs_coff, matched_irradiance))

    numerator = simpson(numerator_1)

    # Compute denominator (total irradiance)
    denominator = simpson(matched_irradiance)

    # Spectral average
    spectral_average = numerator / denominator if denominator != 0 else 0
    
    return spectral_average


def generate_spectral_parameters(optics_directory, spectrum, E_gap):
    """
    Computes spectral average and spectral dispersion parameters from absorption and illumination spectra.

    This function combines absorption data and a truncated light spectrum to compute two
    key descriptors of spectral behaviour:
    - the irradiance-weighted average absorption coefficient
    - the irradiance-weighted dispersion of the logarithmic absorption coefficient

    These metrics are commonly used to quantify how absorption varies across the
    relevant spectral range defined by the material band gap.

    Parameters:
        optics_directory(string or Path): directory containing optical data files,
            including 'absorption.dat'.
        spectrum(np.array): incident light spectrum as a 2D array where:
            - column 0 is wavelength in nm
            - column 1 is spectral irradiance (W m^-2 nm^-1)
        E_gap(float): band gap energy in eV used to define spectral cutoffs.

    Returns:
        spectral_average(float): irradiance-weighted mean absorption coefficient.
        spectral_dispersion(float): irradiance-weighted dispersion of log(absorption).
    """

    abs_file = f'{optics_directory}/absorption.dat'

    abs_energy_eV, abs_coeff = load_absorption(abs_file)
    filtered_wavelengths_abs, filtered_abs_coff = truncate_abs_spectra(E_gap, abs_energy_eV, abs_coeff)
    filtered_wavelengths_spec, filtered_irradiance_spec  = truncate_light_spectra(spectrum, E_gap)
    matched_irradiance = match_wavelengths(filtered_wavelengths_abs, filtered_wavelengths_spec, filtered_irradiance_spec)

    spectral_average = calculate_spectral_average(filtered_abs_coff, matched_irradiance, filtered_wavelengths_abs)
    spectral_dispersion = calculate_spectral_dispersion(filtered_abs_coff, matched_irradiance, filtered_wavelengths_abs)

    return spectral_average, spectral_dispersion

