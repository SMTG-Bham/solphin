"""Irradiance-weighted spectral average and dispersion of absorption coefficients."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.constants as sc
from numpy.typing import NDArray
from scipy.integrate import simpson

h = sc.h  # Planck's constant (J·s)
c = sc.c  # Speed of light (m/s)
k = sc.k  # Boltzmann constant (J/K)
q = sc.e  # Elementary charge (Coulombs)


def _load_absorption(abs_file: str | Path) -> tuple[NDArray, NDArray]:
    """Load absorption coefficient data from a file.

    Parameters
    ----------
    abs_file : str or Path
        Path to the absorption data file: whitespace-delimited text with two
        header rows, column 0 energy in eV, column 1 absorption coefficient.

    Returns
    -------
    abs_energy_eV : numpy.ndarray
        Energy values in eV.
    abs_coeff : numpy.ndarray
        Absorption coefficient at each energy value.
    """
    abs_data = np.loadtxt(abs_file, skiprows=2)

    abs_energy_eV = abs_data[:, 0]
    abs_coeff = abs_data[:, 1]

    return abs_energy_eV, abs_coeff


def _wavelength_conv(abs_energy_eV: NDArray) -> NDArray:
    """Convert photon energies in eV to wavelengths in nm.

    Parameters
    ----------
    abs_energy_eV : numpy.ndarray
        Photon energies in eV.

    Returns
    -------
    numpy.ndarray
        Corresponding wavelengths in nm.
    """
    wavelength_m = (h * c) / (abs_energy_eV * q)
    abs_wavelength_nm = wavelength_m / sc.nano

    return abs_wavelength_nm


def _extract_int_limits(E_gap: float) -> tuple[float, float]:
    """Determine wavelength integration limits from a material band gap.

    Parameters
    ----------
    E_gap : float
        Band gap energy in eV.

    Returns
    -------
    wavelength_min : float
        Lower wavelength integration limit, fixed at 300 nm.
    Eg_wavelength : float
        Wavelength corresponding to the band gap energy, in nm.
    """
    wavelength_min = 300

    Eg_wavelength = ((h * c) / (E_gap * q)) / sc.nano

    return wavelength_min, Eg_wavelength


# Make the truncated spectras

def _truncate_abs_spectra(
        E_gap: float, abs_energy_eV: NDArray, abs_coeff: NDArray
) -> tuple[Sequence[float], Sequence[float]]:
    """Truncate an absorption spectrum to the band-gap-defined wavelength range.

    Converts the absorption energies to wavelengths, then keeps only the values
    between the fixed lower limit and the band-gap wavelength.

    Parameters
    ----------
    E_gap : float
        Band gap energy in eV, setting the upper wavelength cutoff.
    abs_energy_eV : numpy.ndarray
        Photon energies in eV for the absorption data.
    abs_coeff : numpy.ndarray
        Absorption coefficient at each energy point.

    Returns
    -------
    filtered_wavelengths_abs : sequence of float
        Wavelengths in nm within the valid range.
    filtered_abs_coff : sequence of float
        Absorption coefficients for the filtered wavelengths.
    """
    wavelength_min, Eg_wavelength = _extract_int_limits(E_gap)

    abs_wavelength_nm = _wavelength_conv(abs_energy_eV)

    filtered_pairs = [(wl, val) for wl, val in zip(abs_wavelength_nm, abs_coeff) if
                      wavelength_min <= wl <= Eg_wavelength]

    filtered_wavelengths_abs, filtered_abs_coff = zip(*filtered_pairs) if filtered_pairs else ([], [])

    return filtered_wavelengths_abs, filtered_abs_coff


def _truncate_light_spectra(
        spectrum: NDArray, E_gap: float
) -> tuple[Sequence[float], Sequence[float]]:
    """Truncate a light spectrum to the band-gap-defined wavelength range.

    Parameters
    ----------
    spectrum : numpy.ndarray
        2D array; column 0 is wavelength in nm, column 1 spectral irradiance
        in W m⁻² nm⁻¹.
    E_gap : float
        Band gap energy in eV, setting the upper wavelength cutoff.

    Returns
    -------
    filtered_wavelengths_spec : sequence of float
        Wavelengths in nm within the valid range.
    filtered_irradiance_spec : sequence of float
        Spectral irradiance for the filtered wavelengths.
    """
    wavelength_min, Eg_wavelength = _extract_int_limits(E_gap)

    spectrum = np.copy(spectrum)

    wavelength = spectrum[:, 0]
    irradiance = spectrum[:, 1]  # irradiance (W/m2/nm)

    filtered_pairs_spec = [(wl, val) for wl, val in zip(wavelength, irradiance) if
                           wavelength_min <= wl <= Eg_wavelength]

    filtered_wavelengths_spec, filtered_irradiance_spec = zip(*filtered_pairs_spec) if filtered_pairs_spec else ([], [])

    return filtered_wavelengths_spec, filtered_irradiance_spec


def _match_wavelengths(
        filtered_wavelengths_abs: Sequence[float],
        filtered_wavelengths_spec: Sequence[float],
        filtered_irradiance_spec: Sequence[float],
) -> list[float]:
    """Map irradiance values onto an absorption wavelength grid by nearest neighbour.

    For each absorption wavelength, the closest wavelength(s) in the light
    spectrum supply the irradiance value, averaging in case of ties.

    Parameters
    ----------
    filtered_wavelengths_abs : sequence of float
        Wavelengths in nm from the absorption dataset.
    filtered_wavelengths_spec : sequence of float
        Wavelengths in nm from the light spectrum.
    filtered_irradiance_spec : sequence of float
        Spectral irradiance in W m⁻² nm⁻¹ for ``filtered_wavelengths_spec``.

    Returns
    -------
    list of float
        Irradiance values mapped onto the absorption wavelength grid.
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


def calculate_spectral_dispersion(
        filtered_abs_coff: Sequence[float] | NDArray,
        matched_irradiance: Sequence[float] | NDArray,
        filtered_wavelengths_abs: Sequence[float] | NDArray,
) -> float:
    """Calculate the irradiance-weighted dispersion of the log absorption coefficient.

    Analogous to a weighted standard deviation of log(α), with the matched
    spectral irradiance as the weights.

    Parameters
    ----------
    filtered_abs_coff : sequence of float or numpy.ndarray
        Absorption coefficients α at each wavelength.
    matched_irradiance : sequence of float or numpy.ndarray
        Spectral irradiance matched to the absorption grid, used as weights.
    filtered_wavelengths_abs : sequence of float or numpy.ndarray
        Wavelength grid in nm for the absorption data. Unused; accepted for
        interface consistency.

    Returns
    -------
    float
        Weighted spectral dispersion of the log absorption coefficient.
    """
    valid_data = [
        (a, irr)
        for a, irr in zip(filtered_abs_coff, matched_irradiance)
        if a > 0 and np.isfinite(a)
    ]

    filtered_abs_coff, matched_irradiance = zip(*valid_data)
    # Convert to log scale
    log_alpha = [np.log(a) for a in filtered_abs_coff]

    # Compute weighted mean log(α)
    numerator_mean = sum(irr * log_a for irr, log_a in zip(matched_irradiance, log_alpha))  # not mean?
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


def calculate_spectral_average(
        filtered_abs_coff: Sequence[float] | NDArray,
        matched_irradiance: Sequence[float] | NDArray,
        filtered_wavelengths_abs: Sequence[float] | NDArray,
) -> float:
    """Calculate the irradiance-weighted average of the absorption coefficient.

    Integration is performed numerically with Simpson's rule.

    Parameters
    ----------
    filtered_abs_coff : sequence of float or numpy.ndarray
        Absorption coefficients α at each wavelength.
    matched_irradiance : sequence of float or numpy.ndarray
        Spectral irradiance matched to the absorption grid, used as weights.
    filtered_wavelengths_abs : sequence of float or numpy.ndarray
        Wavelength grid in nm for the absorption data. Unused; accepted for
        interface consistency.

    Returns
    -------
    float
        Irradiance-weighted average absorption coefficient.
    """
    # Compute numerator (weighted sum of alpha)
    numerator_1 = list(alpha * irr for alpha, irr in zip(filtered_abs_coff, matched_irradiance))

    numerator = simpson(numerator_1)

    # Compute denominator (total irradiance)
    denominator = simpson(matched_irradiance)

    # Spectral average
    spectral_average = numerator / denominator if denominator != 0 else 0

    return spectral_average


def generate_spectral_parameters(
        optics_directory: str | Path, spectrum: NDArray, E_gap: float
) -> tuple[float, float]:
    """Compute the spectral average and dispersion from absorption and light spectra.

    Combines the absorption data with a truncated light spectrum to quantify
    how absorption varies across the spectral range set by the band gap.

    Parameters
    ----------
    optics_directory : str or Path
        Directory containing the optical data files, including
        ``absorption.dat``.
    spectrum : numpy.ndarray
        Incident light spectrum; column 0 is wavelength in nm, column 1
        spectral irradiance in W m⁻² nm⁻¹.
    E_gap : float
        Band gap energy in eV used to define the spectral cutoffs.

    Returns
    -------
    spectral_average : float
        Irradiance-weighted mean absorption coefficient.
    spectral_dispersion : float
        Irradiance-weighted dispersion of the log absorption coefficient.
    """
    abs_file = f'{optics_directory}/absorption.dat'

    abs_energy_eV, abs_coeff = _load_absorption(abs_file)
    filtered_wavelengths_abs, filtered_abs_coff = _truncate_abs_spectra(E_gap, abs_energy_eV, abs_coeff)
    filtered_wavelengths_spec, filtered_irradiance_spec = _truncate_light_spectra(spectrum, E_gap)
    matched_irradiance = _match_wavelengths(filtered_wavelengths_abs, filtered_wavelengths_spec,
                                            filtered_irradiance_spec)

    spectral_average = calculate_spectral_average(filtered_abs_coff, matched_irradiance, filtered_wavelengths_abs)
    spectral_dispersion = calculate_spectral_dispersion(filtered_abs_coff, matched_irradiance, filtered_wavelengths_abs)

    return spectral_average, spectral_dispersion
