"""Photon-flux-weighted spectral average and dispersion of absorption coefficients.

Implements equations (1) and (2) of Crovetto 2024 (J. Phys. Energy 6 025009):
the average absorption coefficient and the dispersion of its base-10
logarithm over the wavelength window from 300 nm to the band-gap wavelength,
both weighted by the spectral photon flux of the illumination spectrum.
"""

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.constants as sc
from numpy.typing import NDArray
from scipy.integrate import simpson

from solphin.pv_fom import SAMPLED_RANGES

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


def _resample_common_grid(
        abs_energy_eV: NDArray,
        abs_coeff: NDArray,
        spectrum: NDArray,
        E_gap: float,
        num_points: int = 20001,
) -> tuple[NDArray, NDArray, NDArray]:
    """Interpolate absorption and photon flux onto one fine wavelength grid.

    The grid spans 300 nm to the band-gap wavelength, the integration window
    of Crovetto 2024 equations (1)-(2). The absorption data is converted from
    energy to wavelength space, the illumination spectrum from spectral
    irradiance to photon flux, and both are linearly interpolated onto the
    common grid.

    Parameters
    ----------
    abs_energy_eV : numpy.ndarray
        Photon energies in eV for the absorption data.
    abs_coeff : numpy.ndarray
        Absorption coefficient at each energy point.
    spectrum : numpy.ndarray
        2D array; column 0 is wavelength in nm, column 1 spectral irradiance
        in W m⁻² nm⁻¹.
    E_gap : float
        Band gap energy in eV, setting the upper wavelength cutoff.
    num_points : int, optional
        Number of points in the common wavelength grid. Default is 20001.

    Returns
    -------
    wavelengths : numpy.ndarray
        Common wavelength grid in nm, ascending across the window.
    alpha : numpy.ndarray
        Absorption coefficient interpolated onto the grid.
    photon_flux : numpy.ndarray
        Spectral photon flux φ(λ) on the grid, in photons m⁻² s⁻¹ nm⁻¹.

    Raises
    ------
    ValueError
        If the band gap leaves no wavelength window above 300 nm, or the
        spectrum does not reach the window (e.g. an energy-space spectrum
        from ``convert_spectrum`` was passed instead of a wavelength-space
        one).
    """
    wavelength_min, Eg_wavelength = _extract_int_limits(E_gap)
    if Eg_wavelength <= wavelength_min:
        raise ValueError(
            f"Band gap {E_gap} eV puts the gap wavelength at "
            f"{Eg_wavelength:.1f} nm, at or below the 300 nm cutoff; no "
            "spectral window remains."
        )

    spectrum_wavelengths = np.asarray(spectrum[:, 0], dtype=float)
    spectrum_irradiance = np.asarray(spectrum[:, 1], dtype=float)
    if spectrum_wavelengths.max() < wavelength_min:
        raise ValueError(
            "spectrum must be in wavelength space (column 0 in nm, column 1 in"
            " W m⁻² nm⁻¹, as loaded by load_spectrum); its wavelengths end"
            " below the 300 nm integration window. Was an energy-space"
            " spectrum from convert_spectrum passed?"
        )

    wavelengths = np.linspace(wavelength_min, Eg_wavelength, num_points)

    abs_energy = np.asarray(abs_energy_eV, dtype=float)
    abs_alpha = np.asarray(abs_coeff, dtype=float)
    positive_energy = abs_energy > 0
    abs_wavelengths_nm = _wavelength_conv(abs_energy[positive_energy])
    abs_alpha = abs_alpha[positive_energy]

    abs_order = np.argsort(abs_wavelengths_nm)
    alpha = np.interp(wavelengths, abs_wavelengths_nm[abs_order], abs_alpha[abs_order])

    spec_order = np.argsort(spectrum_wavelengths)
    irradiance = np.interp(
        wavelengths, spectrum_wavelengths[spec_order], spectrum_irradiance[spec_order]
    )
    photon_flux = irradiance * (wavelengths * sc.nano) / (h * c)

    return wavelengths, alpha, photon_flux


def _warn_outside_sampled_range(spectral_average: float, spectral_dispersion: float) -> None:
    """Warn when a computed absorption descriptor leaves the Γₚᵥ sampled range.

    Both descriptors are measurements of the supplied absorption data rather
    than user choices, so they are returned either way. The warning is raised
    here rather than left to the figure of merit so that the cause — a weak
    absorber, or a slow absorption onset — is reported where it arises instead
    of surfacing later as a range error on a number the caller did not pick.

    Parameters
    ----------
    spectral_average : float
        Photon-flux-weighted mean absorption coefficient in cm⁻¹.
    spectral_dispersion : float
        Photon-flux-weighted dispersion of log₁₀ of the absorption
        coefficient, dimensionless.

    Warns
    -----
    UserWarning
        If either descriptor lies outside its range in
        :data:`~solphin.pv_fom.SAMPLED_RANGES`.
    """
    descriptors = (
        ("alpha", "Spectral average", spectral_average),
        ("sigma", "Spectral dispersion", spectral_dispersion),
    )

    for key, label, value in descriptors:

        minimum, maximum, unit = SAMPLED_RANGES[key]

        if minimum <= value <= maximum:
            continue

        unit_text = f" {unit}" if unit else ""

        warnings.warn(
            f"{label} {value:.4g}{unit_text} is outside the"
            f" {minimum:.3g} - {maximum:.3g}{unit_text} range sampled by"
            " Crovetto 2024 table 1. This is a property of the supplied"
            " absorption data, but Γₚᵥ will refuse it unless"
            " allow_out_of_range=True is passed.",
            UserWarning,
            stacklevel=3,
        )


def calculate_spectral_average(
        abs_coeff: Sequence[float] | NDArray,
        photon_flux: Sequence[float] | NDArray,
        wavelengths: Sequence[float] | NDArray,
) -> float:
    """Calculate the photon-flux-weighted average of the absorption coefficient.

    Equation (1) of Crovetto 2024: the absorption coefficient is averaged
    over wavelength with the spectral photon flux φ(λ) as the weight.
    Integration is performed numerically with Simpson's rule.

    Parameters
    ----------
    abs_coeff : sequence of float or numpy.ndarray
        Absorption coefficients α at each wavelength.
    photon_flux : sequence of float or numpy.ndarray
        Spectral photon flux on the same wavelength grid, used as weights.
    wavelengths : sequence of float or numpy.ndarray
        Ascending wavelength grid in nm shared by ``abs_coeff`` and
        ``photon_flux``.

    Returns
    -------
    float
        Photon-flux-weighted average absorption coefficient.
    """
    alpha = np.asarray(abs_coeff, dtype=float)
    flux = np.asarray(photon_flux, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)

    numerator = simpson(alpha * flux, x=wavelengths)
    denominator = simpson(flux, x=wavelengths)

    spectral_average = numerator / denominator if denominator != 0 else 0

    return spectral_average


def calculate_spectral_dispersion(
        abs_coeff: Sequence[float] | NDArray,
        photon_flux: Sequence[float] | NDArray,
        wavelengths: Sequence[float] | NDArray,
) -> float:
    """Calculate the photon-flux-weighted dispersion of the log absorption coefficient.

    Equation (2) of Crovetto 2024: the flux-weighted dispersion of log₁₀(α)
    about log₁₀ of the equation-(1) average. Points with α ⩽ 0, where the
    logarithm is undefined, are excluded from every integral, including the
    average used as the centre; for physical data (α > 0 across the window,
    as the paper assumes) this is exactly equation (2).

    Parameters
    ----------
    abs_coeff : sequence of float or numpy.ndarray
        Absorption coefficients α at each wavelength.
    photon_flux : sequence of float or numpy.ndarray
        Spectral photon flux on the same wavelength grid, used as weights.
    wavelengths : sequence of float or numpy.ndarray
        Ascending wavelength grid in nm shared by ``abs_coeff`` and
        ``photon_flux``.

    Returns
    -------
    float
        Photon-flux-weighted dispersion of log₁₀(α), dimensionless.

    Raises
    ------
    ValueError
        If fewer than two points carry a positive absorption coefficient.
    """
    alpha = np.asarray(abs_coeff, dtype=float)
    flux = np.asarray(photon_flux, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)

    valid = np.isfinite(alpha) & (alpha > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError(
            "Fewer than two positive absorption coefficients; the dispersion"
            " of log(α) is undefined."
        )

    alpha = alpha[valid]
    flux = flux[valid]
    wavelengths = wavelengths[valid]

    spectral_average = calculate_spectral_average(alpha, flux, wavelengths)
    deviation = np.log10(alpha) - np.log10(spectral_average)

    numerator = simpson(flux * deviation ** 2, x=wavelengths)
    denominator = simpson(flux, x=wavelengths)

    spectral_dispersion = float(np.sqrt(numerator / denominator))

    return spectral_dispersion


def generate_spectral_parameters(
        optics_directory: str | Path,
        spectrum: NDArray,
        E_gap: float,
        num_points: int = 20001,
) -> tuple[float, float]:
    """Compute the spectral average and dispersion from absorption and light spectra.

    Implements equations (1) and (2) of Crovetto 2024: ``absorption.dat`` and
    the illumination spectrum are interpolated onto a common wavelength grid
    spanning 300 nm to the band-gap wavelength, the irradiance is converted
    to a photon flux, and the flux-weighted average of α and dispersion of
    log₁₀(α) are integrated over wavelength.

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
    num_points : int, optional
        Number of points in the common wavelength grid. Default is 20001.

    Returns
    -------
    spectral_average : float
        Photon-flux-weighted mean absorption coefficient.
    spectral_dispersion : float
        Photon-flux-weighted dispersion of log₁₀ of the absorption
        coefficient.

    Warns
    -----
    UserWarning
        If either descriptor falls outside the range sampled by Crovetto 2024
        table 1, which Γₚᵥ will refuse. Both are returned regardless.
    """
    abs_file = f'{optics_directory}/absorption.dat'

    abs_energy_eV, abs_coeff = _load_absorption(abs_file)
    wavelengths, alpha, photon_flux = _resample_common_grid(
        abs_energy_eV, abs_coeff, spectrum, E_gap, num_points
    )

    spectral_average = calculate_spectral_average(alpha, photon_flux, wavelengths)
    spectral_dispersion = calculate_spectral_dispersion(alpha, photon_flux, wavelengths)

    _warn_outside_sampled_range(spectral_average, spectral_dispersion)

    return spectral_average, spectral_dispersion
