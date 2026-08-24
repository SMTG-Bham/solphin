"""Detailed-balance (Shockley-Queisser) limit efficiency and its constituent quantities."""

from importlib.resources import files
from typing import overload

import numpy as np
import scipy.constants as sc
from numpy.typing import NDArray

h = sc.h  # Planck's constant (J·s)
c = sc.c  # Speed of light (m/s)
k = sc.k  # Boltzmann constant (J/K)
q = sc.e  # Elementary charge (Coulombs)


# Convert the spectrum to the useful units - taken from https://github.com/kaklin/sq-limit?tab=readme-ov-file

def load_spectrum(spectrum_type: str) -> NDArray:
    """Load a predefined spectral irradiance dataset from bundled resources.

    Parameters
    ----------
    spectrum_type : str
        Identifier for the spectrum: ``"AM1.5"``, ``"Fluorescent"``,
        ``"Blue LED"``, ``"Green LED"``, ``"Red LED"``, ``"White LED"``,
        ``"IR LED"`` or ``"Photopic"``. Unrecognised values fall back to
        ``"AM1.5"``.

    Returns
    -------
    numpy.ndarray
        2D array; column 0 is wavelength in nm, column 1 spectral irradiance
        in W m⁻² nm⁻¹.
    """
    if spectrum_type == "AM1.5":
        filename = 'ASTMG173.csv'

    elif spectrum_type == "Fluorescent":
        filename = 'fluorescent.csv'

    elif spectrum_type == "Blue LED":
        filename = 'led_blue.csv'

    elif spectrum_type == "Green LED":
        filename = 'led_green.csv'

    elif spectrum_type == "Red LED":
        filename = 'led_red.csv'

    elif spectrum_type == "White LED":
        filename = 'led_white.csv'

    elif spectrum_type == "IR LED":
        filename = 'led_ir.csv'

    elif spectrum_type == "Photopic":
        filename = 'photopic.csv'

    else:
        print("Unrecognisable spectrum selected")
        print("Options: AM1.5, Fluorescent, Blue LED, Green LED, Red LED, White LED, IR LED, Photopic")
        print("reverting to AM1.5")

        filename = 'ASTMG173.csv'

    csv_path = files("solphin.resources") / f"{filename}"

    with csv_path.open("r", encoding="utf-8") as f:
        spectrum = np.loadtxt(f, delimiter=",", skiprows=1)

    return spectrum


def convert_spectrum(spectrum: NDArray) -> NDArray:
    """Convert an irradiance spectrum to a photon-flux spectrum over energy.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectrum as loaded by ``load_spectrum``: wavelength in nm against
        spectral irradiance in W m⁻² nm⁻¹.

    Returns
    -------
    numpy.ndarray
        Converted spectrum: photon energy in eV against photon flux per unit
        energy in m⁻² s⁻¹ eV⁻¹.
    """
    converted = np.copy(spectrum)
    converted[:, 0] = converted[:, 0] * 1e-9  # wavelength to m
    converted[:, 1] = converted[:, 1] / 1e-9  # irradiance to W/m2/m (from W/m2/nm)

    E = h * c / converted[:, 0]  # Bandgap in J
    d_lambda_d_E = h * c / E ** 2
    converted[:, 1] = converted[:, 1] * d_lambda_d_E * q / E
    converted[:, 0] = E / q

    return converted


def _photons_above_bandgap(E_gap: float, photon_spectrum: NDArray) -> float:
    """Count the photons above a given band gap.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.

    Returns
    -------
    float
        Integrated photon flux above the band gap.
    """
    indexes = np.where(photon_spectrum[:, 0] > E_gap)
    y = photon_spectrum[indexes, 1][0]
    x = photon_spectrum[indexes, 0][0]
    return np.trapezoid(y[::-1], x[::-1])


def _rr0(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> float:
    """Calculate the radiative recombination rate at zero quasi-Fermi-level splitting.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    Tcell : float
        Operating temperature of the cell in K.

    Returns
    -------
    float
        Radiative recombination rate in cm⁻³ s⁻¹.
    """
    k_eV = k / q
    h_eV = h / q
    const = (2 * np.pi) / (c ** 2 * h_eV ** 3)

    E = photon_spectrum[::-1,]  # in increasing order of bandgap energy
    egap_index = np.where(E[:, 0] >= E_gap)
    numerator = E[:, 0] ** 2
    exponential_in = E[:, 0] / (k_eV * Tcell)
    denominator = np.exp(exponential_in) - 1
    integrand = numerator / denominator

    integral = np.trapezoid(integrand[egap_index], E[egap_index, 0])

    result = const * integral
    return result[0]


def recomb_rate(E_gap: float, photon_spectrum: NDArray, voltage: float, Tcell: float) -> float:
    """Calculate the radiative recombination rate at an applied voltage.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    voltage : float
        Applied voltage in V.
    Tcell : float
        Operating temperature of the cell in K.

    Returns
    -------
    float
        Radiative recombination rate in cm⁻³ s⁻¹.
    """
    print('recomb rate')
    return q * _rr0(E_gap, photon_spectrum) * np.exp(q * voltage / (k * Tcell))


@overload
def current_density(
        E_gap: float, photon_spectrum: NDArray, voltage: float, Tcell: float
) -> float: ...
@overload
def current_density(
        E_gap: float, photon_spectrum: NDArray, voltage: NDArray, Tcell: float
) -> NDArray: ...
def current_density(
        E_gap: float, photon_spectrum: NDArray, voltage: float | NDArray, Tcell: float
) -> float | NDArray:
    """Calculate the current density at an applied voltage.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    voltage : float or numpy.ndarray
        Applied voltage in V.
    Tcell : float
        Operating temperature of the cell in K.

    Returns
    -------
    float or numpy.ndarray
        Current density in C cm⁻³ s⁻¹. Scalar for scalar ``voltage``,
        elementwise array otherwise.
    """
    return q * (_photons_above_bandgap(E_gap, photon_spectrum) - _rr0(E_gap, photon_spectrum, Tcell) * np.exp(
        q * voltage / (k * Tcell)) - 1)


def jsc(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> float:
    """Calculate the short-circuit current density.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    Tcell : float
        Operating temperature of the cell in K.

    Returns
    -------
    float
        Current density at zero applied voltage in C cm⁻³ s⁻¹.
    """
    return current_density(E_gap, photon_spectrum, 0, Tcell)


def voc(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> float:
    """Calculate the open-circuit voltage.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    Tcell : float
        Operating temperature of the cell in K.

    Returns
    -------
    float
        Maximum voltage across the cell with no current flow, in V.
    """
    Jph = _photons_above_bandgap(E_gap, photon_spectrum)
    J0 = _rr0(E_gap, photon_spectrum, Tcell)

    return (k * Tcell / q) * np.log(Jph / J0 + 1)


def v_at_mpp(E_gap: float, photon_spectrum: NDArray) -> float:
    """Calculate the voltage at the maximum power point.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.

    Returns
    -------
    float
        Voltage at the maximum power point in V.
    """
    v_open = voc(E_gap, photon_spectrum)
    # print v_open
    v = np.linspace(0, v_open)
    index = np.where(
        v * current_density(E_gap, photon_spectrum, v) == max(v * current_density(E_gap, photon_spectrum, v)))
    return v[index][0]


def j_at_mpp(E_gap: float, photon_spectrum: NDArray) -> float:
    """Calculate the current density at the maximum power point.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.

    Returns
    -------
    float
        Current density at the maximum power point in C cm⁻³ s⁻¹.
    """
    return max_power(E_gap, photon_spectrum) / v_at_mpp(E_gap, photon_spectrum)


def max_power(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> float:
    """Calculate the maximum power of a solar cell.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    Tcell : float
        Operating temperature of the cell in K.

    Returns
    -------
    float
        Maximum power of the cell in V C cm⁻³ s⁻¹.
    """
    v_open = voc(E_gap, photon_spectrum, Tcell)
    v = np.linspace(0, v_open)
    index = np.where(v * current_density(E_gap, photon_spectrum, v, Tcell) == max(
        v * current_density(E_gap, photon_spectrum, v, Tcell)))
    return max(v * current_density(E_gap, photon_spectrum, v, Tcell))


def max_eff(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> float:
    """Calculate the maximum efficiency of a solar cell.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    Tcell : float
        Operating temperature of the cell in K.

    Returns
    -------
    float
        Maximum efficiency of the cell relative to the total irradiance,
        as a dimensionless fraction.
    """
    photon_spectrum_1 = photon_spectrum[::-1, 1]
    photon_spectrum_0 = photon_spectrum[::-1, 0]

    irradiance = np.trapezoid(photon_spectrum_1 * q * photon_spectrum_0, photon_spectrum_0)
    return max_power(E_gap, photon_spectrum, Tcell) / irradiance


def fill_factor(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> float:
    """Calculate the fill factor of a solar cell.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    Tcell : float
        Operating temperature of the cell in K.

    Returns
    -------
    float
        Fill factor of the cell, dimensionless.
    """
    j_sc = jsc(E_gap, photon_spectrum)
    v_oc = voc(E_gap, photon_spectrum, Tcell)
    v_mpp = v_at_mpp(E_gap, photon_spectrum)
    j_mpp = j_at_mpp(E_gap, photon_spectrum)

    fill_factor = (j_mpp * v_mpp) / (j_sc, v_oc)

    return fill_factor
