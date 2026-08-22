import logging
from importlib.resources import files
from typing import overload

import numpy as np
import scipy.constants as sc
from numpy.typing import NDArray

logging.basicConfig(level=logging.INFO)

""" This section details the calculation of the detailed balance limit efficiency and associated values."""

h = sc.h  # Planck's constant (J·s)
c = sc.c  # Speed of light (m/s)
k = sc.k  # Boltzmann constant (J/K)
q = sc.e  # Elementary charge (Coulombs)


# Convert the spectrum to the useful units - taken from https://github.com/kaklin/sq-limit?tab=readme-ov-file

def load_spectrum(spectrum_type: str) -> NDArray:
    """
    Loads a predefined spectral irradiance dataset from bundled resource files.

    This function selects a spectrum based on a predefined set of illumination sources
    (e.g. solar AM1.5, LEDs, fluorescent, photopic response), loads the corresponding
    CSV file from package resources, and returns it as a numerical array.

    Parameters:
        spectrum_type(str): identifier for the desired spectrum.
            Supported options:
            - "AM1.5"
            - "Fluorescent"
            - "Blue LED"
            - "Green LED"
            - "Red LED"
            - "White LED"
            - "IR LED"
            - "Photopic"

            If an unrecognised value is provided, the function defaults to "AM1.5".

    Returns:
        spectrum(np.ndarray): 2D array where:
            - column 0 is wavelength (nm)
            - column 1 is spectral irradiance
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
    """
    Converts the input spectrum from standard format to the required units for this code. 

    Parameters:
        spectrum(numpy.ndarray): Input spectrum loaded from a csv file with numpy.loadtxt.

    Returns:
        photon_spectrum(numpy.ndarray): Output spectrum as numpy ndarray.

    Spectrum input:
        y: Irradiance (W/m2/nm)
        x: Wavelength (nm)
    Converted output:
        y: Number of photons (Np/m2/s/dE)
        x: Energy (eV)
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
    """Counts number of photons above given bandgap.
    
    Parameters:
        E_gap(float): Optical Band Gap in eV  
        photon_spectrum(numpy.ndarray): Output spectrum as numpy ndarray.

    Returns:
       (float): Integration of the spectrum for the number of photons above the bandgap.  
    """
    indexes = np.where(photon_spectrum[:, 0] > E_gap)
    y = photon_spectrum[indexes, 1][0]
    x = photon_spectrum[indexes, 0][0]
    return np.trapezoid(y[::-1], x[::-1])


def _rr0(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> float:
    """
    Calculates the radiative recombination rate at 0 Quasi-Fermi Level splitting. 

    Parameters: 
        E_gap(float):  Optical Band Gap in eV  
        photon_spectrum(numpy.ndarray): Output spectrum as numpy ndarray.
        Tcell(float): Operating temperature of the cell in K

    Returns:
        Radiative recomination rate(float) in cm⁻³s⁻¹ 

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
    """
    Calculates the radiative recombination rate. 

    Parameters: 
        E_gap(float):  Optical Band Gap in eV  
        photon_spectrum(numpy.ndarray): Output spectrum as numpy ndarray.
        voltage(float): Open circuit voltage in V
        Tcell(float): Operating temperature of the cell in K

    Returns:
        Radiative recomination rate(float) in cm⁻³s⁻¹ 

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
    """
    Calculates the current density. 

    Parameters: 
        E_gap(float):  Optical Band Gap in eV  
        photon_spectrum(numpy.ndarray): Output spectrum as numpy ndarray.
        voltage(float): Open circuit voltage in V
        Tcell(float): Operating temperature of the cell in K

    Returns:
        Current density (float):  Current that flows across a cross sectional area in C cm⁻³s⁻¹. 

    """

    return q * (_photons_above_bandgap(E_gap, photon_spectrum) - _rr0(E_gap, photon_spectrum, Tcell) * np.exp(
        q * voltage / (k * Tcell)) - 1)


def jsc(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> float:
    """
    Calculates the current density. 

    Parameters: 
        E_gap(float):  Optical Band Gap in eV  
        photon_spectrum(numpy.ndarray): Output spectrum as numpy ndarray.
        Tcell(float): Operating temperature of the cell in K

    Returns:
        Short circuit current density (float):  Current that flows across a cross sectional area at 0 applied voltage in C cm⁻³s⁻¹.  

    """

    return current_density(E_gap, photon_spectrum, 0, Tcell)


def voc(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> float:
    """
    Calculates the open circuit voltage.

    Parameters: 
        E_gap(float):  Optical Band Gap in eV  
        photon_spectrum(numpy.ndarray): Output spectrum as numpy ndarray.
        Tcell(float): Operating temperature of the cell in K

    Returns:
        Open circuit voltage (float):  Maximum voltage across a solar cell with no current flow in V. 

    """

    Jph = _photons_above_bandgap(E_gap, photon_spectrum)
    J0 = _rr0(E_gap, photon_spectrum, Tcell)

    return (k * Tcell / q) * np.log(Jph / J0 + 1)


def v_at_mpp(E_gap: float, photon_spectrum: NDArray) -> float:
    """
    Calculates the voltage at maximum power point (mpp) of a solar cell.

    Parameters: 
        E_gap(float):  Optical Band Gap in eV  
        photon_spectrum(numpy.ndarray): Output spectrum as numpy ndarray.

    Returns:
        Voltage at MPP (float):  Voltage across a solar cell at the maximum power point in V. 

    """

    v_open = voc(E_gap, photon_spectrum)
    # print v_open
    v = np.linspace(0, v_open)
    index = np.where(
        v * current_density(E_gap, photon_spectrum, v) == max(v * current_density(E_gap, photon_spectrum, v)))
    return v[index][0]


def j_at_mpp(E_gap: float, photon_spectrum: NDArray) -> float:
    """
    Calculates the current at maximum power point (mpp) of a solar cell.

    Parameters: 
        E_gap(float):  Optical Band Gap in eV  
        photon_spectrum(numpy.ndarray): Output spectrum as numpy ndarray.

    Returns:
       Current at MPP (float):  Current across a solar cell at the maximum power point. 

    """

    return max_power(E_gap, photon_spectrum) / v_at_mpp(E_gap, photon_spectrum)


def max_power(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> float:
    """
    Calculates the maximum power of a solar cell.

    Parameters: 
        E_gap(float):  Optical Band Gap in eV  
        photon_spectrum(numpy.ndarray): Output spectrum as numpy ndarray.
        Tcell(float): Operating temperature of the cell in K

    Returns:
       Maximum power (float):  Maximum power of the solar cell in V C cm⁻³ s⁻¹. 

    """

    v_open = voc(E_gap, photon_spectrum, Tcell)
    v = np.linspace(0, v_open)
    index = np.where(v * current_density(E_gap, photon_spectrum, v, Tcell) == max(
        v * current_density(E_gap, photon_spectrum, v, Tcell)))
    return max(v * current_density(E_gap, photon_spectrum, v, Tcell))


def max_eff(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> float:
    """
    Calculates the maximum efficiency of a solar cell.

    Parameters: 
        E_gap(float):  Optical Band Gap in eV  
        photon_spectrum(numpy.ndarray): Output spectrum as numpy ndarray.
        Tcell(float): Operating temperature of the cell in K

    Returns:
       Maximum efficiency (float):  Maximum effeciency of the solar cell relative to the total irradiance in %.
    """

    photon_spectrum_1 = photon_spectrum[::-1, 1]
    photon_spectrum_0 = photon_spectrum[::-1, 0]

    irradiance = np.trapezoid(photon_spectrum_1 * q * photon_spectrum_0, photon_spectrum_0)
    return max_power(E_gap, photon_spectrum, Tcell) / irradiance


def fill_factor(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> float:
    """
    Calculates the fill factor of a solar cell.

    Parameters: 
        E_gap(float):  Optical Band Gap in eV  
        photon_spectrum(numpy.ndarray): Output spectrum as numpy ndarray.
        Tcell(float): Operating temperature of the cell in K

    Returns:
       fill_factor (float):  The fill factor of a solar cell.
    """

    j_sc = jsc(E_gap, photon_spectrum)
    v_oc = voc(E_gap, photon_spectrum, Tcell)
    v_mpp = v_at_mpp(E_gap, photon_spectrum)
    j_mpp = j_at_mpp(E_gap, photon_spectrum)

    fill_factor = (j_mpp * v_mpp) / (j_sc, v_oc)

    return fill_factor
