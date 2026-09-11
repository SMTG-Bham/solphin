"""Detailed-balance (Shockley-Queisser) limit efficiency and its constituent quantities."""

from functools import lru_cache
from importlib.resources import files
from typing import overload

import numpy as np
import scipy.constants as sc
from numpy.typing import NDArray

h = sc.h  # Planck's constant (J·s)
c = sc.c  # Speed of light (m/s)
k = sc.k  # Boltzmann constant (J/K)
q = sc.e  # Elementary charge (Coulombs)

LUMINOUS_EFFICACY = 683.0  # lm/W at 555 nm, the definition of the lumen
DEFAULT_TARGET_LUX = 1000.0  # the usual indoor-photovoltaic reporting condition

# The white LED's luminous efficacy of radiation, used to place the (invisible)
# IR LED on the same radiant-power footing as the visible illuminants. See
# load_spectrum for why the IR spectrum cannot be scaled by illuminance.
IR_REFERENCE_EFFICACY = 306.34  # lm/W

# Common wavelength support, in nm, that every loaded spectrum is padded out to.
# The bounds are those of ASTMG173.csv, the widest bundled file. Padding is not
# cosmetic: _rr0 and optics._eta_d integrate the blackbody over the wavelength
# grid the spectrum supplies, so a spectrum that stops at 780 nm silently
# truncates the radiative recombination integral for any band gap below
# 1.59 eV. See _pad_to_common_support.
SPECTRUM_WL_MIN = 280.0  # nm
SPECTRUM_WL_MAX = 4000.0  # nm
SPECTRUM_PAD_STEP = 1.0  # nm

# Filenames of the bundled spectra, keyed by the identifier load_spectrum takes.
# The LED-* entries are the CIE standard LED illuminants of CIE 15:2018.
SPECTRUM_FILES = {
    "AM1.5": "ASTMG173.csv",
    "Fluorescent": "fluorescent.csv",
    "Blue LED": "led_blue.csv",
    "Green LED": "led_green.csv",
    "Red LED": "led_red.csv",
    "White LED": "led_white.csv",
    "IR LED": "led_ir.csv",
    "LED-B1": "LED-B1.csv",
    "LED-B2": "LED-B2.csv",
    "LED-B3": "LED-B3.csv",
    "LED-B4": "LED-B4.csv",
    "LED-B5": "LED-B5.csv",
    "LED-BH1": "LED-BH1.csv",
    "LED-RGB1": "LED-RGB1.csv",
    "LED-V1": "LED-V1.csv",
    "LED-V2": "LED-V2.csv",
}

# The subset of SPECTRUM_FILES that are CIE 15:2018 standard LED illuminants.
# They share a 380-780 nm support and an arbitrary scale, and behave alike
# downstream, so callers that need to treat the family as a group use this.
CIE_LED_SPECTRA = tuple(name for name in SPECTRUM_FILES if name.startswith("LED-"))


# Convert the spectrum to the useful units - taken from https://github.com/kaklin/sq-limit?tab=readme-ov-file

@lru_cache(maxsize=1)
def _load_photopic_curve() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Load the photopic luminosity function V(λ) from bundled resources.

    Returned as tuples rather than arrays so the result can be cached: the
    curve is a fixed data file, read once and reused by every illuminance
    calculation.

    Returns
    -------
    wavelengths : tuple of float
        Wavelengths in nm, ascending over 300-900 nm.
    response : tuple of float
        Dimensionless photopic response, peaking at 1.0 near 555 nm.
    """
    csv_path = files("solphin.resources") / "photopic.csv"

    with csv_path.open("r", encoding="utf-8") as f:
        curve = np.loadtxt(f, delimiter=",", skiprows=1)

    return tuple(curve[:, 0]), tuple(curve[:, 1])


def calculate_illuminance(spectrum: NDArray) -> float:
    r"""Calculate the illuminance of a spectral irradiance distribution.

    The illuminance is the irradiance weighted by the eye's photopic response,

    .. math:: E_v = 683 \int V(\lambda) E(\lambda) \, d\lambda,

    with V(λ) the photopic luminosity function bundled as ``photopic.csv``.
    Wavelengths outside the 300-900 nm range of that curve contribute nothing,
    which is correct: the eye does not respond there.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectrum as loaded by ``load_spectrum``: wavelength in nm against
        spectral irradiance in W m⁻² nm⁻¹.

    Returns
    -------
    float
        Illuminance in lux.
    """
    wavelengths = np.asarray(spectrum[:, 0], dtype=float)
    irradiance = np.asarray(spectrum[:, 1], dtype=float)

    curve_wavelengths, curve_response = _load_photopic_curve()
    response = np.interp(wavelengths, curve_wavelengths, curve_response, left=0.0, right=0.0)

    return float(LUMINOUS_EFFICACY * np.trapezoid(response * irradiance, wavelengths))


def normalise_spectrum(
        spectrum: NDArray,
        target_lux: float | None = None,
        target_irradiance: float | None = None,
) -> NDArray:
    """Rescale a spectrum to a target illuminance or a target irradiance.

    Only the magnitude changes; the spectral shape is preserved. Exactly one
    target must be given, since the two fix the scale in incompatible ways.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectrum as loaded by ``load_spectrum``: wavelength in nm against
        spectral irradiance in W m⁻² nm⁻¹.
    target_lux : float or None, optional
        Illuminance to scale to, in lux. Default is ``None``.
    target_irradiance : float or None, optional
        Total irradiance to scale to, in W m⁻². Default is ``None``.

    Returns
    -------
    numpy.ndarray
        A copy of the spectrum with column 1 rescaled.

    Raises
    ------
    ValueError
        If both targets or neither is given, if a target is not positive, or
        if the spectrum's current illuminance (or irradiance) is zero, leaving
        the scale factor undefined.
    """
    ONE_TARGET = (
        "give exactly one of target_lux or target_irradiance: an illuminance "
        "and an irradiance fix the scale differently, so a spectrum cannot "
        "satisfy both at once."
    )

    normalised = np.array(spectrum, dtype=float, copy=True)

    # Branching on each argument in turn rather than on a combined condition,
    # so that `target` is known to be a float in both live branches.
    if target_lux is not None and target_irradiance is not None:
        raise ValueError(ONE_TARGET)

    if target_lux is not None:
        target, basis = target_lux, "illuminance"
        current = calculate_illuminance(normalised)
    elif target_irradiance is not None:
        target, basis = target_irradiance, "irradiance"
        current = float(np.trapezoid(normalised[:, 1], normalised[:, 0]))
    else:
        raise ValueError(ONE_TARGET)

    if target <= 0:
        raise ValueError(f"target {basis} must be positive, got {target}.")

    if current <= 0:
        raise ValueError(
            f"the spectrum's {basis} is {current:g}, so it cannot be scaled to "
            f"{target:g}. A spectrum with no photopic overlap (an infrared "
            "source, say) has no meaningful illuminance; normalise it with "
            "target_irradiance instead."
        )

    normalised[:, 1] *= target / current

    return normalised


def _pad_to_common_support(spectrum: NDArray) -> NDArray:
    """Extend a spectrum to the common wavelength support with zero irradiance.

    The bundled files cover very different ranges — 280-4000 nm for AM1.5G,
    380-780 nm for the CIE LED illuminants — and that range is not merely a
    plotting detail. Both :func:`_rr0` and ``optics._eta_d`` integrate the
    blackbody spectrum over whatever wavelength grid the illumination spectrum
    supplies, because the two integrals share a grid. A spectrum that stops at
    780 nm therefore truncates the radiative recombination integral at 1.59 eV,
    and for any band gap below that the recombination current comes back orders
    of magnitude too small and the efficiency correspondingly too high.

    Padding with zeros fixes this without changing any measured quantity: a
    region of zero irradiance contributes nothing to the illuminance, the
    incident power or the short-circuit current, but it does give the blackbody
    integral the grid it needs. Points are added only outside the spectrum's
    own range, so a file that already spans the full support is returned
    unchanged.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectrum with wavelength in nm in column 0 and spectral irradiance in
        W m⁻² nm⁻¹ in column 1, ascending in wavelength.

    Returns
    -------
    numpy.ndarray
        The spectrum extended to cover ``SPECTRUM_WL_MIN`` to
        ``SPECTRUM_WL_MAX``, still ascending in wavelength.
    """
    wavelengths = spectrum[:, 0]

    blocks = []

    if wavelengths[0] > SPECTRUM_WL_MIN:
        below = np.arange(wavelengths[0] - SPECTRUM_PAD_STEP, SPECTRUM_WL_MIN - SPECTRUM_PAD_STEP,
                          -SPECTRUM_PAD_STEP)[::-1]
        below = below[below >= SPECTRUM_WL_MIN]
        blocks.append(np.column_stack([below, np.zeros_like(below)]))

    blocks.append(spectrum)

    if wavelengths[-1] < SPECTRUM_WL_MAX:
        above = np.arange(wavelengths[-1] + SPECTRUM_PAD_STEP,
                          SPECTRUM_WL_MAX + SPECTRUM_PAD_STEP, SPECTRUM_PAD_STEP)
        above = above[above <= SPECTRUM_WL_MAX]
        blocks.append(np.column_stack([above, np.zeros_like(above)]))

    if len(blocks) == 1:
        return spectrum

    return np.vstack(blocks)


def load_spectrum(spectrum_type: str, target_lux: float | None = DEFAULT_TARGET_LUX) -> NDArray:
    """Load a predefined spectral irradiance dataset from bundled resources.

    The bundled illuminant files are not on a common scale as supplied: the red,
    white and IR LED spectra are normalised to a peak of 1.0, while the
    fluorescent, blue and green LED spectra are absolute measurements taken at
    roughly 3000, 10 and 100 lux respectively. Integrated as they stand they
    span 0.1 to 213 W m⁻², so efficiencies computed from them are neither
    physical nor comparable with one another. They are therefore treated as
    relative spectra and rescaled to ``target_lux`` using
    :func:`calculate_illuminance`. The CIE standard LED illuminants
    (``"LED-B1"`` and the rest) are likewise supplied on an arbitrary scale and
    are normalised the same way.

    Two spectra are handled differently:

    - **AM1.5** is an absolute standard integrating to 1000 W m⁻², and is never
      rescaled. ``target_lux`` is ignored for it.
    - **IR LED** peaks at 849 nm, where the photopic response is essentially
      zero, giving it an illuminance of ~0.009 lux; scaling that to
      ``target_lux`` would inflate it by five orders of magnitude. It is
      instead matched on radiant power, to the irradiance that the white LED
      carries at the same illuminance (``target_lux / IR_REFERENCE_EFFICACY``).
      Comparisons involving the IR LED are therefore power-matched rather than
      illuminance-matched.

    Every returned spectrum is padded with zero irradiance to a common
    280-4000 nm support by :func:`_pad_to_common_support`, which the
    detailed-balance integrals depend on; see that function.

    Parameters
    ----------
    spectrum_type : str
        Identifier for the spectrum, one of the keys of
        :data:`SPECTRUM_FILES`: ``"AM1.5"``, ``"Fluorescent"``,
        ``"Blue LED"``, ``"Green LED"``, ``"Red LED"``, ``"White LED"``,
        ``"IR LED"``, or one of the CIE standard LED illuminants
        ``"LED-B1"`` to ``"LED-B5"``, ``"LED-BH1"``, ``"LED-RGB1"``,
        ``"LED-V1"`` and ``"LED-V2"``. Unrecognised values fall back to
        ``"AM1.5"``.
    target_lux : float or None, optional
        Illuminance to rescale the spectrum to, in lux. ``None`` returns the
        file exactly as stored — original arbitrary scale, original wavelength
        range, unpadded — as an escape hatch for inspecting the raw data.
        Default is ``DEFAULT_TARGET_LUX``, 1000 lux.

    Returns
    -------
    numpy.ndarray
        2D array; column 0 is wavelength in nm, column 1 spectral irradiance
        in W m⁻² nm⁻¹.
    """
    if spectrum_type not in SPECTRUM_FILES:
        print("Unrecognisable spectrum selected")
        print("Options: " + ", ".join(SPECTRUM_FILES))
        print("reverting to AM1.5")

        spectrum_type = "AM1.5"

    csv_path = files("solphin.resources") / SPECTRUM_FILES[spectrum_type]

    with csv_path.open("r", encoding="utf-8") as f:
        spectrum = np.loadtxt(f, delimiter=",", skiprows=1)

    if target_lux is None:
        return spectrum

    if spectrum_type == "AM1.5":
        # An absolute standard, and already spanning the full support.
        return _pad_to_common_support(spectrum)

    if spectrum_type == "IR LED":
        # Invisible, so matched on radiant power rather than illuminance.
        scaled = normalise_spectrum(spectrum, target_irradiance=target_lux / IR_REFERENCE_EFFICACY)
    else:
        scaled = normalise_spectrum(spectrum, target_lux=target_lux)

    return _pad_to_common_support(scaled)


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
    converted[:, 0] = converted[:, 0] * sc.nano  # wavelength to m
    converted[:, 1] = converted[:, 1] / sc.nano  # irradiance to W/m2/m (from W/m2/nm)

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
        Integrated photon flux above the band gap, in photons m⁻² s⁻¹.
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
        Radiative recombination rate per unit area, in photons m⁻² s⁻¹.
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
        Radiative recombination current density in A m⁻².
    """
    print('recomb rate')
    return q * _rr0(E_gap, photon_spectrum, Tcell) * np.exp(q * voltage / (k * Tcell))


def _diode_terms(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> tuple[float, float]:
    """Return the two fluxes every diode quantity here is built from.

    Both are tail integrals over the spectrum's grid, and both are the
    expensive part of this module: computing them together means a caller that
    needs the full diode characteristic pays for them once rather than once per
    derived quantity.

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
    Jph : float
        Above-band-gap photon flux, in photons m⁻² s⁻¹.
    J0 : float
        Radiative recombination flux at zero bias, in photons m⁻² s⁻¹.
    """
    return (_photons_above_bandgap(E_gap, photon_spectrum),
            _rr0(E_gap, photon_spectrum, Tcell))


def _voc_from_terms(Jph: float, J0: float, Tcell: float) -> float:
    """Open-circuit voltage from the two diode fluxes.

    Kept separate from :func:`voc` so that the formula has one home while
    callers that already hold ``Jph`` and ``J0`` need not recompute them.

    Parameters
    ----------
    Jph : float
        Above-band-gap photon flux, in photons m⁻² s⁻¹.
    J0 : float
        Radiative recombination flux at zero bias, in photons m⁻² s⁻¹.
    Tcell : float
        Operating temperature of the cell in K.

    Returns
    -------
    float
        Open-circuit voltage in V.
    """
    return (k * Tcell / q) * np.log(Jph / J0 + 1)


def _power_curve(
        E_gap: float, photon_spectrum: NDArray, Tcell: float, n_points: int = 50
) -> tuple[NDArray, NDArray]:
    """Sweep the diode from short circuit to open circuit, once.

    Every maximum-power-point quantity is a different reduction of this one
    sweep, so they share it rather than each rebuilding it.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    Tcell : float
        Operating temperature of the cell in K.
    n_points : int, optional
        Number of voltage samples. Default is ``50``, matching the
        ``numpy.linspace`` default this replaced; the maximum power point is
        located by discrete search, so this sets its resolution.

    Returns
    -------
    voltage : numpy.ndarray
        Voltage samples in V, from 0 to the open-circuit voltage.
    current : numpy.ndarray
        Current density at each voltage, in A m⁻².
    """
    Jph, J0 = _diode_terms(E_gap, photon_spectrum, Tcell)
    voltage = np.linspace(0, _voc_from_terms(Jph, J0, Tcell), n_points)
    current = q * (Jph - J0 * (np.exp(q * voltage / (k * Tcell)) - 1))

    return voltage, current


def incident_power(photon_spectrum: NDArray) -> float:
    """Integrate a photon-flux spectrum back to an incident power density.

    Independent of the band gap, so a sweep over gaps should compute it once
    and pass it to :func:`max_eff` rather than let every call repeat it.

    Parameters
    ----------
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.

    Returns
    -------
    float
        Total incident power density in W m⁻².
    """
    energy = photon_spectrum[::-1, 0]
    flux = photon_spectrum[::-1, 1]

    return float(np.trapezoid(flux * q * energy, energy))


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
        Current density in A m⁻². Scalar for scalar ``voltage``,
        elementwise array otherwise.
    """
    Jph, J0 = _diode_terms(E_gap, photon_spectrum, Tcell)

    return q * (Jph - J0 * (np.exp(q * voltage / (k * Tcell)) - 1))


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
        Current density at zero applied voltage in A m⁻².
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
    Jph, J0 = _diode_terms(E_gap, photon_spectrum, Tcell)

    return _voc_from_terms(Jph, J0, Tcell)


# Tcell is optional on v_at_mpp and j_at_mpp only: both are documented and
# called as two-argument functions, and 300 K is the standard-condition
# temperature the rest of this module's anchors (SQ limit, Voc tests) assume.
def v_at_mpp(E_gap: float, photon_spectrum: NDArray, Tcell: float = 300.0) -> float:
    """Calculate the voltage at the maximum power point.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    Tcell : float, optional
        Operating temperature of the cell in K. Default is ``300.0``.

    Returns
    -------
    float
        Voltage at the maximum power point in V.
    """
    voltage, current = _power_curve(E_gap, photon_spectrum, Tcell)

    return float(voltage[np.argmax(voltage * current)])


def j_at_mpp(E_gap: float, photon_spectrum: NDArray, Tcell: float = 300.0) -> float:
    """Calculate the current density at the maximum power point.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    Tcell : float, optional
        Operating temperature of the cell in K. Default is ``300.0``.

    Returns
    -------
    float
        Current density at the maximum power point in A m⁻².
    """
    voltage, current = _power_curve(E_gap, photon_spectrum, Tcell)

    return float(current[np.argmax(voltage * current)])


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
        Maximum areal power density of the cell in W m⁻².
    """
    voltage, current = _power_curve(E_gap, photon_spectrum, Tcell)

    return float(np.max(voltage * current))


def max_eff(
        E_gap: float, photon_spectrum: NDArray, Tcell: float, power_in: float | None = None
) -> float:
    """Calculate the maximum efficiency of a solar cell.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    Tcell : float
        Operating temperature of the cell in K.
    power_in : float or None, optional
        Incident power density in W m⁻², as returned by
        :func:`incident_power`. It does not depend on the band gap, so a sweep
        over gaps should compute it once and pass it here. Default is
        ``None``, which computes it from ``photon_spectrum``.

    Returns
    -------
    float
        Maximum efficiency of the cell relative to the total irradiance,
        as a dimensionless fraction.
    """
    if power_in is None:
        power_in = incident_power(photon_spectrum)

    return max_power(E_gap, photon_spectrum, Tcell) / power_in


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
    j_sc = jsc(E_gap, photon_spectrum, Tcell)
    v_oc = voc(E_gap, photon_spectrum, Tcell)
    v_mpp = v_at_mpp(E_gap, photon_spectrum, Tcell)
    j_mpp = j_at_mpp(E_gap, photon_spectrum, Tcell)

    fill_factor = (j_mpp * v_mpp) / (j_sc * v_oc)

    return fill_factor
