"""Photon-flux-weighted absorption descriptors.

The two headline functions implement equations (1)-(2) of Crovetto 2024
(J. Phys. Energy 6 025009) and have exact analytic limits - a constant
absorption coefficient must come back unchanged from the weighted average,
and must give exactly zero dispersion - which makes them checkable without
pinning numbers. The conventions the paper leaves implicit (photon-flux
weights, base-10 logarithm, centring on the linear mean) are pinned by the
sigma range its supplementary material states for power-law onset spectra.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.constants as sc
from numpy.typing import NDArray

import solphin.spectral as spectral

HC_EV_NM = sc.h * sc.c / sc.e * 1e9

# The direct gap of the committed band structure in tests/data.
REFERENCE_E_GAP = 1.3900999999999994


# --- unit conversions and integration limits -------------------------------


def test_wavelength_conv_roundtrip() -> None:
    """E * lambda = hc = 1239.84 eV.nm for every point."""
    energies_eV = np.array([0.5, 1.0, 1.3901, 2.5, 4.0])

    wavelengths_nm = spectral._wavelength_conv(energies_eV)

    np.testing.assert_allclose(energies_eV * wavelengths_nm, HC_EV_NM, rtol=1e-12)


def test_extract_int_limits() -> None:
    """Lower limit is fixed at 300 nm; upper is the band-gap wavelength."""
    wavelength_min, gap_wavelength = spectral._extract_int_limits(1.0)

    assert wavelength_min == 300
    assert gap_wavelength == pytest.approx(HC_EV_NM, rel=1e-12)


# --- resampling onto the common wavelength grid ----------------------------


def test_resample_grid_spans_integration_window(am15: NDArray) -> None:
    """The common grid runs ascending from 300 nm to the band-gap wavelength."""
    energies_eV = np.linspace(0.3, 5.0, 400)
    coefficients = np.linspace(1.0, 1e5, 400)

    wavelengths, alpha, flux = spectral._resample_common_grid(
        energies_eV, coefficients, am15, E_gap=1.5, num_points=501
    )

    _, gap_wavelength = spectral._extract_int_limits(1.5)
    assert len(wavelengths) == len(alpha) == len(flux) == 501
    assert wavelengths[0] == 300
    assert wavelengths[-1] == pytest.approx(gap_wavelength, rel=1e-12)
    assert np.all(np.diff(wavelengths) > 0)


def test_resample_converts_irradiance_to_photon_flux() -> None:
    """A flat 1 W m-2 nm-1 spectrum gives phi(lambda) = lambda_m / (h c) exactly."""
    energies_eV = np.linspace(0.5, 5.0, 100)
    coefficients = np.ones(100)
    flat_spectrum = np.column_stack(
        (np.linspace(250.0, 1500.0, 200), np.ones(200))
    )

    wavelengths, _, flux = spectral._resample_common_grid(
        energies_eV, coefficients, flat_spectrum, E_gap=1.5, num_points=101
    )

    expected = wavelengths * sc.nano / (sc.h * sc.c)
    np.testing.assert_allclose(flux, expected, rtol=1e-12)


def test_resample_rejects_huge_bandgap(am15: NDArray) -> None:
    """A gap so wide that its wavelength falls below 300 nm leaves no window."""
    energies_eV = np.linspace(0.3, 5.0, 400)
    coefficients = np.ones(400)

    with pytest.raises(ValueError, match="no.*spectral window|no spectral window"):
        spectral._resample_common_grid(energies_eV, coefficients, am15, E_gap=100.0)


def test_resample_rejects_energy_space_spectrum(photon_spectrum: NDArray) -> None:
    """An energy-space spectrum is refused by name rather than failing obscurely."""
    energies_eV = np.linspace(0.3, 5.0, 400)
    coefficients = np.ones(400)

    with pytest.raises(ValueError, match="wavelength space"):
        spectral._resample_common_grid(
            energies_eV, coefficients, photon_spectrum, E_gap=1.5
        )


# --- the descriptors ------------------------------------------------------


def test_spectral_average_of_constant_is_that_constant() -> None:
    """A weighted mean of a constant is that constant, whatever the weights."""
    wavelengths = np.linspace(300.0, 800.0, 51)
    constant_alpha = np.full(51, 7.0)
    flux = np.linspace(1.0, 2.0, 51)

    average = spectral.calculate_spectral_average(constant_alpha, flux, wavelengths)

    assert average == pytest.approx(7.0, rel=1e-12)


def test_spectral_dispersion_of_constant_is_zero() -> None:
    """Dispersion measures the spread of log(alpha); a constant has none."""
    wavelengths = np.linspace(300.0, 800.0, 51)
    constant_alpha = np.full(51, 7.0)
    flux = np.linspace(1.0, 2.0, 51)

    dispersion = spectral.calculate_spectral_dispersion(constant_alpha, flux, wavelengths)

    assert dispersion == pytest.approx(0.0, abs=1e-12)


def test_spectral_dispersion_increases_with_spread() -> None:
    """A wider spread of log(alpha) must give a larger dispersion."""
    wavelengths = np.linspace(300.0, 800.0, 51)
    flux = np.ones(51)
    centre = np.log(1e4)

    dispersions = [
        spectral.calculate_spectral_dispersion(
            np.exp(centre + width * np.linspace(-1.0, 1.0, 51)), flux, wavelengths
        )
        for width in (0.1, 0.5, 1.0, 2.0)
    ]

    assert np.all(np.diff(dispersions) > 0)


def test_spectral_dispersion_ignores_non_positive_alpha() -> None:
    """calculate_spectral_dispersion filters alpha <= 0 before taking a log."""
    wavelengths = np.linspace(300.0, 800.0, 5)
    alpha = np.array([0.0, 7.0, 7.0, -1.0, 7.0])
    flux = np.ones(5)

    dispersion = spectral.calculate_spectral_dispersion(alpha, flux, wavelengths)

    assert dispersion == pytest.approx(0.0, abs=1e-12)


def test_spectral_dispersion_needs_two_positive_points() -> None:
    """All-zero absorption leaves log(alpha) undefined, and says so."""
    wavelengths = np.linspace(300.0, 800.0, 5)
    alpha = np.zeros(5)
    flux = np.ones(5)

    with pytest.raises(ValueError, match="positive absorption"):
        spectral.calculate_spectral_dispersion(alpha, flux, wavelengths)


def test_dispersion_matches_paper_s10_anchors(am15: NDArray) -> None:
    """Power-law onsets reproduce the sigma range of Crovetto's Eq. S10 spectra.

    The supplementary material builds alpha(E) from (E-Eg)^1/2, (E-Eg)^2 and
    (E-Eg)^5/2 terms and states the resulting mixtures span 0.29 < sigma < 1.42
    for Eg = 1.2 eV under AM1.5G. The pure 1/2 and 5/2 endpoints must bracket
    that range from outside, which pins every convention at once: photon-flux
    weights, the base-10 logarithm, centring on the log of the linear
    equation-(1) mean, and integration over wavelength.
    """
    E_gap = 1.2
    energies_eV = np.linspace(0.3, 4.5, 2000)

    sigmas = {}
    for exponent in (0.5, 2.5):
        alpha_E = 1e5 * np.where(
            energies_eV > E_gap, np.clip(energies_eV - E_gap, 0.0, None) ** exponent, 0.0
        )
        wavelengths, alpha, flux = spectral._resample_common_grid(
            energies_eV, alpha_E, am15, E_gap
        )
        sigmas[exponent] = spectral.calculate_spectral_dispersion(alpha, flux, wavelengths)

    assert sigmas[0.5] == pytest.approx(0.263, abs=0.01)
    assert sigmas[2.5] == pytest.approx(1.521, abs=0.01)
    assert sigmas[0.5] < 0.29 < 1.42 < sigmas[2.5]


# --- the file-backed orchestrator -----------------------------------------


def test_load_absorption_drops_the_zero_energy_row(opt_dir: Path) -> None:
    """skiprows=2 discards the E = 0 point along with the single header line.

    absorption.dat carries one '#' header and 2000 data rows, so 1999 survive.
    Dropping E = 0 looks deliberate - _wavelength_conv would divide by zero
    there - so this pins the behaviour rather than treating it as a defect.
    """
    energies_eV, coefficients = spectral._load_absorption(opt_dir / "absorption.dat")

    assert len(energies_eV) == len(coefficients) == 1999
    assert np.all(energies_eV > 0)


def test_generate_spectral_parameters_reference(opt_dir: Path, am15: NDArray) -> None:
    """The absorption descriptors of the reference data, per the paper's definition.

    Cross-checked against an independent implementation of equations (1)-(2)
    on a 20001-point grid, which gave 67574.2 cm-1 and 1.1807; the 0.002
    difference in sigma is the 52 alpha = 0 grid points at the gap edge,
    which that implementation kept in the normalising integral and this one
    excludes consistently.
    """
    average, dispersion = spectral.generate_spectral_parameters(
        str(opt_dir), am15, E_gap=REFERENCE_E_GAP
    )

    assert average == pytest.approx(67574.168, rel=1e-6)
    assert dispersion == pytest.approx(1.1827393, rel=1e-6)


def test_generate_spectral_parameters_rejects_converted_spectrum(
        opt_dir: Path, photon_spectrum: NDArray
) -> None:
    """It wants wavelength-space input; energy-space input is refused by name.

    convert_spectrum output spans 0.31-4.43 eV, and none of those values land
    in the 300 nm - lambda_gap window, so the resampler raises an error naming
    the expected units and the likely misuse.
    """
    with pytest.raises(ValueError, match="wavelength space"):
        spectral.generate_spectral_parameters(
            str(opt_dir), photon_spectrum, E_gap=REFERENCE_E_GAP
        )


# --- the table 1 range of the descriptors this module produces -------------
#
# alpha and sigma are measurements of the supplied absorption data, not user
# choices, so they are always returned. The warning exists so that a weak
# absorber or a slow absorption onset is reported where it arises, rather than
# surfacing two cells later as a range error on a number the caller never
# picked.


def test_generate_spectral_parameters_is_quiet_for_the_reference_data(
        opt_dir: Path, am15: NDArray
) -> None:
    """The committed reference absorption sits inside table 1, so nothing is said."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        spectral.generate_spectral_parameters(
            str(opt_dir), am15, E_gap=REFERENCE_E_GAP
        )


def test_warns_for_an_absorption_average_below_the_sampled_range() -> None:
    """A weak absorber is a real material, so it warns rather than raising."""
    with pytest.warns(UserWarning, match="Spectral average"):
        spectral._warn_outside_sampled_range(1.2e3, 1.0)


def test_warns_for_a_dispersion_above_the_sampled_range() -> None:
    """A slow absorption onset pushes sigma past 1.8."""
    with pytest.warns(UserWarning, match="Spectral dispersion"):
        spectral._warn_outside_sampled_range(1e5, 2.4)


def test_warning_names_the_bound_and_the_way_out() -> None:
    """The message has to explain itself without the paper to hand."""
    with pytest.warns(UserWarning) as record:
        spectral._warn_outside_sampled_range(1.2e3, 1.0)

    message = str(record[0].message)

    assert "5e+03" in message
    assert "cm⁻¹" in message
    assert "allow_out_of_range=True" in message


def test_no_warning_at_either_endpoint() -> None:
    """The sampled ranges are closed intervals, here as in pv_fom."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        spectral._warn_outside_sampled_range(5e3, 0.2)
        spectral._warn_outside_sampled_range(5e5, 1.8)
