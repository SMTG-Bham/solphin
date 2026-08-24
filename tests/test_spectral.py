"""Irradiance-weighted absorption descriptors.

The two headline functions have exact analytic limits - a constant absorption
coefficient must come back unchanged from the weighted average, and must give
exactly zero dispersion - which makes them checkable without pinning numbers.
"""

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


# --- truncation ------------------------------------------------------------


def test_truncate_abs_spectra_respects_bounds() -> None:
    """Truncated absorption wavelengths all fall between 300 nm and the gap wavelength."""
    energies_eV = np.linspace(0.3, 5.0, 400)
    coefficients = np.linspace(1.0, 1e5, 400)

    wavelengths, coeffs = spectral._truncate_abs_spectra(1.5, energies_eV, coefficients)

    _, gap_wavelength = spectral._extract_int_limits(1.5)
    assert len(wavelengths) == len(coeffs) > 0
    assert all(300 <= wl <= gap_wavelength for wl in wavelengths)


def test_truncate_light_spectra_respects_bounds(am15: NDArray) -> None:
    """Truncated spectrum wavelengths all fall between 300 nm and the gap wavelength."""
    wavelengths, irradiance = spectral._truncate_light_spectra(am15, 1.5)

    _, gap_wavelength = spectral._extract_int_limits(1.5)
    assert len(wavelengths) == len(irradiance) > 0
    assert all(300 <= wl <= gap_wavelength for wl in wavelengths)


def test_truncate_light_spectra_does_not_mutate_input(am15: NDArray) -> None:
    """_truncate_light_spectra works on a copy and leaves its input untouched."""
    before = am15.copy()

    spectral._truncate_light_spectra(am15, 1.5)

    np.testing.assert_array_equal(am15, before)


def test_truncate_returns_empty_for_huge_bandgap(am15: NDArray) -> None:
    """A gap so wide that its wavelength falls below 300 nm empties the window."""
    wavelengths, irradiance = spectral._truncate_light_spectra(am15, 100.0)

    assert list(wavelengths) == []
    assert list(irradiance) == []


# --- nearest-neighbour matching -------------------------------------------


def test_match_wavelengths_exact_grid() -> None:
    """Identical grids must map each point onto itself."""
    grid = [400.0, 500.0, 600.0]
    irradiance = [1.0, 2.0, 3.0]

    matched = spectral._match_wavelengths(grid, grid, irradiance)

    assert matched == irradiance


def test_match_wavelengths_averages_ties() -> None:
    """Two equidistant neighbours are averaged, as documented."""
    matched = spectral._match_wavelengths([500.0], [400.0, 600.0], [1.0, 3.0])

    assert matched == [2.0]


def test_match_wavelengths_picks_nearest() -> None:
    """An off-grid wavelength takes the irradiance of its nearest neighbour."""
    matched = spectral._match_wavelengths([505.0], [400.0, 500.0, 700.0], [1.0, 2.0, 3.0])

    assert matched == [2.0]


# --- the descriptors ------------------------------------------------------


def test_spectral_average_of_constant_is_that_constant() -> None:
    """A weighted mean of a constant is that constant, whatever the weights."""
    wavelengths = np.linspace(300.0, 800.0, 51)
    constant_alpha = np.full(51, 7.0)
    irradiance = np.linspace(1.0, 2.0, 51)

    average = spectral.calculate_spectral_average(
        constant_alpha, irradiance, wavelengths
    )

    assert average == pytest.approx(7.0, rel=1e-12)


def test_spectral_dispersion_of_constant_is_zero() -> None:
    """Dispersion measures the spread of log(alpha); a constant has none."""
    wavelengths = np.linspace(300.0, 800.0, 51)
    constant_alpha = np.full(51, 7.0)
    irradiance = np.linspace(1.0, 2.0, 51)

    dispersion = spectral.calculate_spectral_dispersion(
        constant_alpha, irradiance, wavelengths
    )

    assert dispersion == pytest.approx(0.0, abs=1e-12)


def test_spectral_dispersion_increases_with_spread() -> None:
    """A wider spread of log(alpha) must give a larger dispersion."""
    wavelengths = np.linspace(300.0, 800.0, 51)
    irradiance = np.ones(51)
    centre = np.log(1e4)

    dispersions = [
        spectral.calculate_spectral_dispersion(
            np.exp(centre + width * np.linspace(-1.0, 1.0, 51)), irradiance, wavelengths
        )
        for width in (0.1, 0.5, 1.0, 2.0)
    ]

    assert np.all(np.diff(dispersions) > 0)


def test_spectral_average_ignores_non_positive_alpha() -> None:
    """calculate_spectral_dispersion filters alpha <= 0 before taking a log."""
    wavelengths = np.linspace(300.0, 800.0, 5)
    alpha = np.array([0.0, 7.0, 7.0, -1.0, 7.0])
    irradiance = np.ones(5)

    dispersion = spectral.calculate_spectral_dispersion(alpha, irradiance, wavelengths)

    assert dispersion == pytest.approx(0.0, abs=1e-12)


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
    """The absorption descriptors of the reference data."""
    average, dispersion = spectral.generate_spectral_parameters(
        str(opt_dir), am15, E_gap=REFERENCE_E_GAP
    )

    assert average == pytest.approx(128846.05, rel=1e-6)
    assert dispersion == pytest.approx(1.5663474, rel=1e-6)


def test_generate_spectral_parameters_rejects_converted_spectrum(
        opt_dir: Path, photon_spectrum: NDArray
) -> None:
    """It wants wavelength-space input; energy-space input is caught, but obscurely.

    convert_spectrum output spans 0.31-4.43 eV, and none of those values land in
    the 300 nm - lambda_gap window, so _truncate_light_spectra returns nothing
    and _match_wavelengths ends up calling min() on an empty list. The misuse
    does fail rather than returning a plausible wrong number, which is what
    matters - but the error names neither the argument nor the units.
    """
    # The wording is CPython's, and 3.12 changed it: "min() arg is an empty
    # sequence" became "min() iterable argument is empty".
    with pytest.raises(ValueError, match="empty sequence|iterable argument is empty"):
        spectral.generate_spectral_parameters(
            str(opt_dir), photon_spectrum, E_gap=REFERENCE_E_GAP
        )
