"""Detailed-balance / Shockley-Queisser chain.

Assertions here favour analytic limits and published values over pinned output,
as CONTRIBUTING asks. The anchors are the AM1.5G integrated irradiance
(1000.4 W m^-2 by definition of the standard) and the Shockley-Queisser limit
(33.7 % at 1.34 eV, 300 K).
"""

import numpy as np
import pytest
import scipy.constants as sc

import solphin.db_fom as db_fom

# hc in eV.nm - converting between photon energy and wavelength.
HC_EV_NM = sc.h * sc.c / sc.e * 1e9

SPECTRUM_TYPES = [
    "AM1.5",
    "Fluorescent",
    "Blue LED",
    "Green LED",
    "Red LED",
    "White LED",
    "IR LED",
    "Photopic",
]


# --- loading and unit conversion -------------------------------------------


@pytest.mark.parametrize("spectrum_type", SPECTRUM_TYPES)
def test_load_spectrum_all_types(spectrum_type):
    spectrum = db_fom.load_spectrum(spectrum_type)

    assert spectrum.ndim == 2
    assert spectrum.shape[1] == 2
    assert spectrum.shape[0] > 100
    assert np.all(np.isfinite(spectrum))
    # Wavelength ascending, irradiance non-negative.
    assert np.all(np.diff(spectrum[:, 0]) > 0)
    assert np.all(spectrum[:, 1] >= 0)


def test_load_spectrum_unknown_falls_back_to_am15(am15):
    """An unrecognised name is documented to fall back to AM1.5 rather than raise."""
    fallback = db_fom.load_spectrum("not a real spectrum")

    np.testing.assert_array_equal(fallback, am15)


def test_am15_integrated_irradiance(am15):
    """AM1.5G integrates to 1000.4 W m^-2 - the defining property of the standard."""
    irradiance = np.trapezoid(am15[:, 1], am15[:, 0])

    assert irradiance == pytest.approx(1000.4, rel=1e-3)


def test_convert_spectrum_does_not_mutate_input(am15):
    before = am15.copy()

    db_fom.convert_spectrum(am15)

    np.testing.assert_array_equal(am15, before)


def test_convert_spectrum_energy_wavelength_relation(am15, photon_spectrum):
    """Every row's energy must be hc/lambda for that row's wavelength."""
    expected_eV = HC_EV_NM / am15[:, 0]

    np.testing.assert_allclose(photon_spectrum[:, 0], expected_eV, rtol=1e-12)


def test_convert_spectrum_conserves_power(am15, photon_spectrum):
    """Changing variable from nm to eV must not create or destroy power.

    integral(irradiance dlambda) == integral(E * photon_flux dE). This is the
    test that catches a dropped Jacobian or a unit slip in convert_spectrum.
    """
    power_wavelength = np.trapezoid(am15[:, 1], am15[:, 0])

    # convert_spectrum returns descending energy; reverse for integration.
    energy_eV = photon_spectrum[::-1, 0]
    flux = photon_spectrum[::-1, 1]
    power_energy = np.trapezoid(flux * sc.e * energy_eV, energy_eV)

    assert power_energy == pytest.approx(power_wavelength, rel=1e-6)


# --- photon counting and radiative recombination ---------------------------


def test_photons_above_bandgap_monotonic_decreasing(photon_spectrum):
    """Raising the gap can only discard photons."""
    gaps = np.arange(0.5, 3.01, 0.25)
    counts = [db_fom._photons_above_bandgap(g, photon_spectrum) for g in gaps]

    assert np.all(np.diff(counts) < 0)
    assert counts[0] > 0


def test_photons_above_bandgap_above_spectrum_max_is_zero(photon_spectrum):
    """No photons exist above the spectrum's highest energy."""
    above_max = photon_spectrum[:, 0].max() + 1.0

    assert db_fom._photons_above_bandgap(above_max, photon_spectrum) == 0.0


def test_rr0_increases_with_temperature(photon_spectrum):
    """A hotter cell emits more; J0 is the blackbody flux at zero QFL splitting."""
    cold = db_fom._rr0(1.34, photon_spectrum, 250.0)
    hot = db_fom._rr0(1.34, photon_spectrum, 400.0)

    assert hot > cold > 0


def test_rr0_decreases_with_bandgap(photon_spectrum):
    """The blackbody tail falls off exponentially, so a wider gap emits less."""
    gaps = [0.8, 1.1, 1.4, 1.7, 2.0]
    rates = [db_fom._rr0(g, photon_spectrum, 300.0) for g in gaps]

    assert np.all(np.diff(rates) < 0)


# --- the diode ------------------------------------------------------------


def test_voc_below_bandgap(photon_spectrum):
    """q*Voc < E_gap for every gap - the thermodynamic ceiling on open-circuit voltage."""
    for gap in np.arange(0.5, 3.01, 0.25):
        voc = db_fom.voc(gap, photon_spectrum, 300.0)

        assert 0 < voc < gap


def test_jsc_equals_q_times_photon_flux(photon_spectrum):
    """At short circuit every above-gap photon contributes one electron."""
    expected = sc.e * db_fom._photons_above_bandgap(1.34, photon_spectrum)

    assert db_fom.jsc(1.34, photon_spectrum, 300.0) == pytest.approx(expected, rel=1e-12)


def test_current_density_broadcasts_over_voltage(photon_spectrum):
    voltages = np.linspace(0.0, 0.8, 17)

    current = db_fom.current_density(1.34, photon_spectrum, voltages, 300.0)

    assert isinstance(current, np.ndarray)
    assert current.shape == voltages.shape


def test_current_density_non_increasing_in_voltage(photon_spectrum):
    """Forward bias only ever removes current.

    The dark term is negligible until V approaches Voc, so most consecutive
    differences are zero at float64 precision - hence non-increasing overall,
    strictly decreasing near Voc.
    """
    v_oc = db_fom.voc(1.34, photon_spectrum, 300.0)
    voltages = np.linspace(0.0, v_oc, 200)
    current = db_fom.current_density(1.34, photon_spectrum, voltages, 300.0)

    deltas = np.diff(current)

    assert np.all(deltas <= 0)
    assert np.all(deltas[int(0.9 * len(deltas)):] < 0)


def test_current_density_vanishes_at_voc(photon_spectrum):
    """Voc is by definition the voltage at which the net current is zero."""
    v_oc = db_fom.voc(1.34, photon_spectrum, 300.0)
    j_sc = db_fom.jsc(1.34, photon_spectrum, 300.0)

    at_voc = db_fom.current_density(1.34, photon_spectrum, v_oc, 300.0)

    assert abs(at_voc) < 1e-6 * j_sc


# --- efficiency -----------------------------------------------------------


def test_max_eff_matches_sq_limit(photon_spectrum):
    """The published single-junction SQ limit is 33.7 % at 1.34 eV under AM1.5G."""
    efficiency = db_fom.max_eff(1.34, photon_spectrum, 300.0)

    assert efficiency == pytest.approx(0.337, abs=0.005)


def test_max_eff_peaks_near_1_34_eV(photon_spectrum):
    """The SQ curve's maximum sits at 1.34 eV; allow a window for the grid."""
    gaps = np.arange(0.6, 2.51, 0.02)
    efficiencies = np.array([db_fom.max_eff(g, photon_spectrum, 300.0) for g in gaps])

    assert 1.1 < gaps[efficiencies.argmax()] < 1.5


def test_max_eff_bounded(photon_spectrum):
    """max_eff returns a fraction, so it must sit strictly inside (0, 1)."""
    for gap in (0.8, 1.34, 2.0, 2.8):
        efficiency = db_fom.max_eff(gap, photon_spectrum, 300.0)

        assert 0.0 < efficiency < 1.0


def test_max_power_between_zero_and_jsc_voc(photon_spectrum):
    """P_max = FF * Jsc * Voc with 0 < FF < 1, so it is bounded by the Jsc-Voc rectangle."""
    p_max = db_fom.max_power(1.34, photon_spectrum, 300.0)
    j_sc = db_fom.jsc(1.34, photon_spectrum, 300.0)
    v_oc = db_fom.voc(1.34, photon_spectrum, 300.0)

    assert 0 < p_max < j_sc * v_oc


# --- entry points that have never been reachable from the tutorial ---------
#
# These four are public API and appear in the Sphinx docs, but nothing calls
# them, so their stale call signatures have gone unnoticed. Each test states the
# behaviour the function is documented to have; strict xfail means fixing the
# source turns the test green and reports the marker as obsolete.


@pytest.mark.xfail(
    strict=True,
    reason="db_fom.py:177 calls _rr0(E_gap, photon_spectrum) without Tcell -> TypeError",
)
def test_recomb_rate_returns_finite_float(photon_spectrum):
    rate = db_fom.recomb_rate(1.34, photon_spectrum, 0.5, 300.0)

    assert np.isfinite(rate)
    assert rate > 0


@pytest.mark.xfail(
    strict=True,
    reason="db_fom.py:248 calls voc(E_gap, photon_spectrum) without Tcell -> TypeError",
)
def test_v_at_mpp_between_zero_and_voc(photon_spectrum):
    v_mpp = db_fom.v_at_mpp(1.34, photon_spectrum)
    v_oc = db_fom.voc(1.34, photon_spectrum, 300.0)

    assert 0 < v_mpp < v_oc


@pytest.mark.xfail(
    strict=True,
    reason="db_fom.py:269 calls max_power(E_gap, photon_spectrum) without Tcell -> TypeError",
)
def test_j_at_mpp_below_jsc(photon_spectrum):
    j_mpp = db_fom.j_at_mpp(1.34, photon_spectrum)
    j_sc = db_fom.jsc(1.34, photon_spectrum, 300.0)

    assert 0 < j_mpp < j_sc


@pytest.mark.xfail(
    strict=True,
    reason=(
        "db_fom.py:326 calls jsc() without Tcell, and db_fom.py:331 then divides "
        "by the tuple (j_sc, v_oc) - two bugs, the second hidden behind the first"
    ),
)
def test_fill_factor_between_zero_and_one(photon_spectrum):
    """A well-behaved single-junction cell near the SQ limit has FF around 0.8-0.9."""
    ff = db_fom.fill_factor(1.34, photon_spectrum, 300.0)

    assert 0.7 < ff < 0.95
