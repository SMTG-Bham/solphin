"""Detailed-balance / Shockley-Queisser chain.

Assertions here favour analytic limits and published values over pinned output,
as CONTRIBUTING asks. The anchors are the AM1.5G integrated irradiance
(1000.4 W m^-2 by definition of the standard) and the Shockley-Queisser limit
(33.7 % at 1.34 eV, 300 K).
"""

import numpy as np
import pytest
import scipy.constants as sc
from numpy.typing import NDArray

import solphin.db_fom as db_fom

# hc in eV.nm - converting between photon energy and wavelength.
HC_EV_NM = sc.h * sc.c / sc.e * 1e9

# Every selectable spectrum, taken from the loader so a newly bundled file
# cannot be added without the checks below covering it.
SPECTRUM_TYPES = list(db_fom.SPECTRUM_FILES)

# The illuminants that can be placed on a matched-illuminance footing. The IR
# LED is excluded: it emits at 849 nm, where the photopic response is ~0, so it
# is power-matched instead - see load_spectrum.
VISIBLE_ILLUMINANTS = [
    name for name in SPECTRUM_TYPES if name not in ("AM1.5", "IR LED")
]

# Irradiance each bundled illuminant carries once normalised to 1000 lux, in
# W m^-2. These follow from the spectral shapes and the photopic curve, and are
# the numbers a downstream efficiency is computed from.
IRRADIANCE_AT_1000_LUX = {
    "Fluorescent": 2.8096,
    "Blue LED": 10.5786,
    "Green LED": 1.5307,
    "Red LED": 4.0349,
    "White LED": 3.2643,
    "IR LED": 3.2643,  # power-matched to the white LED, not illuminance-matched
    # The CIE standard LED illuminants, whose luminous efficacies of radiation
    # (249-363 lm/W) put them all within a watt or so of one another.
    "LED-B1": 3.0467,
    "LED-B2": 3.0297,
    "LED-B3": 3.0043,
    "LED-B4": 2.9886,
    "LED-B5": 3.1174,
    "LED-BH1": 2.7559,
    "LED-RGB1": 3.2552,
    "LED-V1": 4.0190,
    "LED-V2": 3.8448,
}


# --- loading and unit conversion -------------------------------------------


@pytest.mark.parametrize("spectrum_type", SPECTRUM_TYPES)
def test_load_spectrum_all_types(spectrum_type: str) -> None:
    """Every bundled spectrum loads as a finite two-column array with ascending wavelength."""
    spectrum = db_fom.load_spectrum(spectrum_type)

    assert spectrum.ndim == 2
    assert spectrum.shape[1] == 2
    assert spectrum.shape[0] > 100
    assert np.all(np.isfinite(spectrum))
    # Wavelength ascending, irradiance non-negative.
    assert np.all(np.diff(spectrum[:, 0]) > 0)
    assert np.all(spectrum[:, 1] >= 0)


def test_load_spectrum_unknown_falls_back_to_am15(am15: NDArray) -> None:
    """An unrecognised name is documented to fall back to AM1.5 rather than raise."""
    fallback = db_fom.load_spectrum("not a real spectrum")

    np.testing.assert_array_equal(fallback, am15)


def test_load_spectrum_photopic_is_no_longer_an_illuminant(am15: NDArray) -> None:
    """The photopic curve is V(lambda), not a light source, so it is not selectable.

    It was previously loadable as a spectrum and integrated as though it were
    an irradiance, which is a category error: it is the eye's response, and it
    now serves only to weight other spectra in calculate_illuminance.
    """
    fallback = db_fom.load_spectrum("Photopic")

    np.testing.assert_array_equal(fallback, am15)


def test_am15_integrated_irradiance(am15: NDArray) -> None:
    """AM1.5G integrates to 1000.4 W m^-2 - the defining property of the standard."""
    irradiance = np.trapezoid(am15[:, 1], am15[:, 0])

    assert irradiance == pytest.approx(1000.4, rel=1e-3)


# --- illuminance and spectrum normalisation --------------------------------


def test_illuminance_of_555nm_source_is_683_lumens_per_watt() -> None:
    """At 555 nm one watt is 683 lumens, which is what defines the lumen.

    A narrow band centred on the photopic peak must therefore return 683 times
    its integrated irradiance. This pins both the constant and the alignment of
    the V(lambda) grid.
    """
    wavelengths = np.linspace(554.0, 556.0, 201)
    irradiance = np.ones_like(wavelengths)
    spectrum = np.column_stack([wavelengths, irradiance])

    illuminance = db_fom.calculate_illuminance(spectrum)
    integrated = np.trapezoid(irradiance, wavelengths)

    assert illuminance == pytest.approx(683.0 * integrated, rel=1e-3)


def test_illuminance_ignores_wavelengths_outside_photopic_range() -> None:
    """The eye does not respond in the infrared, so such power adds no lux."""
    wavelengths = np.linspace(1500.0, 2000.0, 101)
    spectrum = np.column_stack([wavelengths, np.ones_like(wavelengths)])

    assert db_fom.calculate_illuminance(spectrum) == pytest.approx(0.0)


def test_am15_illuminance_matches_full_sun(am15: NDArray) -> None:
    """One sun is of order 10^5 lux; AM1.5G gives 1.16e5."""
    assert db_fom.calculate_illuminance(am15) == pytest.approx(1.1566e5, rel=1e-3)


@pytest.mark.parametrize("spectrum_type", VISIBLE_ILLUMINANTS)
def test_visible_illuminants_load_at_the_target_illuminance(spectrum_type: str) -> None:
    """The whole point of the default: every visible illuminant arrives at 1000 lux."""
    spectrum = db_fom.load_spectrum(spectrum_type)

    assert db_fom.calculate_illuminance(spectrum) == pytest.approx(1000.0, rel=1e-6)


@pytest.mark.parametrize("target_lux", [50.0, 200.0, 1000.0, 25_000.0])
def test_normalise_spectrum_round_trips_any_target(target_lux: float) -> None:
    """Scaling to a requested illuminance and measuring it back must agree."""
    raw = db_fom.load_spectrum("White LED", target_lux=None)

    normalised = db_fom.normalise_spectrum(raw, target_lux=target_lux)

    assert db_fom.calculate_illuminance(normalised) == pytest.approx(target_lux, rel=1e-9)


def test_normalise_spectrum_preserves_shape() -> None:
    """Normalisation changes only the magnitude, never the spectral shape."""
    raw = db_fom.load_spectrum("Fluorescent", target_lux=None)

    normalised = db_fom.normalise_spectrum(raw, target_lux=1000.0)

    np.testing.assert_array_equal(normalised[:, 0], raw[:, 0])
    ratio = normalised[raw[:, 1] > 0, 1] / raw[raw[:, 1] > 0, 1]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)


def test_normalise_spectrum_does_not_mutate_input() -> None:
    """Like convert_spectrum, normalise_spectrum works on a copy."""
    raw = db_fom.load_spectrum("Red LED", target_lux=None)
    before = raw.copy()

    db_fom.normalise_spectrum(raw, target_lux=1000.0)

    np.testing.assert_array_equal(raw, before)


@pytest.mark.parametrize("spectrum_type", sorted(IRRADIANCE_AT_1000_LUX))
def test_normalised_irradiances_are_indoor_scale(spectrum_type: str) -> None:
    """1000 lux lands at a few W m^-2 - three orders below one sun, as it should.

    The raw files span 0.1 to 213 W m^-2, the latter a fifth of full sun for
    what is meant to be room lighting, because red, white and IR were stored
    normalised to a peak of 1.0. This is the test that catches a regression to
    treating those relative spectra as absolute.
    """
    spectrum = db_fom.load_spectrum(spectrum_type)

    irradiance = np.trapezoid(spectrum[:, 1], spectrum[:, 0])

    assert irradiance == pytest.approx(IRRADIANCE_AT_1000_LUX[spectrum_type], rel=1e-3)


def test_ir_led_is_power_matched_not_illuminance_matched() -> None:
    """The IR LED is invisible, so it is scaled on radiant power instead.

    Its illuminance is ~0.009 lux as stored; forcing that to 1000 lux would
    multiply it by 10^5 and hand the efficiency model megawatts per square
    metre. It is matched to the white LED's irradiance at the same target.
    """
    ir = db_fom.load_spectrum("IR LED")
    white = db_fom.load_spectrum("White LED")

    ir_power = np.trapezoid(ir[:, 1], ir[:, 0])
    white_power = np.trapezoid(white[:, 1], white[:, 0])

    assert ir_power == pytest.approx(white_power, rel=1e-3)
    # And it is emphatically not at the target illuminance.
    assert db_fom.calculate_illuminance(ir) < 1.0


def test_load_spectrum_none_returns_the_raw_file() -> None:
    """target_lux=None is the escape hatch for anyone wanting the stored data.

    The red, white and IR files are peak-normalised, so a peak of exactly 1.0
    is the signature of the unscaled data. Raw mode is also unpadded, and the
    CIE LED files are the ones whose native range that shows.
    """
    raw = db_fom.load_spectrum("Red LED", target_lux=None)

    assert raw[:, 1].max() == 1.0

    native = db_fom.load_spectrum("LED-B1", target_lux=None)

    assert (native[0, 0], native[-1, 0]) == (380.0, 780.0)


# --- the common wavelength support -----------------------------------------


@pytest.mark.parametrize("spectrum_type", SPECTRUM_TYPES)
def test_every_spectrum_spans_the_common_support(spectrum_type: str) -> None:
    """A loaded spectrum must cover 280-4000 nm however narrow its source file.

    The detailed-balance integrals take their wavelength grid from the
    illumination spectrum, so the support is a correctness requirement rather
    than a presentational one.
    """
    spectrum = db_fom.load_spectrum(spectrum_type)

    assert spectrum[0, 0] <= db_fom.SPECTRUM_WL_MIN
    assert spectrum[-1, 0] >= db_fom.SPECTRUM_WL_MAX
    assert np.all(np.diff(spectrum[:, 0]) > 0)


@pytest.mark.parametrize("spectrum_type", ["LED-B1", "Blue LED", "White LED"])
def test_padding_adds_no_power_and_no_light(spectrum_type: str) -> None:
    """Zero-irradiance padding must not disturb any measured quantity."""
    raw = db_fom.load_spectrum(spectrum_type, target_lux=None)
    scaled = db_fom.normalise_spectrum(raw, target_lux=1000.0)
    padded = db_fom._pad_to_common_support(scaled)

    assert db_fom.calculate_illuminance(padded) == pytest.approx(
        db_fom.calculate_illuminance(scaled), rel=1e-6
    )
    assert np.trapezoid(padded[:, 1], padded[:, 0]) == pytest.approx(
        np.trapezoid(scaled[:, 1], scaled[:, 0]), rel=1e-3
    )


def test_pad_leaves_a_full_range_spectrum_untouched() -> None:
    """ASTMG173 already spans the support, so padding must be a no-op on it."""
    am15 = db_fom.load_spectrum("AM1.5", target_lux=None)

    np.testing.assert_array_equal(db_fom._pad_to_common_support(am15), am15)


def test_radiative_recombination_is_independent_of_the_illuminant() -> None:
    """rr0 is dark blackbody emission above the gap: the lamp cannot change it.

    It is computed on the wavelength grid the illumination spectrum supplies,
    which is why the grids have to share a support. Before padding, a spectrum
    ending at 780 nm truncated this integral at 1.59 eV and returned a value
    orders of magnitude too small for any gap below that, silently inflating
    every efficiency derived from it.
    """
    spectra = {
        name: db_fom.convert_spectrum(db_fom.load_spectrum(name))
        for name in ("AM1.5", "LED-B1", "Blue LED", "White LED")
    }

    for E_gap in (0.9, 1.1, 1.3, 1.5):
        rates = [db_fom._rr0(E_gap, s, 300.0) for s in spectra.values()]

        assert max(rates) == pytest.approx(min(rates), rel=1e-2)


def test_sq_limit_varies_with_gap_below_the_source_cutoff() -> None:
    """The SQ limit must keep responding to the gap below the LED's 780 nm edge.

    The truncation bug did not merely shift this curve, it flattened it: every
    gap below 1.59 eV returned an identical efficiency because the recombination
    integral had stopped depending on the gap at all.
    """
    photon_spectrum = db_fom.convert_spectrum(db_fom.load_spectrum("LED-B1"))

    efficiencies = [db_fom.max_eff(E_gap, photon_spectrum, 300.0) for E_gap in (0.9, 1.1, 1.3, 1.5)]

    assert all(b > a for a, b in zip(efficiencies, efficiencies[1:]))


def test_am15_is_never_rescaled(am15: NDArray) -> None:
    """AM1.5 is an absolute standard, so target_lux must not touch it."""
    rescaled = db_fom.load_spectrum("AM1.5", target_lux=200.0)

    np.testing.assert_array_equal(rescaled, am15)


def test_normalise_spectrum_rejects_zero_illuminance() -> None:
    """An invisible spectrum has no illuminance to scale, so say so."""
    wavelengths = np.linspace(1500.0, 2000.0, 101)
    infrared = np.column_stack([wavelengths, np.ones_like(wavelengths)])

    with pytest.raises(ValueError, match="target_irradiance instead"):
        db_fom.normalise_spectrum(infrared, target_lux=1000.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one"),
        ({"target_lux": 100.0, "target_irradiance": 1.0}, "exactly one"),
        ({"target_lux": 0.0}, "must be positive"),
        ({"target_irradiance": -1.0}, "must be positive"),
    ],
)
def test_normalise_spectrum_rejects_bad_targets(kwargs: dict, match: str) -> None:
    """The two bases are mutually exclusive, and neither accepts a non-positive target."""
    spectrum = db_fom.load_spectrum("White LED", target_lux=None)

    with pytest.raises(ValueError, match=match):
        db_fom.normalise_spectrum(spectrum, **kwargs)


def test_convert_spectrum_does_not_mutate_input(am15: NDArray) -> None:
    """convert_spectrum works on a copy and leaves its input array untouched."""
    before = am15.copy()

    db_fom.convert_spectrum(am15)

    np.testing.assert_array_equal(am15, before)


def test_convert_spectrum_energy_wavelength_relation(
        am15: NDArray, photon_spectrum: NDArray
) -> None:
    """Every row's energy must be hc/lambda for that row's wavelength."""
    expected_eV = HC_EV_NM / am15[:, 0]

    np.testing.assert_allclose(photon_spectrum[:, 0], expected_eV, rtol=1e-12)


def test_convert_spectrum_conserves_power(am15: NDArray, photon_spectrum: NDArray) -> None:
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


def test_photons_above_bandgap_monotonic_decreasing(photon_spectrum: NDArray) -> None:
    """Raising the gap can only discard photons."""
    gaps = np.arange(0.5, 3.01, 0.25)
    counts = [db_fom._photons_above_bandgap(g, photon_spectrum) for g in gaps]

    assert np.all(np.diff(counts) < 0)
    assert counts[0] > 0


def test_photons_above_bandgap_above_spectrum_max_is_zero(photon_spectrum: NDArray) -> None:
    """No photons exist above the spectrum's highest energy."""
    above_max = photon_spectrum[:, 0].max() + 1.0

    assert db_fom._photons_above_bandgap(above_max, photon_spectrum) == 0.0


def test_rr0_increases_with_temperature(photon_spectrum: NDArray) -> None:
    """A hotter cell emits more; J0 is the blackbody flux at zero QFL splitting."""
    cold = db_fom._rr0(1.34, photon_spectrum, 250.0)
    hot = db_fom._rr0(1.34, photon_spectrum, 400.0)

    assert hot > cold > 0


def test_rr0_decreases_with_bandgap(photon_spectrum: NDArray) -> None:
    """The blackbody tail falls off exponentially, so a wider gap emits less."""
    gaps = [0.8, 1.1, 1.4, 1.7, 2.0]
    rates = [db_fom._rr0(g, photon_spectrum, 300.0) for g in gaps]

    assert np.all(np.diff(rates) < 0)


# --- the diode ------------------------------------------------------------


def test_voc_below_bandgap(photon_spectrum: NDArray) -> None:
    """q*Voc < E_gap for every gap - the thermodynamic ceiling on open-circuit voltage."""
    for gap in np.arange(0.5, 3.01, 0.25):
        voc = db_fom.voc(gap, photon_spectrum, 300.0)

        assert 0 < voc < gap


def test_jsc_equals_q_times_photon_flux(photon_spectrum: NDArray) -> None:
    """At short circuit every above-gap photon contributes one electron."""
    expected = sc.e * db_fom._photons_above_bandgap(1.34, photon_spectrum)

    assert db_fom.jsc(1.34, photon_spectrum, 300.0) == pytest.approx(expected, rel=1e-12)


def test_current_density_broadcasts_over_voltage(photon_spectrum: NDArray) -> None:
    """An array of voltages yields an array of currents of the same shape."""
    voltages = np.linspace(0.0, 0.8, 17)

    current = db_fom.current_density(1.34, photon_spectrum, voltages, 300.0)

    assert isinstance(current, np.ndarray)
    assert current.shape == voltages.shape


def test_current_density_non_increasing_in_voltage(photon_spectrum: NDArray) -> None:
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


def test_current_density_vanishes_at_voc(photon_spectrum: NDArray) -> None:
    """Voc is by definition the voltage at which the net current is zero."""
    v_oc = db_fom.voc(1.34, photon_spectrum, 300.0)
    j_sc = db_fom.jsc(1.34, photon_spectrum, 300.0)

    at_voc = db_fom.current_density(1.34, photon_spectrum, v_oc, 300.0)

    assert abs(at_voc) < 1e-6 * j_sc


# --- efficiency -----------------------------------------------------------


def test_max_eff_matches_sq_limit(photon_spectrum: NDArray) -> None:
    """The published single-junction SQ limit is 33.7 % at 1.34 eV under AM1.5G."""
    efficiency = db_fom.max_eff(1.34, photon_spectrum, 300.0)

    assert efficiency == pytest.approx(0.337, abs=0.005)


def test_max_eff_peaks_near_1_34_eV(photon_spectrum: NDArray) -> None:
    """The SQ curve's maximum sits at 1.34 eV; allow a window for the grid."""
    gaps = np.arange(0.6, 2.51, 0.02)
    efficiencies = np.array([db_fom.max_eff(g, photon_spectrum, 300.0) for g in gaps])

    assert 1.1 < gaps[efficiencies.argmax()] < 1.5


def test_max_eff_bounded(photon_spectrum: NDArray) -> None:
    """max_eff returns a fraction, so it must sit strictly inside (0, 1)."""
    for gap in (0.8, 1.34, 2.0, 2.8):
        efficiency = db_fom.max_eff(gap, photon_spectrum, 300.0)

        assert 0.0 < efficiency < 1.0


def test_incident_power_recovers_the_irradiance(am15: NDArray, photon_spectrum: NDArray) -> None:
    """incident_power inverts convert_spectrum, so it returns the AM1.5 irradiance."""
    power = db_fom.incident_power(photon_spectrum)

    assert power == pytest.approx(np.trapezoid(am15[:, 1], am15[:, 0]), rel=1e-6)


def test_max_eff_accepts_a_precomputed_incident_power(photon_spectrum: NDArray) -> None:
    """Passing power_in is an optimisation for sweeps, never a change of answer.

    The incident power does not depend on the band gap, so a gap sweep hoists
    it out of the loop; the result must be identical to letting max_eff work it
    out for itself.
    """
    power = db_fom.incident_power(photon_spectrum)

    for E_gap in (0.9, 1.34, 2.0):
        hoisted = db_fom.max_eff(E_gap, photon_spectrum, 300.0, power_in=power)
        internal = db_fom.max_eff(E_gap, photon_spectrum, 300.0)

        assert hoisted == internal


def test_power_curve_spans_short_to_open_circuit(photon_spectrum: NDArray) -> None:
    """The shared voltage sweep runs from 0 to Voc, with J falling to zero there.

    Every maximum-power-point quantity is a reduction of this one sweep, so its
    endpoints are what pin them all.
    """
    voltage, current = db_fom._power_curve(1.34, photon_spectrum, 300.0)

    assert voltage[0] == 0.0
    assert voltage[-1] == pytest.approx(db_fom.voc(1.34, photon_spectrum, 300.0), rel=1e-12)
    assert current[0] == pytest.approx(db_fom.jsc(1.34, photon_spectrum, 300.0), rel=1e-12)
    assert current[-1] == pytest.approx(0.0, abs=1e-9 * current[0])


def test_mpp_quantities_come_from_one_sweep(photon_spectrum: NDArray) -> None:
    """P_max, V_mpp and J_mpp must be mutually consistent.

    They used to be computed by three independent rebuilds of the same
    voltage sweep; sharing it is only safe if the identity below still holds.
    """
    p_max = db_fom.max_power(1.34, photon_spectrum, 300.0)
    v_mpp = db_fom.v_at_mpp(1.34, photon_spectrum)
    j_mpp = db_fom.j_at_mpp(1.34, photon_spectrum)

    assert v_mpp * j_mpp == pytest.approx(p_max, rel=1e-12)


def test_max_power_between_zero_and_jsc_voc(photon_spectrum: NDArray) -> None:
    """P_max = FF * Jsc * Voc with 0 < FF < 1, so it is bounded by the Jsc-Voc rectangle."""
    p_max = db_fom.max_power(1.34, photon_spectrum, 300.0)
    j_sc = db_fom.jsc(1.34, photon_spectrum, 300.0)
    v_oc = db_fom.voc(1.34, photon_spectrum, 300.0)

    assert 0 < p_max < j_sc * v_oc


# --- public entry points outside the tutorial workflow ----------------------


def test_recomb_rate_returns_finite_float(photon_spectrum: NDArray) -> None:
    """recomb_rate should return a finite positive rate."""
    rate = db_fom.recomb_rate(1.34, photon_spectrum, 0.5, 300.0)

    assert np.isfinite(rate)
    assert rate > 0


def test_v_at_mpp_between_zero_and_voc(photon_spectrum: NDArray) -> None:
    """The maximum-power voltage should sit between zero and Voc."""
    v_mpp = db_fom.v_at_mpp(1.34, photon_spectrum)
    v_oc = db_fom.voc(1.34, photon_spectrum, 300.0)

    assert 0 < v_mpp < v_oc


def test_j_at_mpp_below_jsc(photon_spectrum: NDArray) -> None:
    """The maximum-power current should sit between zero and Jsc."""
    j_mpp = db_fom.j_at_mpp(1.34, photon_spectrum)
    j_sc = db_fom.jsc(1.34, photon_spectrum, 300.0)

    assert 0 < j_mpp < j_sc


def test_fill_factor_between_zero_and_one(photon_spectrum: NDArray) -> None:
    """A well-behaved single-junction cell near the SQ limit has FF around 0.8-0.9."""
    ff = db_fom.fill_factor(1.34, photon_spectrum, 300.0)

    assert 0.7 < ff < 0.95
