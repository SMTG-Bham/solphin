"""Tests for the AMSET transport reader and the doping-dependent Γₚᵥ sweep.

Every transport file here is synthetic, written into ``tmp_path`` by
:func:`write_transport` in AMSET's own JSON or text layout. The mobilities
follow closed-form power laws combined by Matthiessen's rule, so the reader's
averages, interpolation and impurity shares are checked against analytic
values rather than against a stored real calculation.
"""

import gzip
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

import solphin.final_results as final_results
import solphin.transport as transport

# Phonon-limited mobilities, flat in doping, and an ionized-impurity mobility
# falling as 1/n, normalised so impurities are a chosen share of the
# scattering at a chosen doping.
POP_E = 80.0
POP_H = 20.0
ADP = 2.0e4


def imp(n: NDArray, at: float = 1e18, mu: float = 800.0) -> NDArray:
    """Ionized-impurity mobility in cm² V⁻¹ s⁻¹, equal to ``mu`` at ``at`` cm⁻³."""
    return mu * at / n


def tensor(mu: float, anisotropy: float = 1.0) -> list[list[float]]:
    """A diagonal mobility tensor with isotropic mean ``mu`` and eigenvalue ratio ``anisotropy``."""
    a = 3 * mu / (2 + anisotropy)
    return np.diag([a, a, a * anisotropy]).tolist()


def build(
        doping: list[float],
        temperatures: tuple[float, ...] = (300.0,),
        imp_scale: float = 1.0,
        anisotropy: float = 1.0,
        mechanisms: bool = True,
) -> dict:
    """Return an AMSET-layout transport dict for the given signed dopings."""
    doping_arr = np.asarray(doping, dtype=float)
    n = np.abs(doping_arr)
    pop = np.where(doping_arr < 0, POP_E, POP_H)
    mu_imp = imp(n) / imp_scale
    mechs = {"IMP": mu_imp, "ADP": np.full_like(n, ADP), "POP": pop}
    overall = 1 / sum(1 / m for m in mechs.values())

    def per_t(values: NDArray | float, aniso: float = 1.0) -> list:
        values = np.atleast_1d(values)
        return [[tensor(v, aniso) for _ in temperatures] for v in values]

    mobility = {"overall": per_t(overall, anisotropy)}

    if mechanisms:
        mobility.update({k: per_t(v) for k, v in mechs.items()})

    zeros = np.zeros((len(doping), len(temperatures), 3, 3)).tolist()

    return {
        "doping": doping_arr.tolist(),
        "temperatures": list(temperatures),
        "fermi_levels": np.zeros((len(doping), len(temperatures))).tolist(),
        "conductivity": zeros,
        "seebeck": zeros,
        "electronic_thermal_conductivity": zeros,
        "mobility": mobility,
    }


def write_transport(path: Path, data: dict) -> Path:
    """Write ``data`` as AMSET JSON, gzipped JSON or text, by the suffix of ``path``."""
    if path.suffix == ".json":
        path.write_text(json.dumps(data))
    elif path.suffix == ".gz":
        with gzip.open(path, "wt") as handle:
            json.dump(data, handle)
    else:
        # AMSET's to_data()/np.savetxt layout: one row per (doping, T), six
        # upper-triangle components per tensor.
        triu = np.triu_indices(3)
        comps = ("xx", "xy", "xz", "yy", "yz", "zz")
        headers = ["doping[cm^-3]", "temperature[K]", "Fermi_level[eV]"]
        for prop, unit in [("cond", "S/m"), ("seebeck", "µV/K"), ("kappa", "?")]:
            headers += [f"{prop}_{d}[{unit}]" for d in comps]
        for name in data["mobility"]:
            headers += [f"{name}_mobility_{d}[cm^2/V.s]" for d in comps]

        rows = []
        for i, t in np.ndindex(len(data["doping"]), len(data["temperatures"])):
            row = [data["doping"][i], data["temperatures"][t], data["fermi_levels"][i][t]]
            row += [0.0] * 18
            for values in data["mobility"].values():
                row += list(np.asarray(values[i][t])[triu])
            rows.append(row)

        np.savetxt(path, rows, header=" ".join(headers), encoding="utf-8")

    return path


# Most synthetic files trip one of select_mobility's checks they are not
# about (one carrier, impurities at the top of the grid). Tests that are about a
# warning assert it with pytest.warns, which records regardless of this filter.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

N_TYPE = [-1e15, -1e16, -1e17, -1e18, -1e19]
P_TYPE = [1e15, 1e16, 1e17, 1e18, 1e19]


@pytest.fixture
def both(tmp_path: Path) -> Path:
    """A transport file with electrons and holes from 1e15 to 1e19."""
    return write_transport(tmp_path / "transport_both.json", build(N_TYPE + P_TYPE))


@pytest.fixture
def n_only(tmp_path: Path) -> Path:
    """A transport file with electrons only, like a typical n-type AMSET run."""
    return write_transport(tmp_path / "transport_n.json", build(N_TYPE))


def quiet_select(path: Path, **kwargs) -> transport.MobilityModel:
    """select_mobility with its warnings captured and its summary silenced."""
    with pytest.warns(UserWarning):
        return transport.select_mobility(path, verbose=False, **kwargs)


# --- reading ---------------------------------------------------------------


def test_json_reader_recovers_layout(both: Path) -> None:
    """Carriers come from the doping signs, mechanisms from the mobility keys."""
    data = transport.load_amset_transport(both)

    assert data.carriers == ("electrons", "holes")
    assert data.mechanisms == ("overall", "IMP", "ADP", "POP")
    assert data.mobility["overall"].shape == (10, 1, 3, 3)


@pytest.mark.parametrize("suffix", [".txt", ".json.gz"])
def test_text_and_gzip_readers_match_json(tmp_path: Path, suffix: str) -> None:
    """AMSET's text and gzipped layouts parse to the same arrays as its JSON."""
    raw = build(N_TYPE + P_TYPE, temperatures=(200.0, 300.0), anisotropy=1.2)
    reference = transport.load_amset_transport(write_transport(tmp_path / "t.json", raw))
    other = transport.load_amset_transport(write_transport(tmp_path / f"t{suffix}", raw))

    np.testing.assert_allclose(other.doping, reference.doping)
    np.testing.assert_allclose(other.temperatures, reference.temperatures)
    for name in reference.mechanisms:
        np.testing.assert_allclose(other.mobility[name], reference.mobility[name], rtol=1e-12)


def test_isotropic_average_is_trace_over_three(tmp_path: Path) -> None:
    """An anisotropic tensor still reports its eigenvalue mean, and its ratio."""
    path = write_transport(tmp_path / "t.json", build(N_TYPE, anisotropy=2.0))
    data = transport.load_amset_transport(path)

    n, mu = data.mobility_vs_doping("electrons")
    _, ratio = data.anisotropy("electrons")
    expected = 1 / (1 / imp(n) + 1 / ADP + 1 / POP_E)

    np.testing.assert_allclose(n, np.abs(N_TYPE))
    np.testing.assert_allclose(mu, expected)
    np.testing.assert_allclose(ratio, 2.0)


def test_imp_share_is_matthiessen_fraction(n_only: Path) -> None:
    """μ_overall / μ_IMP is the share of 1/μ that ionized impurities contribute."""
    result = transport.load_amset_transport(n_only).imp_share("electrons")
    assert result is not None
    n, share = result
    expected = (1 / imp(n)) / (1 / imp(n) + 1 / ADP + 1 / POP_E)

    np.testing.assert_allclose(share, expected)


def test_reader_refuses_files_without_mobility(tmp_path: Path) -> None:
    """A run with calculate_mobility: false is named as such."""
    raw = build(N_TYPE)
    raw["mobility"] = None
    path = write_transport(tmp_path / "t.json", raw)

    with pytest.raises(ValueError, match="calculate_mobility"):
        transport.load_amset_transport(path)


def test_reader_refuses_a_non_amset_json(tmp_path: Path) -> None:
    """A JSON without AMSET's keys is refused with the keys named."""
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"foo": 1}))

    with pytest.raises(ValueError, match="doping"):
        transport.load_amset_transport(path)


# --- choosing the mobility --------------------------------------------------


def test_limiting_takes_the_lower_carrier_pointwise(both: Path) -> None:
    """With both carriers, the model returns min(μe, μh) at every doping."""
    model = transport.select_mobility(both, dop_range=(1e15, 1e19), verbose=False)
    n = np.logspace(15, 19, 9)
    per_carrier = model.by_carrier(n)

    np.testing.assert_allclose(model(n), np.minimum(per_carrier["electrons"], per_carrier["holes"]))
    assert set(model.limiting_carrier(n)) == {"holes"}
    assert any("Limiting carrier: holes from 1e+15" in note for note in model.notes)


def test_explicit_carrier_is_used_alone(both: Path) -> None:
    """carrier='electrons' ignores the holes, even where they are slower."""
    model = transport.select_mobility(
        both, carrier="electrons", dop_range=(1e15, 1e19), verbose=False
    )

    assert list(model.curves) == ["electrons"]
    assert model(1e15) > POP_H


def test_single_carrier_file_warns_and_says_which(n_only: Path) -> None:
    """Only n-type data: warned, and the summary names the electron mobility."""
    with pytest.warns(UserWarning, match="Only electron mobility"):
        model = transport.select_mobility(n_only, dop_range=(1e15, 1e19), verbose=False)

    assert "only electrons in the file" in str(model)


def test_missing_carrier_temperature_or_mechanism_is_refused(n_only: Path) -> None:
    """Asking for data the file does not hold fails with what is available."""
    with pytest.raises(ValueError, match="no hole mobility"):
        transport.select_mobility(n_only, carrier="holes", verbose=False)

    with pytest.raises(ValueError, match="available: 300 K"):
        quiet_select(n_only, temperature=350.0)

    with pytest.raises(ValueError, match="available: overall"):
        transport.select_mobility(n_only, mechanism="PIE", verbose=False)

    with pytest.raises(ValueError, match="carrier must be"):
        transport.select_mobility(n_only, carrier="minority", verbose=False)


def test_single_mechanism_is_flagged(n_only: Path) -> None:
    """Choosing one mechanism is allowed, warned and reported."""
    with pytest.warns(UserWarning, match="'POP'-limited"):
        model = transport.select_mobility(
            n_only, mechanism="POP", dop_range=(1e15, 1e19), verbose=False
        )

    assert model(1e17) == pytest.approx(POP_E)
    assert "'POP' only" in str(model)


# --- doping coverage --------------------------------------------------------


def test_flat_extrapolation_below_grid_when_phonon_limited(n_only: Path) -> None:
    """Below a phonon-limited grid start, μ is held flat and only noted."""
    with pytest.warns(UserWarning) as record:
        model = transport.select_mobility(n_only, dop_range=(1e10, 1e18), verbose=False)

    assert not any("held at" in str(w.message) for w in record)
    assert any(note.startswith("ℹ") and "accurate" in note for note in model.notes)
    assert model(1e10) == pytest.approx(model(1e15))


def test_flat_extrapolation_warns_when_impurities_still_matter(tmp_path: Path) -> None:
    """If impurities still scatter at the grid start, the flat value is flagged."""
    path = write_transport(tmp_path / "t.json", build(N_TYPE, imp_scale=1e3))

    with pytest.warns(UserWarning, match="underestimated"):
        transport.select_mobility(path, dop_range=(1e10, 1e18), verbose=False)


def test_range_above_grid_is_refused_or_held(n_only: Path) -> None:
    """Above the grid: refused by default, held at the last value on request."""
    with pytest.raises(ValueError, match="ends at 1e\\+19"):
        quiet_select(n_only, dop_range=(1e15, 1e20))

    model = quiet_select(n_only, dop_range=(1e15, 1e20), allow_out_of_range=True)

    with pytest.warns(UserWarning, match="held at its last value"):
        assert model(1e20) == pytest.approx(model(1e19))


def test_model_refuses_densities_above_grid(n_only: Path) -> None:
    """A model built for the default range still refuses a later out-of-grid call."""
    model = quiet_select(n_only, dop_range=(1e15, 1e19))

    with pytest.raises(ValueError, match="above the highest doping"):
        model(1e21)


def test_loglog_interpolation_is_exact_for_power_laws(tmp_path: Path) -> None:
    """Between grid points, a pure power law μ ∝ n^-1 is reproduced exactly."""
    raw = build(N_TYPE)
    pure = [[tensor(v)] for v in imp(np.abs(N_TYPE))]
    raw["mobility"] = {"overall": pure}
    path = write_transport(tmp_path / "t.json", raw)

    with pytest.warns(UserWarning, match="no separate IMP"):
        model = transport.select_mobility(path, dop_range=(1e10, 1e19), verbose=False)

    n = np.array([3e15, 4.2e16, 7e18])
    np.testing.assert_allclose(model(n), imp(n), rtol=1e-12)


# --- sensitivity checks ------------------------------------------------------


def test_significant_impurity_scattering_is_warned(n_only: Path) -> None:
    """Impurities above 10% of the scattering in range name compensation_factor."""
    with pytest.warns(UserWarning, match="compensation_factor"):
        transport.select_mobility(n_only, dop_range=(1e15, 1e19), verbose=False)


def test_anisotropy_is_warned(tmp_path: Path) -> None:
    """A tensor whose eigenvalues differ by 2x is flagged."""
    path = write_transport(tmp_path / "t.json", build(N_TYPE, anisotropy=2.0))

    with pytest.warns(UserWarning, match="anisotropic"):
        transport.select_mobility(path, dop_range=(1e15, 1e19), verbose=False)


def test_summary_is_printed_by_default(both: Path, capsys: pytest.CaptureFixture) -> None:
    """verbose=True prints which carrier, mechanism and tensor average are used."""
    transport.select_mobility(both, dop_range=(1e15, 1e19))
    out = capsys.readouterr().out

    assert "limiting carrier" in out
    assert "'overall' (total)" in out
    assert "isotropic average" in out


# --- the doping sweep ---------------------------------------------------------

# dict[str, Any] for the same reason as in test_final_results: it is splatted
# into functions with keyword-only bool and tuple parameters.
MAPI: dict[str, Any] = {
    "E_gap": 1.55,
    "alpha": 9.9e4,
    "tau": 6.9e-7,
    "sigma": 0.63,
    "dos_mass": 0.15,
    "epsilon": 33.5,
}


def test_constant_mobility_sweep_matches_single_evaluations(photon_spectrum: NDArray) -> None:
    """A float mobility reproduces SQ_relative_FOM_PV_efficiency point by point."""
    densities, mobilities, efficiencies = final_results.FOM_vs_doping(
        photon_spectrum=photon_spectrum, mobility=10.0, Tcell=300.0, n_points=3, **MAPI
    )
    _, _, at_middle = final_results.SQ_relative_FOM_PV_efficiency(
        photon_spectrum=photon_spectrum, dop_density=densities[1], mu=10.0, Tcell=300.0, **MAPI
    )

    np.testing.assert_allclose(mobilities, 10.0)
    assert efficiencies[1] == pytest.approx(at_middle)


def test_callable_mobility_is_evaluated_at_each_density(photon_spectrum: NDArray) -> None:
    """A function of n is called at every density of the sweep."""
    mobility: Callable[[float], float] = lambda n: 1e3 * (1e10 / n) ** 0.25  # noqa: E731

    densities, mobilities, _ = final_results.FOM_vs_doping(
        photon_spectrum=photon_spectrum, mobility=mobility, Tcell=300.0, n_points=5, **MAPI
    )

    np.testing.assert_allclose(mobilities, [mobility(d) for d in densities])


def test_sweep_with_amset_model_and_plots(n_only: Path, photon_spectrum: NDArray) -> None:
    """The model plugs into the sweep and both figures draw."""
    model = quiet_select(n_only)

    ax_mu, ax_eff = final_results.plot_FOM_vs_doping(
        photon_spectrum=photon_spectrum, mobility=model, Tcell=300.0, mu_reference=10.0, **MAPI
    )
    ax = transport.plot_mobility(n_only)

    assert len(ax_mu.lines) == 2 and len(ax_eff.lines) == 2
    # total + three mechanisms, electrons only
    assert len(ax.lines) == 4
    plt.close("all")


def test_sweep_refuses_range_outside_table_1(photon_spectrum: NDArray) -> None:
    """The doping sweep keeps the table 1 guard of the other sweeps."""
    with pytest.raises(ValueError, match="dop_density"):
        final_results.FOM_vs_doping(
            photon_spectrum=photon_spectrum, mobility=10.0, Tcell=300.0,
            dop_range=(1e10, 1e20), **MAPI,
        )
