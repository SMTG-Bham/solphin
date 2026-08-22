"""Density of states -> effective mass.

_calculate_DOS fits g(E) = A*sqrt(dE) and inverts m = hbar^2/2 * (2*pi^2*A)^(2/3),
which is the standard three-dimensional parabolic-band relation. That makes it
invertible: build a DOS for a chosen mass and require the mass back.

Note the import style. dos.py defines a public helper literally named
test_dos_mass_windows; importing that name into this module's namespace would
have pytest collect it as a test and error on a missing fixture. Everything is
reached through the module instead.
"""

import numpy as np
import pytest
from conftest import requires_potcars

import solphin.dos as dos
import solphin.vasp_inputs as vasp_inputs

# --- the fit ---------------------------------------------------------------


def test_check_fit_perfect_and_null():
    """R^2 is 1 for data that lies exactly on the fit, and <= 0 for a flat signal."""
    x = np.linspace(0.1, 1.0, 20)

    assert dos._check_fit(3.0, x, 3.0 * x) == pytest.approx(1.0, abs=1e-12)
    assert dos._check_fit(3.0, x, np.full_like(x, 3.0 * x.mean())) <= 0.0


@pytest.mark.parametrize("target_mass", [0.10, 0.50, 1.00])
def test_calculate_dos_recovers_known_mass(target_mass):
    """Run the parabolic-band relation backwards and require the mass back.

    g(E) = (1/(2*pi^2)) * (2m/hbar^2)^(3/2) * sqrt(E), so choosing m fixes the
    coefficient A that _calculate_DOS should recover.
    """
    mass_si = target_mass * dos.M_E
    coefficient = (2 * mass_si / dos.HBAR**2) ** 1.5 / (2 * np.pi**2)
    delta_E_J = np.linspace(1e-23, 1e-20, 50)
    dos_si = coefficient * np.sqrt(delta_E_J)

    m_eff_rel, m_eff_si, r2, _, fitted_A = dos._calculate_DOS(delta_E_J, dos_si)

    assert m_eff_rel == pytest.approx(target_mass, rel=1e-9)
    assert m_eff_si == pytest.approx(mass_si, rel=1e-9)
    assert fitted_A == pytest.approx(coefficient, rel=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_calculate_dos_rejects_zero_energy_spread():
    with pytest.raises(ValueError, match="zero energy spread"):
        dos._calculate_DOS(np.zeros(5), np.zeros(5))


def test_calculate_dos_rejects_non_positive_coefficient():
    delta_E_J = np.linspace(1e-23, 1e-20, 10)

    with pytest.raises(ValueError, match="non-positive"):
        dos._calculate_DOS(delta_E_J, -np.sqrt(delta_E_J))


def test_clean_dos_values_filters():
    """Drops points at or below the edge, and any DOS not above min_dos.

    energy_window is only used to phrase the error message - the windowing
    itself has already happened by the time this is called.
    """
    delta_E_ev = np.array([-0.05, 0.02, 0.05, np.nan, 0.08, 0.11])
    density = np.array([1.0, 2.0, 0.0, 4.0, 5.0, 6.0])

    kept_E, kept_dos = dos._clean_dos_values(delta_E_ev, density, 0.0, 0.1)

    np.testing.assert_allclose(kept_E, [0.02, 0.08, 0.11])
    np.testing.assert_allclose(kept_dos, [2.0, 5.0, 6.0])


def test_clean_dos_values_requires_three_points():
    """Fewer than three survivors cannot support a fit, and the message says so."""
    delta_E_ev = np.array([0.01, 0.02, -0.03])
    density = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="usable DOS points"):
        dos._clean_dos_values(delta_E_ev, density, 0.0, 0.1)


# --- against the committed DOS calculation --------------------------------


def test_compute_dos_tutorial_value(dos_result):
    """The effective mass the tutorial feeds into the figure of merit."""
    assert dos_result.final_result == pytest.approx(0.0727540, rel=1e-6)
    assert dos_result.cbm == pytest.approx(1.40177392, rel=1e-6)
    assert dos_result.cell_volume_m3 == pytest.approx(2.2340578e-28, rel=1e-6)
    assert dos_result.carrier == "electrons"


def test_compute_dos_poor_fit_is_visible(dos_result):
    """The tutorial's own DOS is too coarse to resolve the mass, and says so.

    R^2 = 0.671 against a 0.80 threshold, on 7 points against a 10-point
    threshold. This is a property of the shipped data, and it must stay visible
    rather than being quietly rounded off.
    """
    assert dos_result.fit_quality_e < dos.MIN_DOS_FIT_R2
    assert dos_result.em_electrons.n_points < dos.MIN_DOS_FIT_POINTS


def test_compute_dos_holes_carrier(dos_vasprun):
    """Selecting holes returns the hole mass, and em_result follows the selection."""
    result = dos.compute_dos(
        dos_vasprun=str(dos_vasprun), carrier="holes", energy_window=0.1
    )

    assert result.carrier == "holes"
    assert result.em_result is result.em_holes
    assert result.final_result == pytest.approx(result.em_holes.m_eff_rel, rel=1e-12)
    # Holes in Cu2GeS3 are far heavier than electrons.
    assert result.final_result > 0.5


def test_compute_dos_rejects_bad_carrier(dos_vasprun):
    with pytest.raises(ValueError):
        dos.compute_dos(dos_vasprun=str(dos_vasprun), carrier="phonons")


def test_compute_dos_m_eff_override(dos_vasprun):
    """An explicit mass bypasses the fit - the documented escape hatch for coarse data."""
    result = dos.compute_dos(
        dos_vasprun=str(dos_vasprun), m_eff=0.25, carrier="electrons", energy_window=0.1
    )

    assert result.final_result == pytest.approx(0.25, rel=1e-12)


def test_get_dos_effective_mass_matches_compute_dos(dos_vasprun, dos_result):
    single = dos.get_dos_effective_mass(
        dos_vasprun=str(dos_vasprun), carrier="electrons", energy_window=0.1
    )

    assert single.m_eff_rel == pytest.approx(dos_result.final_result, rel=1e-12)
    assert single.E_c == single.E_edge


def test_dos_mass_windows_widen_the_fit(dos_vasprun):
    """Wider windows admit more points; reached through the module to avoid collection."""
    rows = dos.test_dos_mass_windows(
        str(dos_vasprun), carrier="electrons", windows=(0.05, 0.1, 0.2)
    )

    assert len(rows) == 3


def test_dos_result_str_contains_mass(dos_result):
    rendered = str(dos_result)

    assert "0.072754" in rendered
    assert "electrons" in rendered.lower()


# --- input generation ------------------------------------------------------


def test_generate_local_kpoints_mesh_size():
    """A 3x3x3 local mesh around Gamma is 27 points, all within delta of the centre."""
    kpoints = dos._generate_local_kpoints(np.array([0.0, 0.0, 0.0]), (3, 3, 3), 0.01)

    assert kpoints.num_kpts == 27
    assert np.max(np.abs(np.asarray(kpoints.kpts))) <= 0.01 + 1e-12


@requires_potcars
def test_write_eff_mass_creates_inputs(relax_dir, tmp_path):
    structure = vasp_inputs.read_structure_pmg(relax_dir / "CONTCAR")

    dos.write_eff_mass(
        k0_frac=np.array([0.0, 0.0, 0.0]),
        structure=structure,
        functional="HSE06",
        encut=450,
        folder=str(tmp_path / "eff_mass"),
    )

    written = {p.name for p in (tmp_path / "eff_mass").iterdir()}
    assert {"INCAR", "POSCAR", "KPOINTS", "POTCAR"} <= written


@pytest.mark.xfail(
    strict=True,
    reason=(
        "dos.py:1283 calls len() and .tolist() on the pymatgen Kpoints object "
        "returned by _generate_local_kpoints; Kpoints has neither -> TypeError"
    ),
)
def test_write_local_kpoints_writes_file(tmp_path):
    dos.write_local_kpoints(str(tmp_path), np.array([0.0, 0.0, 0.0]), (3, 3, 3), 0.01)

    assert (tmp_path / "KPOINTS").is_file()
