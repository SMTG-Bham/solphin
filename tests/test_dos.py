"""Density of states -> effective mass.

_calculate_DOS fits g(E) = A*sqrt(dE) and inverts m = hbar^2/2 * (2*pi^2*A)^(2/3),
which is the standard three-dimensional parabolic-band relation. That makes it
invertible: build a DOS for a chosen mass and require the mass back.

Note the import style. dos.py defines a public helper literally named
test_dos_mass_windows; importing that name into this module's namespace would
have pytest collect it as a test and error on a missing fixture. Everything is
reached through the module instead.
"""

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Kpoints

import castep_fixtures
import solphin.dos as dos
import solphin.vasp_inputs as vasp_inputs
from conftest import requires_potcars
from solphin.dos import DOSResult


def test_check_fit_perfect_and_null() -> None:
    """R^2 is 1 for data that lies exactly on the fit, and <= 0 for a flat signal."""
    x = np.linspace(0.1, 1.0, 20)

    assert dos._check_fit(3.0, x, 3.0 * x) == pytest.approx(1.0, abs=1e-12)
    assert dos._check_fit(3.0, x, np.full_like(x, 3.0 * x.mean())) <= 0.0


@pytest.mark.parametrize("target_mass", [0.10, 0.50, 1.00])
def test_calculate_dos_recovers_known_mass(target_mass: float) -> None:
    """Run the parabolic-band relation backwards and require the mass back.

    g(E) = (1/(2*pi^2)) * (2m/hbar^2)^(3/2) * sqrt(E), so choosing m fixes the
    coefficient A that _calculate_DOS should recover.
    """
    mass_si = target_mass * dos.M_E
    coefficient = (2 * mass_si / dos.HBAR ** 2) ** 1.5 / (2 * np.pi ** 2)
    delta_E_J = np.linspace(1e-23, 1e-20, 50)
    dos_si = coefficient * np.sqrt(delta_E_J)

    m_eff_rel, m_eff_si, r2, _, fitted_A = dos._calculate_DOS(delta_E_J, dos_si)

    assert m_eff_rel == pytest.approx(target_mass, rel=1e-9)
    assert m_eff_si == pytest.approx(mass_si, rel=1e-9)
    assert fitted_A == pytest.approx(coefficient, rel=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_calculate_dos_rejects_zero_energy_spread() -> None:
    """All-zero energies cannot support a fit and raise the zero-spread ValueError."""
    with pytest.raises(ValueError, match="zero energy spread"):
        dos._calculate_DOS(np.zeros(5), np.zeros(5))


def test_calculate_dos_rejects_non_positive_coefficient() -> None:
    """A negative DOS fits to a non-positive coefficient, which raises."""
    delta_E_J = np.linspace(1e-23, 1e-20, 10)

    with pytest.raises(ValueError, match="non-positive"):
        dos._calculate_DOS(delta_E_J, -np.sqrt(delta_E_J))


def test_clean_dos_values_filters() -> None:
    """Drops points at or below the edge, and any DOS not above min_dos.

    energy_window is only used to phrase the error message - the windowing
    itself has already happened by the time this is called.
    """
    delta_E_ev = np.array([-0.05, 0.02, 0.05, np.nan, 0.08, 0.11])
    density = np.array([1.0, 2.0, 0.0, 4.0, 5.0, 6.0])

    kept_E, kept_dos = dos._clean_dos_values(delta_E_ev, density, 0.0, 0.1)

    np.testing.assert_allclose(kept_E, [0.02, 0.08, 0.11])
    np.testing.assert_allclose(kept_dos, [2.0, 5.0, 6.0])


def test_clean_dos_values_requires_three_points() -> None:
    """Fewer than three survivors cannot support a fit, and the message says so."""
    delta_E_ev = np.array([0.01, 0.02, -0.03])
    density = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="usable DOS points"):
        dos._clean_dos_values(delta_E_ev, density, 0.0, 0.1)


def test_compute_dos_reference_value(dos_result: DOSResult) -> None:
    """The DOS effective mass of the reference data.

    final_result is the geometric average of Crovetto's equation (S6), so both
    per-carrier masses are pinned alongside it: 0.0727540 on its own is below
    the 0.12 lower bound of the FOM's sampled range, and sqrt(m_e * m_h) is
    not.
    """
    assert dos_result.em_electrons is not None
    assert dos_result.em_holes is not None
    assert dos_result.em_electrons.m_eff_rel == pytest.approx(0.0727540, rel=1e-6)
    assert dos_result.em_holes.m_eff_rel == pytest.approx(0.7758652, rel=1e-6)
    assert dos_result.final_result == pytest.approx(0.2375865, rel=1e-6)
    assert dos_result.cbm == pytest.approx(1.40177392, rel=1e-6)
    assert dos_result.cell_volume_m3 == pytest.approx(2.2340578e-28, rel=1e-6)
    assert dos_result.carrier == "electrons"


def test_compute_dos_final_result_is_the_geometric_average(
        dos_result: DOSResult
) -> None:
    """final_result is sqrt(m_e * m_h), equation (S6) of Crovetto 2024.

    The QFLS depends on the N_c N_v product, which goes as (m_e m_h)^(3/2), so
    the mass entering the FOM is the geometric average of the two rather than
    either one alone.
    """
    assert dos_result.em_electrons is not None
    assert dos_result.em_holes is not None

    expected = np.sqrt(
        dos_result.em_electrons.m_eff_rel * dos_result.em_holes.m_eff_rel
    )

    assert dos_result.final_result == pytest.approx(expected, rel=1e-12)


def test_compute_dos_poor_fit_is_visible(dos_result: DOSResult) -> None:
    """The reference DOS is too coarse to resolve the mass, and says so.

    R^2 = 0.671 against a 0.80 threshold, on 7 points against a 10-point
    threshold. This is a property of the shipped data, and it must stay visible
    rather than being quietly rounded off.
    """
    assert dos_result.fit_quality_e is not None
    assert dos_result.em_electrons is not None
    assert dos_result.fit_quality_e < dos.MIN_DOS_FIT_R2
    assert dos_result.em_electrons.n_points < dos.MIN_DOS_FIT_POINTS


def test_compute_dos_holes_carrier(dos_vasprun: Path) -> None:
    """Selecting holes returns the hole mass, and em_result follows the selection."""
    result = dos.compute_dos(
        dos_vasprun=str(dos_vasprun), carrier="holes", energy_window=0.1
    )

    assert result.carrier == "holes"
    assert result.em_result is result.em_holes
    assert result.em_holes is not None
    # Holes in Cu2GeS3 are far heavier than electrons.
    assert result.em_holes.m_eff_rel > 0.5
    # carrier picks which fit em_result exposes, not final_result: that stays
    # the geometric average either way, so both selections agree on it.
    assert result.em_electrons is not None
    assert result.final_result == pytest.approx(
        np.sqrt(result.em_electrons.m_eff_rel * result.em_holes.m_eff_rel), rel=1e-12
    )


def test_compute_dos_rejects_bad_carrier(dos_vasprun: Path) -> None:
    """An unknown carrier name raises a ValueError."""
    with pytest.raises(ValueError):
        dos.compute_dos(dos_vasprun=str(dos_vasprun), carrier="phonons")


def test_compute_dos_m_eff_override(dos_vasprun: Path) -> None:
    """An explicit mass bypasses the fit - the documented escape hatch for coarse data."""
    result = dos.compute_dos(
        dos_vasprun=str(dos_vasprun), m_eff=0.25, carrier="electrons", energy_window=0.1
    )

    assert result.final_result == pytest.approx(0.25, rel=1e-12)


def test_get_dos_effective_mass_matches_compute_dos(
        dos_vasprun: Path, dos_result: DOSResult
) -> None:
    """The single-carrier entry point agrees with compute_dos, and E_c aliases E_edge."""
    single = dos.get_dos_effective_mass(
        dos_vasprun=str(dos_vasprun), carrier="electrons", energy_window=0.1
    )

    assert dos_result.em_electrons is not None
    assert single.m_eff_rel == pytest.approx(
        dos_result.em_electrons.m_eff_rel, rel=1e-12
    )
    assert single.E_c == single.E_edge


def test_dos_mass_windows_widen_the_fit(dos_vasprun: Path) -> None:
    """Wider windows admit more points; reached through the module to avoid collection."""
    rows = dos.test_dos_mass_windows(
        str(dos_vasprun), carrier="electrons", windows=(0.05, 0.1, 0.2)
    )

    assert len(rows) == 3


def test_dos_result_str_contains_mass(dos_result: DOSResult) -> None:
    """The rendered summary names the fitted mass and the carrier."""
    rendered = str(dos_result)

    assert "0.072754" in rendered
    assert "electrons" in rendered.lower()


def test_generate_local_kpoints_mesh_size() -> None:
    """A 3x3x3 local mesh around Gamma is 27 points, all within delta of the centre."""
    kpoints = dos._generate_local_kpoints(np.array([0.0, 0.0, 0.0]), (3, 3, 3), 0.01)

    assert kpoints.num_kpts == 27
    assert np.max(np.abs(np.asarray(kpoints.kpts))) <= 0.01 + 1e-12


@requires_potcars
def test_write_eff_mass_creates_inputs(relax_dir: Path, tmp_path: Path) -> None:
    """write_eff_mass writes the four VASP input files into the target folder."""
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


def test_castep_dos_fixture_regenerates_byte_identically(castep_dos_bands: Path) -> None:
    """The committed toy.bands is exactly what tests/castep_fixtures.py emits."""
    assert castep_dos_bands.read_text() == castep_fixtures.dos_bands_text()


def test_castep_dos_recovers_electron_mass(castep_dos_bands: Path) -> None:
    """The parabolic fixture returns the electron mass it encodes.

    The tolerance budgets the ~bin_width offset of the histogram band edge;
    a wrong bin-width division or missing spin factor moves the mass by
    0.01^(2/3) or 2^(2/3), far outside it.
    """
    result = dos.get_dos_effective_mass(
        dos_vasprun=str(castep_dos_bands),
        carrier="electrons",
        energy_window=0.25,
        code="castep",
    )

    assert result.m_eff_rel == pytest.approx(castep_fixtures.DOS_M_ELECTRON, rel=0.05)
    # The histogram band-edge offset costs a little R^2; 0.95 still rules out
    # any of the unit mistakes above, which drive it negative.
    assert result.fit_quality > 0.95


def test_castep_dos_recovers_hole_mass(castep_dos_bands: Path) -> None:
    """The parabolic fixture returns the hole mass it encodes."""
    result = dos.get_dos_effective_mass(
        dos_vasprun=str(castep_dos_bands),
        carrier="holes",
        energy_window=0.25,
        code="castep",
    )

    assert result.m_eff_rel == pytest.approx(castep_fixtures.DOS_M_HOLE, rel=0.05)
    assert result.fit_quality > 0.95


def test_castep_compute_dos_summary(castep_dos_bands: Path) -> None:
    """compute_dos reads gap, volume and both masses from the .bands file."""
    result = dos.compute_dos(
        dos_vasprun=str(castep_dos_bands),
        carrier="electrons",
        energy_window=0.25,
        code="castep",
    )

    assert result.cbm == pytest.approx(castep_fixtures.DOS_GAP_EV, abs=0.03)
    assert result.vbm == 0.0
    assert result.cell_volume_m3 == pytest.approx(castep_fixtures.VOLUME_M3, rel=1e-6)
    assert result.em_electrons is not None
    assert result.em_electrons.m_eff_rel == pytest.approx(
        castep_fixtures.DOS_M_ELECTRON, rel=0.05
    )
    assert result.em_holes is not None
    assert result.em_holes.m_eff_rel == pytest.approx(castep_fixtures.DOS_M_HOLE, rel=0.05)
    # The fixture encodes both masses analytically, so equation (S6) has a
    # closed-form answer here: sqrt(0.30 * 1.20) = 0.6.
    assert result.final_result == pytest.approx(
        np.sqrt(castep_fixtures.DOS_M_ELECTRON * castep_fixtures.DOS_M_HOLE), rel=0.05
    )


def test_castep_compute_dos_m_eff_override(castep_dos_bands: Path) -> None:
    """The explicit-mass escape hatch works on the CASTEP path too."""
    result = dos.compute_dos(
        dos_vasprun=str(castep_dos_bands), m_eff=0.25, code="castep"
    )

    assert result.final_result == pytest.approx(0.25, rel=1e-12)


def test_castep_dos_mass_windows_sweep(castep_dos_bands: Path) -> None:
    """The window sweep threads code= through to each fit."""
    rows = dos.test_dos_mass_windows(
        str(castep_dos_bands), carrier="electrons", windows=(0.1, 0.2), code="castep"
    )

    assert len(rows) == 2


def test_castep_dos_unknown_code_raises(castep_dos_bands: Path) -> None:
    """An unsupported code name raises instead of falling back to VASP."""
    with pytest.raises(ValueError, match="banana"):
        dos.get_dos_effective_mass(dos_vasprun=str(castep_dos_bands), code="banana")


def test_castep_write_eff_mass_writes_inputs(tmp_path: Path) -> None:
    """write_eff_mass(code="castep") writes the spectral local-mesh inputs."""
    structure = Structure(Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]])

    dos.write_eff_mass(
        k0_frac=np.array([0.0, 0.0, 0.0]),
        structure=structure,
        functional="PBE",
        encut=450,
        folder=str(tmp_path / "eff_mass"),
        mesh=(5, 5, 5),
        delta=0.01,
        code="castep",
    )

    cell_text = (tmp_path / "eff_mass" / "Si.cell").read_text()
    param_text = (tmp_path / "eff_mass" / "Si.param").read_text()

    assert "%block spectral_kpoint_list" in cell_text
    # A 5x5x5 mesh is 125 k-point rows inside the block.
    block = cell_text.split("%block spectral_kpoint_list")[1]
    assert len(block.split("%endblock")[0].strip().splitlines()) == 125
    assert "Spectral" in param_text
    assert "BandStructure" in param_text
    assert "450" in param_text


def test_write_eff_mass_unknown_code_raises(tmp_path: Path) -> None:
    """An unsupported code raises before anything is written."""
    structure = Structure(Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="banana"):
        dos.write_eff_mass(
            k0_frac=np.array([0.0, 0.0, 0.0]),
            structure=structure,
            functional="PBE",
            encut=450,
            folder=str(tmp_path),
            code="banana",
        )


def test_write_local_kpoints_writes_file(tmp_path: Path) -> None:
    """write_local_kpoints should produce a KPOINTS file in the folder."""
    dos.write_local_kpoints(str(tmp_path), np.array([0.0, 0.0, 0.0]), (3, 3, 3), 0.01)

    kpoints = Kpoints.from_file(str(tmp_path / "KPOINTS"))

    assert (tmp_path / "KPOINTS").is_file()
    assert kpoints.num_kpts == 27
    assert len(kpoints.kpts) == 27
    assert kpoints.style is Kpoints.supported_modes.Reciprocal
