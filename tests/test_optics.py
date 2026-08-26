"""Dielectric function -> absorption -> SLME.

The two strongest tests here regenerate absorption.dat and n_real.dat from the
committed vasprun.xml and compare against the committed .dat files. Those are
provenance-verified reference data rather than a snapshot of whatever the code
happens to emit today, which is the distinction CONTRIBUTING asks for.
"""

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import scipy.constants as sc
from numpy.typing import NDArray
from scipy.optimize import brentq

import castep_fixtures
import solphin.optics as optics


# --- the dielectric tensor -------------------------------------------------


def test_calc_dielectric_returns_five_tuple(
        dielectric: tuple[float, NDArray, NDArray, NDArray, NDArray]
) -> None:
    """calc_dielectric returns five values: eps_inf, tensor, eps_full, eps_imag, energies."""
    assert len(dielectric) == 5

    eps_inf, tensor, eps_full, eps_imag, energies = dielectric

    assert np.ndim(eps_inf) == 0
    assert np.shape(tensor) == (3, 3)
    assert len(energies) == len(eps_full) == len(eps_imag)


def test_calc_dielectric_reference_values(
        dielectric: tuple[float, NDArray, NDArray, NDArray, NDArray]
) -> None:
    """High-frequency dielectric constant of Cu2GeS3 from the committed run."""
    eps_inf, tensor, *_ = dielectric

    assert eps_inf == pytest.approx(6.109433, rel=1e-6)
    np.testing.assert_allclose(
        np.diagonal(np.asarray(tensor, dtype=float)),
        [6.0546, 6.1405, 6.1332],
        rtol=1e-4,
    )


def test_eps_inf_is_tensor_trace_average(
        dielectric: tuple[float, NDArray, NDArray, NDArray, NDArray]
) -> None:
    """The scalar eps_inf is the mean of the tensor's diagonal."""
    eps_inf, tensor, *_ = dielectric

    expected = np.mean(np.diagonal(np.asarray(tensor, dtype=float)))

    assert eps_inf == pytest.approx(expected, rel=1e-12)


# --- absorption from the dielectric function -------------------------------


def test_calc_absorption_keys(dielectric: tuple[float, NDArray, NDArray, NDArray, NDArray]) -> None:
    """The optics dict carries the six documented keys, each on the full energy grid."""
    _, _, eps_full, _, energies = dielectric

    data = optics.calc_absorption(eps_full, energies)

    expected = {"eps_real", "eps_imag", "n_real", "n_imag", "loss", "absorption"}
    assert set(data) == expected
    assert all(len(data[key]) == len(energies) for key in expected)


def test_calc_absorption_on_lossless_dielectric() -> None:
    """For real, isotropic eps: n = sqrt(eps) and a transparent medium absorbs nothing."""
    energies = np.linspace(0.1, 5.0, 50)
    eps_full = np.zeros((50, 3, 3), dtype=complex)
    for axis in range(3):
        eps_full[:, axis, axis] = 4.0 + 0j

    data = optics.calc_absorption(eps_full, energies)

    np.testing.assert_allclose(data["n_real"], 2.0, rtol=1e-12)
    np.testing.assert_allclose(data["n_imag"], 0.0, atol=1e-12)
    np.testing.assert_allclose(data["absorption"], 0.0, atol=1e-12)


def test_absorption_non_negative(
        dielectric: tuple[float, NDArray, NDArray, NDArray, NDArray]
) -> None:
    """Absorption is never negative and the real refractive index stays positive."""
    _, _, eps_full, _, energies = dielectric

    data = optics.calc_absorption(eps_full, energies)

    assert np.all(np.asarray(data["absorption"]) >= 0)
    assert np.all(np.asarray(data["n_real"]) > 0)


# --- blackbody helpers -----------------------------------------------------


def test_bb_per_eV_peak_position() -> None:
    """The photon-flux blackbody E^2/(exp(E/kT)-1) peaks where 2(1-e^-x) = x."""
    x = brentq(lambda x: 2 * (1 - np.exp(-x)) - x, 1e-6, 10)
    kT_eV = sc.k * optics._T / sc.e

    energies = np.linspace(1e-4, 0.5, 500_000)
    peak = energies[np.argmax(optics._bb_per_eV(energies))]

    assert peak == pytest.approx(x * kT_eV, rel=1e-3)


def test_bb_representations_agree() -> None:
    """Total photon flux is the same whether integrated over dE or d(lambda).

    Tolerance is loose because both integrals truncate an infinite tail on a
    finite grid, not because the identity is approximate.
    """
    energies = np.linspace(1e-3, 3.0, 200_000)
    flux_from_energy = np.trapezoid(optics._bb_per_eV(energies), energies)

    wavelengths = np.linspace(1e-7, 2e-4, 200_000)
    flux_from_wavelength = np.trapezoid(optics._bb_per_wl(wavelengths), wavelengths)

    assert flux_from_energy == pytest.approx(flux_from_wavelength, rel=0.02)


def test_calc_incident_power_am15() -> None:
    """The AM1.5 spectrum optics uses must still integrate to about 1000 W m^-2.

    Note this reads pymatgen's am1.5G.dat, not the bundled ASTMG173.csv that
    db_fom.load_spectrum uses - a different source on a different grid.
    """
    wavelengths, irradiance, use_slme = optics._spectrum_select("AM1.5")

    power = optics._calc_incident_power(irradiance, wavelengths)

    assert use_slme is True
    assert power == pytest.approx(1000.0, rel=0.05)


def test_spectrum_select_falls_back_to_bundled_resources() -> None:
    """Anything other than AM1.5 routes through db_fom.load_spectrum instead."""
    wavelengths, irradiance, use_slme = optics._spectrum_select("Red LED")

    assert use_slme is False
    assert len(wavelengths) == len(irradiance) > 0


def test_convert_spec_photon_flux() -> None:
    """Photon flux = irradiance * lambda / hc, with lambda returned in metres."""
    wavelengths_nm = np.array([400.0, 800.0])
    irradiance = np.array([1.0, 1.0])

    wavelengths_m, flux = optics._convert_spec(wavelengths_nm, irradiance)

    np.testing.assert_allclose(wavelengths_m, [4e-7, 8e-7], rtol=1e-12)
    # Twice the wavelength means half the energy per photon, so twice the flux.
    assert flux[1] == pytest.approx(2 * flux[0], rel=1e-12)


# --- writing the .dat files ------------------------------------------------


def test_fixture_copy_is_write_protected(opt_dir: Path) -> None:
    """The fixture data is read-only, so a stray write fails where the bug is.

    Pinned because the protection lives in one `chmod` in conftest.py, and
    losing it - a `copyfile` quietly becoming a `copy2`, say - would show up
    nowhere else until a test corrupted an input another test was relying on.
    """
    with pytest.raises(PermissionError):
        (opt_dir / "vasprun.xml").open("a")


def test_generate_absorption_reproduces_committed_file(tmp_opt_dir: Path, opt_dir: Path) -> None:
    """Regenerating from vasprun.xml must reproduce the committed absorption.dat."""
    optics.generate_absorption(str(tmp_opt_dir))

    regenerated = np.loadtxt(tmp_opt_dir / "absorption.dat", skiprows=1)
    committed = np.loadtxt(opt_dir / "absorption.dat", skiprows=1)

    np.testing.assert_allclose(regenerated, committed, rtol=1e-10)


def test_generate_n_real_reproduces_committed_file(tmp_opt_dir: Path, opt_dir: Path) -> None:
    """Same for n_real.dat, which make_blank_plot reads back off disk."""
    optics.generate_n_real(str(tmp_opt_dir))

    regenerated = np.loadtxt(tmp_opt_dir / "n_real.dat", skiprows=1)
    committed = np.loadtxt(opt_dir / "n_real.dat", skiprows=1)

    np.testing.assert_allclose(regenerated, committed, rtol=1e-10)


def test_print_absorption_file_converts_to_per_cm(tmp_path: Path) -> None:
    """calc_absorption works in m^-1; the file is documented as cm^-1."""
    energies = np.array([1.0, 2.0, 3.0])
    data = {"absorption": np.array([1.0e7, 2.0e7, 3.0e7])}  # m^-1

    optics.print_absorption_file(data, energies, tmp_path)

    written = np.loadtxt(tmp_path / "absorption.dat", skiprows=1)
    header = (tmp_path / "absorption.dat").read_text().splitlines()[0]

    assert header.startswith("#")
    assert "cm^-1" in header
    np.testing.assert_allclose(written[:, 1], [1.0e5, 2.0e5, 3.0e5], rtol=1e-12)


def test_generated_dat_files_have_one_header_line(tmp_opt_dir: Path) -> None:
    """spectral._load_absorption assumes exactly one '#' line plus data."""
    optics.generate_absorption(str(tmp_opt_dir))

    lines = (tmp_opt_dir / "absorption.dat").read_text().splitlines()

    assert lines[0].startswith("#")
    assert not lines[1].startswith("#")


# --- efficiency and plotting ----------------------------------------------


def test_power_efficiency_bounded(tmp_opt_dir: Path) -> None:
    """Efficiency is a fraction, and a thicker absorber never collects less."""
    optics.generate_absorption(str(tmp_opt_dir))
    optics.generate_n_real(str(tmp_opt_dir))
    energy_abs, alpha_cm = np.loadtxt(
        tmp_opt_dir / "absorption.dat", skiprows=1, unpack=True
    )
    n_real = np.loadtxt(tmp_opt_dir / "n_real.dat", skiprows=1, usecols=1)
    alpha_m = alpha_cm * 100.0
    absorbance = np.ones_like(energy_abs)

    efficiencies = [
        optics.power_efficiency(absorbance, energy_abs, n_real, alpha_m, d)
        for d in (1e-8, 1e-6, 1e-4)
    ]

    assert all(0.0 <= eff <= 1.0 for eff in efficiencies)
    assert np.all(np.diff(efficiencies) >= 0)


def test_make_blank_plot_smoke(tmp_opt_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A short explicit thickness_range - the default sweeps 80 points twice over.

    Runs inside tmp_path because save=True writes slme.png into the cwd.
    """
    import matplotlib.pyplot as plt

    optics.generate_absorption(str(tmp_opt_dir))
    optics.generate_n_real(str(tmp_opt_dir))
    monkeypatch.chdir(tmp_opt_dir)

    optics.make_blank_plot(
        str(tmp_opt_dir),
        direct_gap=1.3901,
        indirect_gap=1.3901,
        spectrum_type="AM1.5",
        Qi=1.0,
        n=3.5,
        thickness_range=np.array([1e-7, 1e-6, 1e-5]),
        save=False,
    )

    assert plt.get_fignums()


def test_plot_absorption_smoke(tmp_opt_dir: Path) -> None:
    """Plotting the absorption spectrum produces a matplotlib figure."""
    import matplotlib.pyplot as plt

    optics.generate_absorption(str(tmp_opt_dir))

    optics.plot_absorption(optics_directory=str(tmp_opt_dir))

    assert plt.get_fignums()


def test_plot_absorption_saves_into_out_directory(tmp_opt_dir: Path, tmp_path: Path) -> None:
    """save=True writes absorption.png into out_directory rather than the cwd."""
    optics.generate_absorption(str(tmp_opt_dir))
    figures = tmp_path / "figures"
    figures.mkdir()

    optics.plot_absorption(optics_directory=str(tmp_opt_dir), save=True, out_directory=figures)

    assert (figures / "absorption.png").is_file()


def test_make_blank_plot_passes_out_directory_through(tmp_opt_dir: Path, tmp_path: Path) -> None:
    """out_directory reaches plot_blank, which is what actually writes slme.png."""
    optics.generate_absorption(str(tmp_opt_dir))
    optics.generate_n_real(str(tmp_opt_dir))
    figures = tmp_path / "figures"
    figures.mkdir()

    optics.make_blank_plot(
        str(tmp_opt_dir),
        direct_gap=1.3901,
        indirect_gap=1.3901,
        thickness_range=np.array([1e-7, 1e-6, 1e-5]),
        save=True,
        out_directory=figures,
    )

    assert (figures / "slme.png").is_file()


# --- CASTEP ----------------------------------------------------------------


def test_castep_epsilon_fixtures_regenerate_byte_identically(
        castep_opt_dir: Path, castep_opt_tensor_dir: Path
) -> None:
    """The committed OptaDOS fixtures are exactly what castep_fixtures emits."""
    poly = (castep_opt_dir / "toy_epsilon.dat").read_text()
    tensor = (castep_opt_tensor_dir / "toy_epsilon.dat").read_text()

    assert poly == castep_fixtures.epsilon_poly_text()
    assert tensor == castep_fixtures.epsilon_tensor_text()


def test_castep_calc_dielectric_returns_five_tuple(
        castep_dielectric: tuple[float, NDArray, NDArray, NDArray, NDArray]
) -> None:
    """The CASTEP path honours the same 5-tuple contract as the VASP path."""
    assert len(castep_dielectric) == 5

    eps_inf, tensor, eps_full, eps_imag, energies = castep_dielectric

    assert np.ndim(eps_inf) == 0
    assert np.shape(tensor) == (3, 3)
    assert eps_full.shape == eps_imag.shape == (len(energies), 3, 3)


def test_castep_calc_dielectric_lorentz_static_limit(
        castep_dielectric: tuple[float, NDArray, NDArray, NDArray, NDArray]
) -> None:
    """The Lorentz fixture has eps(0) = 1 + S = 6 exactly, isotropic."""
    eps_inf, tensor, _, _, _ = castep_dielectric

    assert eps_inf == pytest.approx(6.0, rel=1e-10)
    np.testing.assert_allclose(tensor, 6.0 * np.eye(3), rtol=1e-10)


def test_castep_calc_dielectric_matches_analytic_model(
        castep_dielectric: tuple[float, NDArray, NDArray, NDArray, NDArray]
) -> None:
    """Every parsed value reproduces the Lorentz oscillator it was built from."""
    _, _, eps_full, _, energies = castep_dielectric

    expected = np.array([castep_fixtures.lorentz_epsilon(e) for e in energies])

    np.testing.assert_allclose(eps_full[:, 0, 0], expected, rtol=1e-8)


def test_castep_absorption_static_refractive_index(
        castep_dielectric: tuple[float, NDArray, NDArray, NDArray, NDArray]
) -> None:
    """n(0) = sqrt(eps(0)) = sqrt(6) for the Lorentz fixture."""
    _, _, eps_full, _, energies = castep_dielectric

    data = optics.calc_absorption(eps_full, energies)

    assert data["n_real"][0] == pytest.approx(np.sqrt(6.0), rel=1e-10)


def test_castep_tensor_geometry_lossless(castep_opt_tensor_dir: Path) -> None:
    """The tensor fixture parses to diag(2, 3, 4) and zero absorption throughout.

    The CASTEP twin of test_calc_absorption_on_lossless_dielectric, but end to
    end through the OptaDOS parser: constant real eigenvalues mean
    n = (sqrt(2) + sqrt(3) + sqrt(4)) / 3 everywhere and no extinction.
    """
    eps_inf, tensor, eps_full, _, energies = optics.calc_dielectric(
        str(castep_opt_tensor_dir / "toy_epsilon.dat"), code="castep"
    )

    assert eps_inf == pytest.approx(3.0, rel=1e-10)
    np.testing.assert_allclose(tensor, np.diag([2.0, 3.0, 4.0]), atol=1e-12)

    data = optics.calc_absorption(eps_full, energies)
    expected_n = (np.sqrt(2.0) + np.sqrt(3.0) + np.sqrt(4.0)) / 3.0

    np.testing.assert_allclose(data["n_real"], expected_n, rtol=1e-10)
    np.testing.assert_allclose(data["absorption"], 0.0, atol=1e-8)


def test_castep_generate_absorption_writes_readable_file(
        castep_opt_dir: Path, tmp_path: Path
) -> None:
    """generate_absorption(code="castep") writes the standard absorption.dat."""
    optics.generate_absorption(
        str(castep_opt_dir), out_directory=tmp_path, code="castep"
    )

    written = tmp_path / "absorption.dat"
    assert written.is_file()
    data = np.loadtxt(written, skiprows=1)
    assert data.shape[1] == 2
    # alpha(0) = 0: the Lorentz model is lossless at zero energy.
    assert data[0, 1] == pytest.approx(0.0, abs=1e-8)


def test_castep_generate_n_real_writes_readable_file(
        castep_opt_dir: Path, tmp_path: Path
) -> None:
    """generate_n_real(code="castep") writes n_real.dat with n(0) = sqrt(6)."""
    optics.generate_n_real(str(castep_opt_dir), out_directory=tmp_path, code="castep")

    data = np.loadtxt(tmp_path / "n_real.dat", skiprows=1)
    assert data[0, 1] == pytest.approx(np.sqrt(6.0), rel=1e-10)


def test_castep_plot_absorption_smoke(castep_opt_dir: Path, tmp_path: Path) -> None:
    """plot_absorption(code="castep") renders and saves without error."""
    optics.plot_absorption(
        str(castep_opt_dir), save=True, out_directory=tmp_path, code="castep"
    )

    assert (tmp_path / "absorption.png").is_file()


def test_find_epsilon_file_explicit_seedname(castep_opt_dir: Path) -> None:
    """An explicit seedname resolves <seed>_epsilon.dat directly."""
    path = optics._find_epsilon_file(castep_opt_dir, "toy")

    assert path.name == "toy_epsilon.dat"


def test_find_epsilon_file_missing_raises(tmp_path: Path) -> None:
    """An empty directory raises FileNotFoundError, not a silent glob miss."""
    with pytest.raises(FileNotFoundError):
        optics._find_epsilon_file(tmp_path, None)


def test_find_epsilon_file_ambiguous_raises(
        castep_opt_dir: Path, tmp_path: Path
) -> None:
    """Two epsilon files without a seedname raise naming both candidates."""
    for seed in ("a", "b"):
        shutil.copyfile(
            castep_opt_dir / "toy_epsilon.dat", tmp_path / f"{seed}_epsilon.dat"
        )

    with pytest.raises(ValueError, match="a_epsilon.dat.*b_epsilon.dat"):
        optics._find_epsilon_file(tmp_path, None)


def test_read_optados_epsilon_rejects_bad_block_count(tmp_path: Path) -> None:
    """A block count that is neither 1 nor 6 is a format error, not a guess."""
    two_blocks = "# header\n0.0 1.0 0.0\n\n# second\n0.0 2.0 0.0\n"
    bad = tmp_path / "bad_epsilon.dat"
    bad.write_text(two_blocks)

    with pytest.raises(ValueError, match="found 2"):
        optics._read_optados_epsilon(bad)


def test_read_optados_epsilon_rejects_unparseable_line(tmp_path: Path) -> None:
    """A malformed data line raises naming the line, not a numpy stack trace."""
    bad = tmp_path / "bad_epsilon.dat"
    bad.write_text("0.0 1.0 zero\n")

    with pytest.raises(ValueError, match="zero"):
        optics._read_optados_epsilon(bad)


def test_read_optados_epsilon_rejects_short_line(tmp_path: Path) -> None:
    """A data line with fewer than three columns is a format error."""
    bad = tmp_path / "bad_epsilon.dat"
    bad.write_text("0.0 1.0\n")

    with pytest.raises(ValueError, match="3 columns"):
        optics._read_optados_epsilon(bad)


def test_read_optados_epsilon_rejects_mismatched_tensor_grids(tmp_path: Path) -> None:
    """Six blocks whose energy grids disagree raise instead of mixing data."""
    blocks = []
    for index in range(6):
        energy = 0.0 if index < 5 else 1.0  # last block on a shifted grid
        blocks.append(f"# Component: 1 1\n{energy} 1.0 0.0\n")
    bad = tmp_path / "bad_epsilon.dat"
    bad.write_text("\n".join(blocks))

    with pytest.raises(ValueError, match="energy grid"):
        optics._read_optados_epsilon(bad)


def test_find_epsilon_file_explicit_seedname_missing_raises(tmp_path: Path) -> None:
    """A named seed with no epsilon file raises naming the expected path."""
    with pytest.raises(FileNotFoundError, match="ghost_epsilon.dat"):
        optics._find_epsilon_file(tmp_path, "ghost")


@pytest.mark.parametrize(
    "call",
    [
        lambda d: optics.calc_dielectric(d / "toy_epsilon.dat", code="banana"),
        lambda d: optics.generate_absorption(str(d), code="banana"),
    ],
)
def test_optics_unknown_code_raises(
        castep_opt_dir: Path, call: Callable[[Path], object]
) -> None:
    """An unsupported code name raises instead of falling back to VASP."""
    with pytest.raises(ValueError, match="banana"):
        call(castep_opt_dir)
