"""Dielectric function -> absorption -> SLME.

The two strongest tests here regenerate absorption.dat and n_real.dat from the
committed vasprun.xml and compare against the committed .dat files. Those are
provenance-verified reference data rather than a snapshot of whatever the code
happens to emit today, which is the distinction CONTRIBUTING asks for.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.constants as sc
from numpy.typing import NDArray
from scipy.optimize import brentq

import solphin.optics as optics

# --- the dielectric tensor -------------------------------------------------


def test_calc_dielectric_returns_five_tuple(
        dielectric: tuple[float, NDArray, NDArray, NDArray, NDArray]
) -> None:
    """The docstring documents two return values; there are five."""
    assert len(dielectric) == 5

    eps_inf, tensor, eps_full, eps_imag, energies = dielectric

    assert np.ndim(eps_inf) == 0
    assert np.shape(tensor) == (3, 3)
    assert len(energies) == len(eps_full) == len(eps_imag)


def test_calc_dielectric_tutorial_values(
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
    import matplotlib.pyplot as plt

    optics.generate_absorption(str(tmp_opt_dir))

    optics.plot_absorption(optics_directory=str(tmp_opt_dir))

    assert plt.get_fignums()
