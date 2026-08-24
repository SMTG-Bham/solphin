"""Agg smoke tests for the plotting layer, plus the one algebraic identity in it.

These are thin on purpose - every number they draw is already checked in
test_db_fom.py and test_pv_fom.py, so the tests only confirm the wrappers run,
draw onto the axes they were handed, and return what they promise.

The four *_interactive entry points are excluded. They set
fig.canvas.header_visible, which only exists under ipympl, and ipympl is neither
declared in pyproject.toml nor installed. They need a live notebook kernel.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from numpy.typing import NDArray

import solphin.db_plots as db_plots
import solphin.dos as dos
import solphin.final_results as final_results

TCELL = 300.0
E_GAP = 1.34

# Crovetto's defaults, as set in tutorial cell 41.
FOM_ARGS = dict(alpha=1.3e5,
                tau=1e-6,
                sigma=1.5,
                dos_mass=0.073,
                dop_density=1e10,
                epsilon=6.11,
                mu=1e6)


# --- db_plots --------------------------------------------------------------


def test_photons_above_bandgap_plot_on_axis(photon_spectrum: NDArray) -> None:
    """Given an axis, it draws there and hands the axis back."""
    _, ax = plt.subplots()

    returned = db_plots.photons_above_bandgap_plot(photon_spectrum, E_GAP, ax=ax)

    assert returned is ax
    assert ax.lines


def test_photons_above_bandgap_plot_standalone(photon_spectrum: NDArray) -> None:
    """Without an axis it draws on the current figure and returns None."""
    result = db_plots.photons_above_bandgap_plot(photon_spectrum, E_GAP)

    assert result is None
    assert plt.get_fignums()


def test_iv_curve_plot(photon_spectrum: NDArray) -> None:
    """The IV curve draws onto the current axes."""
    db_plots.iv_curve_plot(photon_spectrum, E_GAP, TCELL)

    assert plt.gca().lines


def test_iv_curve_plot_power_variant(photon_spectrum: NDArray) -> None:
    """power=True switches the plot to the power-voltage curve."""
    db_plots.iv_curve_plot(photon_spectrum, E_GAP, TCELL, power=True)

    assert plt.gca().get_ylabel().startswith("Power")


def test_iv_pv_curve_plot(photon_spectrum: NDArray) -> None:
    """Current and power each land on the axis they were handed."""
    _, ax1 = plt.subplots()
    _, ax2 = plt.subplots()

    db_plots.iv_pv_curve_plot(photon_spectrum, E_GAP, TCELL, ax1=ax1, ax2=ax2)

    assert ax1.lines
    assert ax2.lines


def test_sq_limit_plot(photon_spectrum: NDArray) -> None:
    """The SQ-limit curve draws onto the supplied axis."""
    _, ax = plt.subplots()

    db_plots.sq_limit_plot(photon_spectrum, E_GAP, TCELL, ax=ax)

    assert ax.lines


def test_plot_db_combined(photon_spectrum: NDArray) -> None:
    """The combined three-panel figure builds without pre-made axes."""
    db_plots.plot_db_combined(photon_spectrum, E_GAP, TCELL, "AM1.5")

    assert plt.get_fignums()


# --- dos / optics plotting -------------------------------------------------


def test_plot_dos_smoke(opt_dir: Path) -> None:
    """Plotting the DOS from a vasprun produces a figure."""
    dos.plot_dos(filename=str(opt_dir / "vasprun.xml"), xmin=-3, xmax=4, gaussian=0.05)

    assert plt.get_fignums()


def test_plot_dos_saves_into_out_directory(opt_dir: Path, tmp_path: Path) -> None:
    """save=True writes dos.png into out_directory rather than the cwd."""
    dos.plot_dos(filename=str(opt_dir / "vasprun.xml"), save=True, out_directory=tmp_path)

    assert (tmp_path / "dos.png").is_file()


def test_plot_dos_castep_smoke(castep_band_dir: Path) -> None:
    """Plotting the DOS from a CASTEP .bands file produces a figure."""
    dos.plot_dos(
        filename=str(castep_band_dir / "toy.bands"), xmin=-3, xmax=4, code="castep"
    )

    assert plt.get_fignums()


# --- final_results ---------------------------------------------------------


def test_sq_relative_identity(photon_spectrum: NDArray) -> None:
    """SQ_relative is exactly 100 * FOM_efficiency / SQ.

    Both are formed from the same SQ_eff, which cancels - so this holds for any
    spectrum and any parameter set, and pins the meaning of the three returns.
    """
    sq, sq_relative, fom_efficiency = final_results.SQ_relative_FOM_PV_efficiency(
        E_GAP, photon_spectrum, FOM_ARGS["alpha"], FOM_ARGS["tau"], FOM_ARGS["sigma"],
        FOM_ARGS["dos_mass"], FOM_ARGS["dop_density"], FOM_ARGS["epsilon"],
        FOM_ARGS["mu"], TCELL,
    )

    assert sq_relative == pytest.approx(100 * fom_efficiency / sq, rel=1e-12)
    assert 0 < fom_efficiency < sq


def test_sq_relative_is_spectrum_independent(photon_spectrum: NDArray) -> None:
    """The SQ-relative ratio depends only on the material, not the illumination."""
    import solphin.db_fom as db_fom

    other = db_fom.convert_spectrum(db_fom.load_spectrum("White LED"))

    args = (FOM_ARGS["alpha"], FOM_ARGS["tau"], FOM_ARGS["sigma"], FOM_ARGS["dos_mass"],
            FOM_ARGS["dop_density"], FOM_ARGS["epsilon"], FOM_ARGS["mu"], TCELL)
    _, from_am15, _ = final_results.SQ_relative_FOM_PV_efficiency(
        E_GAP, photon_spectrum, *args
    )
    _, from_led, _ = final_results.SQ_relative_FOM_PV_efficiency(E_GAP, other, *args)

    assert from_am15 == pytest.approx(from_led, rel=1e-12)


def test_plot_fom_three_panels(photon_spectrum: NDArray) -> None:
    """plot_FOM draws a curve in each of the three panels."""
    fig, axes = plt.subplots(1, 3)

    final_results.plot_FOM(
        fig, list(axes), E_GAP, photon_spectrum, FOM_ARGS["alpha"], FOM_ARGS["tau"],
        FOM_ARGS["sigma"], FOM_ARGS["dos_mass"], FOM_ARGS["dop_density"],
        FOM_ARGS["epsilon"], FOM_ARGS["mu"], TCELL,
        dop_range=(1e10, 1e14), tau_range=(1e-9, 1e-6), mu_range=(1e0, 1e3),
    )

    assert all(ax.lines for ax in axes)


@pytest.mark.xfail(
    strict=True,
    reason=(
            "final_results.py:506 appends the whole (SQ, SQ_relative, FOM_efficiency) "
            "tuple instead of indexing one element the way plot_FOM does, so "
            "matplotlib unpacks each row into three lines and the axis labelled "
            "'PV efficiency' carries the SQ limit and the SQ-relative ratio too"
    ),
)
def test_mobility_plot_draws_one_line_per_lifetime(photon_spectrum: NDArray) -> None:
    """Two lifetimes should give two curves, not two curves' worth of columns."""
    final_results.mobility_plot(
        E_GAP, photon_spectrum, FOM_ARGS["alpha"], FOM_ARGS["sigma"],
        FOM_ARGS["dos_mass"], FOM_ARGS["epsilon"],
        mob_min=0, mob_max=2, lifetime_min=-9, lifetime_max=-7, step=1, Tcell=TCELL,
    )

    assert len(plt.gca().lines) == 2
