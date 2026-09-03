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
import numpy as np
import pytest
from numpy.typing import NDArray

import solphin.db_plots as db_plots
import solphin.dos as dos
import solphin.final_results as final_results

TCELL = 300.0
E_GAP = 1.34

# Crovetto's defaults, as set in tutorial cell 41. dos_mass is the geometric
# average sqrt(m_e * m_h) of equation (S6) for the reference data, which is what
# compute_dos now reports; the electron mass alone, 0.073, is below the 0.12
# lower bound of table 1 and the figure of merit refuses it.
FOM_ARGS = dict(alpha=1.3e5,
                tau=1e-6,
                sigma=1.5,
                dos_mass=0.2375865,
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


def test_mobility_plot_draws_one_line_per_lifetime(photon_spectrum: NDArray) -> None:
    """Each lifetime should give one curve, not one curve's worth of columns.

    The exponent bounds are inclusive, so lifetime_min=-9 with lifetime_max=-7
    sweeps 1e-9, 1e-8 and 1e-7 - three lifetimes, three lines.
    """
    final_results.mobility_plot(
        E_GAP, photon_spectrum, FOM_ARGS["alpha"], FOM_ARGS["sigma"],
        FOM_ARGS["dos_mass"], FOM_ARGS["epsilon"],
        mob_min=0, mob_max=2, lifetime_min=-9, lifetime_max=-7, step=1, Tcell=TCELL,
    )

    assert len(plt.gca().lines) == 3


# --- the table 1 guard on the sweep bounds --------------------------------
#
# The sweeps step one property across a range, so an endpoint outside table 1
# is refused for the same reason a single value is - and refused before the
# first point is drawn, so the message names the range rather than arriving
# fifty times from inside the figure of merit.


def test_plot_fom_refuses_a_sweep_outside_table_1(photon_spectrum: NDArray) -> None:
    """A lifetime range reaching past 1e3 s is not something the fit was trained on."""
    fig, axes = plt.subplots(1, 3)

    with pytest.raises(ValueError, match="tau"):
        final_results.plot_FOM(
            fig, list(axes), E_GAP, photon_spectrum, FOM_ARGS["alpha"], FOM_ARGS["tau"],
            FOM_ARGS["sigma"], FOM_ARGS["dos_mass"], FOM_ARGS["dop_density"],
            FOM_ARGS["epsilon"], FOM_ARGS["mu"], TCELL,
            tau_range=(1e-15, 1e5),
        )


def test_plot_fom_sweep_opt_out_still_draws(photon_spectrum: NDArray) -> None:
    """With the flag, the out-of-range span is warned about and then plotted."""
    fig, axes = plt.subplots(1, 3)

    with pytest.warns(UserWarning, match="tau"):
        final_results.plot_FOM(
            fig, list(axes), E_GAP, photon_spectrum, FOM_ARGS["alpha"], FOM_ARGS["tau"],
            FOM_ARGS["sigma"], FOM_ARGS["dos_mass"], FOM_ARGS["dop_density"],
            FOM_ARGS["epsilon"], FOM_ARGS["mu"], TCELL,
            tau_range=(1e-15, 1e5), allow_out_of_range=True,
        )

    assert all(ax.lines for ax in axes)


def test_plot_fom_defaults_span_table_1(photon_spectrum: NDArray) -> None:
    """The default sweeps are the sampled ranges themselves, so they must pass."""
    fig, axes = plt.subplots(1, 3)

    final_results.plot_FOM(
        fig, list(axes), E_GAP, photon_spectrum, FOM_ARGS["alpha"], FOM_ARGS["tau"],
        FOM_ARGS["sigma"], FOM_ARGS["dos_mass"], FOM_ARGS["dop_density"],
        FOM_ARGS["epsilon"], FOM_ARGS["mu"], TCELL,
    )

    assert all(ax.lines for ax in axes)


def test_plot_fom_panels_are_all_logarithmic(photon_spectrum: NDArray) -> None:
    """Each sweep spans decades, so a linear axis would sample almost none of it."""
    fig, axes = plt.subplots(1, 3)

    final_results.plot_FOM(
        fig, list(axes), E_GAP, photon_spectrum, FOM_ARGS["alpha"], FOM_ARGS["tau"],
        FOM_ARGS["sigma"], FOM_ARGS["dos_mass"], FOM_ARGS["dop_density"],
        FOM_ARGS["epsilon"], FOM_ARGS["mu"], TCELL,
    )

    assert [ax.get_xscale() for ax in axes] == ["log", "log", "log"]


def test_mobility_plot_refuses_an_exponent_outside_table_1(
        photon_spectrum: NDArray
) -> None:
    """The exponents are converted to values before being checked."""
    with pytest.raises(ValueError, match="mu"):
        final_results.mobility_plot(
            E_GAP, photon_spectrum, FOM_ARGS["alpha"], FOM_ARGS["sigma"],
            FOM_ARGS["dos_mass"], FOM_ARGS["epsilon"],
            mob_min=0, mob_max=12, lifetime_min=-9, lifetime_max=-7, step=1,
            Tcell=TCELL,
        )


def test_mobility_plot_defaults_reach_the_documented_bounds(
        photon_spectrum: NDArray
) -> None:
    """The docstring promises 1e-2 to 1e9, so the default grid must reach 1e9."""
    final_results.mobility_plot(
        E_GAP, photon_spectrum, FOM_ARGS["alpha"], FOM_ARGS["sigma"],
        FOM_ARGS["dos_mass"], FOM_ARGS["epsilon"], Tcell=TCELL,
    )

    mobilities = np.asarray(plt.gca().lines[0].get_xdata(), dtype=float)

    assert mobilities[0] == pytest.approx(1e-2)
    assert mobilities[-1] == pytest.approx(1e9)
    assert len(plt.gca().lines) == 19


@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (-2, 9, 1, [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        # 9 is not on a three-wide grid from -2, so the sweep stops below it
        # rather than overshooting to 10 - which would leave both the caller's
        # maximum and the table 1 range behind.
        (-2, 9, 3, [-2, 1, 4, 7]),
        (-15, 3, 2, [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3]),
    ],
)
def test_exponent_grid_includes_the_stop_without_passing_it(
        start: float, stop: float, step: float, expected: list[float]
) -> None:
    """np.arange excluded the stop; pushing it out by a step could overshoot it."""
    grid = final_results._exponent_grid(start, stop, step)

    assert list(grid) == pytest.approx(expected)
    assert grid.max() <= stop
