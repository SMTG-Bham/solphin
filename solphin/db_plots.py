"""Plots of detailed-balance quantities: photon flux, J-V curves and efficiency limits."""

from typing import Any, overload

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, widgets
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

import solphin.db_fom as db_fom

# Number of band-gap samples the sweep plots draw by default. The curves are
# smooth, so a few hundred points resolve them; the previous implementations
# reused the spectrum's own energy grid, which after padding put ~74% of its
# samples below 1 eV at sub-meV spacing - thousands of evaluations at a
# resolution no reader of the figure benefits from.
DEFAULT_GAP_POINTS = 300


def _gap_axis(
        spectrum: NDArray,
        Emin: float | None,
        Emax: float | None,
        n_points: int,
) -> NDArray:
    """Build the band-gap axis for a sweep plot.

    Decoupling this axis from the spectrum's own energy grid is what keeps the
    cost of these plots independent of how finely the illuminant happens to be
    tabulated.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Converted photon flux spectrum from ``db_fom.convert_spectrum``.
    Emin : float or None
        Lowest band gap to plot, in eV. None uses the spectrum's lowest
        photon energy.
    Emax : float or None
        Highest band gap to plot, in eV. None uses the third-highest photon
        energy in the spectrum: at the top two samples the above-gap flux has
        already fallen to zero while the recombination term has underflowed,
        so the ratio there is not representable. The previous implementations
        skipped the same two points.
    n_points : int
        Number of samples across the range.

    Returns
    -------
    numpy.ndarray
        Ascending band-gap values in eV.
    """
    energies = np.sort(np.asarray(spectrum[:, 0], dtype=float))

    low = energies[0] if Emin is None else Emin
    high = energies[-3] if Emax is None else Emax

    return np.linspace(low, high, n_points)



@overload
def photons_above_bandgap_plot(
        spectrum: NDArray,
        Egap: float,
        ax: Axes,
        Emin: float | None = None,
        Emax: float | None = None,
        n_points: int = ...,
) -> Axes: ...


@overload
def photons_above_bandgap_plot(
        spectrum: NDArray,
        Egap: float,
        ax: None = None,
        Emin: float | None = None,
        Emax: float | None = None,
        n_points: int = ...,
) -> None: ...


def photons_above_bandgap_plot(
        spectrum: NDArray,
        Egap: float,
        ax: Axes | None = None,
        Emin: float | None = None,
        Emax: float | None = None,
        n_points: int = DEFAULT_GAP_POINTS,
) -> Axes | None:
    """Plot the number of photons above a bandgap as a function of bandgap energy.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Converted photon flux spectrum from ``db_fom.convert_spectrum``;
        column 0 is photon energy in eV, column 1 photon flux.
    Egap : float
        Bandgap energy in eV at which the above-bandgap photon flux is
        highlighted.
    ax : Axes or None, optional
        Axis to draw the plot on. Default is None, which creates and shows
        a standalone figure.
    Emin : float or None, optional
        Lowest band gap to plot, in eV. Default is None; see
        :func:`_gap_axis`.
    Emax : float or None, optional
        Highest band gap to plot, in eV. Default is None; see
        :func:`_gap_axis`.
    n_points : int, optional
        Number of band-gap samples. Default is ``DEFAULT_GAP_POINTS``.

    Returns
    -------
    Axes or None
        The axis passed in, or None when the function created and showed
        its own figure.
    """
    gaps = _gap_axis(spectrum, Emin, Emax, n_points)
    counts = np.array([db_fom._photons_above_bandgap(gap, spectrum) for gap in gaps])

    canvas: Any = ax if ax else plt

    canvas.plot(gaps, counts, color='#231123')

    p_above_1_1 = db_fom._photons_above_bandgap(Egap, spectrum)
    canvas.plot([Egap], [p_above_1_1], 'ro', markersize=5)  # , color='#FF6666')
    canvas.text(Egap + 0.1, p_above_1_1, '{:.4} eV, {:.4}'.format(Egap, p_above_1_1))

    if ax:
        canvas.set_xlabel('$E_{gap}$ (eV)')
        canvas.set_ylabel('# Photons $m^{-2}s^{-1}$')
        canvas.set_title('# above-bandgap \nphotons vs bandgap')
        return ax

    else:
        canvas.xlabel('$E_{gap}$ (eV)')
        canvas.ylabel('# Photons $m^{-2}s^{-1}$')
        canvas.title('Number of above-bandgap \nphotons as a function of bandgap')

        fig = plt.gcf()
        fig.set_size_inches(3, 3)
        fig.set_dpi(150)
        plt.show()

        return None


def iv_curve_plot(
        spectrum: NDArray, Egap: float, Tcell: float, power: bool = False
) -> None:
    """Plot the ideal current-voltage or power-voltage curve for a material.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Converted photon flux spectrum from ``db_fom.convert_spectrum``.
    Egap : float
        Bandgap energy of the material in eV.
    Tcell : float
        Cell temperature in K.
    power : bool, optional
        Plot power against voltage instead of current density against
        voltage. Default is ``False``.
    """
    v_open = db_fom.voc(Egap, spectrum, Tcell)
    v = np.linspace(0, v_open)
    if power:
        p = v * db_fom.current_density(Egap, spectrum, v, Tcell)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Power generated ($W$)')
        plt.title('Power Curve')
        plt.plot(v, p, color='#231123')
    else:
        i = db_fom.current_density(Egap, spectrum, v, Tcell)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current density $J$ ($Am^{-2}$)')
        plt.title('IV Curve')
        plt.plot(v, i, color='#231123')


def iv_pv_curve_plot(
        spectrum: NDArray,
        Egap: float,
        Tcell: float,
        power: bool = False,
        ax1: Axes | None = None,
        ax2: Axes | None = None,
) -> None:
    """Plot the ideal current-voltage and power-voltage curves for a material.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Converted photon flux spectrum from ``db_fom.convert_spectrum``.
    Egap : float
        Bandgap energy of the material in eV.
    Tcell : float
        Cell temperature in K.
    power : bool, optional
        Unused flag, kept for compatibility; power is always plotted.
        Default is ``False``.
    ax1 : Axes or None, optional
        Primary axis for the current density. Default is None, which
        creates a new dual-axis figure.
    ax2 : Axes or None, optional
        Secondary axis for the power curve. Default is None, which twins the
        y-axis of ``ax1``.
    """
    v_open = db_fom.voc(Egap, spectrum, Tcell)
    v = np.linspace(0, v_open)

    if ax1 is None:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

    elif ax2 is None:
        # A caller can hand over the primary axis on its own; the power curve
        # still needs a second y-axis to draw on.
        ax2 = ax1.twinx()

    p = v * db_fom.current_density(Egap, spectrum, v, Tcell)
    i = db_fom.current_density(Egap, spectrum, v, Tcell)

    ax1.plot(v, i, color='#231123')
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current density $J$ ($Am^{-2}$)')
    ax1.legend(['Current'], loc=2)

    ax2.plot(v, p, color='#FF6666')
    ax2.set_ylabel('Power generated ($W$)')
    ax2.legend(['Power'], loc=3)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    if ax1:
        ax1.set_title("IV & PV Plot")

    else:
        fig = plt.gcf()
        fig.set_size_inches(3, 3)
        fig.set_dpi(150)


def sq_limit_plot(
        spectrum: NDArray,
        Egap: float,
        Tcell: float,
        ax: Axes | None = None,
        Emin: float | None = None,
        Emax: float | None = None,
        n_points: int = DEFAULT_GAP_POINTS,
) -> None:
    """Plot the Shockley-Queisser efficiency limit against bandgap energy.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Converted photon flux spectrum from ``db_fom.convert_spectrum``.
    Egap : float
        Bandgap energy in eV at which the SQ efficiency is highlighted.
    Tcell : float
        Cell temperature in K.
    ax : Axes or None, optional
        Axis to draw the plot on. Default is None, which creates a new
        figure.
    Emin : float or None, optional
        Lowest band gap to plot, in eV. Default is None; see
        :func:`_gap_axis`.
    Emax : float or None, optional
        Highest band gap to plot, in eV. Default is None; see
        :func:`_gap_axis`.
    n_points : int, optional
        Number of band-gap samples. Default is ``DEFAULT_GAP_POINTS``.
    """
    # Plot the famous SQ limit. The incident power does not depend on the band
    # gap, so it is computed once here rather than inside every max_eff call.
    gaps = _gap_axis(spectrum, Emin, Emax, n_points)
    power_in = db_fom.incident_power(spectrum)
    efficiencies = np.array(
        [db_fom.max_eff(gap, spectrum, Tcell, power_in=power_in) for gap in gaps]
    ) * 100

    canvas: Any = ax if ax else plt

    canvas.plot(gaps, efficiencies)
    e_gap = Egap
    p_above_1_1 = db_fom.max_eff(e_gap, spectrum, Tcell, power_in=power_in)
    percentage_sq = p_above_1_1 * 100

    canvas.plot([e_gap], [percentage_sq], 'ro', markersize=5)
    canvas.text(e_gap + 0.1, percentage_sq - 0.5, '{:.2} eV, {:.2f}%'.format(e_gap, percentage_sq))

    if ax:
        ax.set_xlabel('$E_{gap}$ (eV)')
        ax.set_ylabel('Max efficiency (%)')
        ax.set_title('SQ Limit')
        return

    else:
        plt.xlabel('$E_{gap}$ (eV)')
        plt.ylabel('Max efficiency (%)')
        plt.title('SQ Limit')

        fig = plt.gcf()
        fig.set_size_inches(3, 3)
        fig.set_dpi(150)


def plot_db_combined(
        spectrum: NDArray,
        Egap: float,
        Tcell: float,
        spectrum_type: str,
        target_lux: float | None = None,
        fig: Figure | None = None,
        axes: NDArray | None = None,
        ax2: Axes | None = None,
) -> None:
    """Generate a combined detailed-balance analysis figure.

    Three panels: photons above the bandgap, ideal IV and power curves, and
    the Shockley-Queisser efficiency limit.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Converted photon flux spectrum from ``db_fom.convert_spectrum``.
    Egap : float
        Bandgap energy in eV.
    Tcell : float
        Cell temperature in K.
    spectrum_type : str
        Label for the spectrum used, e.g. ``"AM1.5"``.
    target_lux : float or None, optional
        Illuminance the spectrum was normalised to, in lux. Added to the
        figure title for indoor spectra, whose efficiencies are meaningless
        without it. Default is None, which omits it.
    fig : Figure or None, optional
        Existing figure to draw into. Default is None, which creates a new
        figure.
    axes : numpy.ndarray of Axes or None, optional
        Array of axes for the subplots. Default is None, which creates a
        new 1x3 subplot grid.
    ax2 : Axes or None, optional
        Secondary y-axis for the IV-plot power curve. Default is None.

    Raises
    ------
    ValueError
        If ``fig`` is given without ``axes``.
    """
    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3), dpi=120)
        ax2 = axes[1].twinx()

    elif axes is None:
        raise ValueError(
            "axes is required alongside fig: the three panels are drawn onto "
            "axes[0], axes[1] and axes[2], and a bare figure carries none."
        )

    photons_above_bandgap_plot(spectrum, Egap, ax=axes[0])
    iv_pv_curve_plot(spectrum, Egap, Tcell, ax1=axes[1], ax2=ax2)
    sq_limit_plot(spectrum, Egap, Tcell, ax=axes[2])

    fig.subplots_adjust(wspace=0.5)
    # AM1.5 is an absolute standard, so its light level needs no stating; an
    # indoor efficiency is only interpretable alongside its illuminance.
    illuminance = ""
    if target_lux is not None and spectrum_type != "AM1.5":
        illuminance = f" at {target_lux:.0f} lx"
    fig.suptitle(
        "$E_{gap}$" + f"= {Egap:.2f} eV, T = {Tcell} K, {spectrum_type} Spectrum{illuminance}",
        y=1.1,
    )


def plot_db_combined_interactive(
        Tmin: float = 1,
        Tmax: float = 1000,
        Emin: float = 0.1,
        Emax: float = 3.1,
        target_lux: float = db_fom.DEFAULT_TARGET_LUX,
) -> None:
    """Create an interactive detailed-balance dashboard with Jupyter widgets.

    Sliders explore the bandgap, operating temperature, illumination spectrum
    and illuminance, dynamically updating the combined detailed-balance plots.

    Parameters
    ----------
    Tmin : float, optional
        Minimum operating temperature in K. Default is ``1``.
    Tmax : float, optional
        Maximum operating temperature in K. Default is ``1000``.
    Emin : float, optional
        Minimum bandgap energy in eV. Default is ``0.1``.
    Emax : float, optional
        Maximum bandgap energy in eV. Default is ``3.1``.
    target_lux : float, optional
        Starting value for the illuminance slider, in lux. The slider has no
        effect while ``"AM1.5"`` is selected, that spectrum being an absolute
        standard. Default is ``db_fom.DEFAULT_TARGET_LUX``, 1000 lux.
    """
    plt.close("all")
    # No constrained_layout here: plot_db_combined finishes with
    # fig.subplots_adjust(wspace=0.5), which a layout engine would override and
    # warn about, so the static and interactive figures are built the same way.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), dpi=120)
    ax2 = axes[1].twinx()

    # These three attributes exist only on the ipympl canvas, i.e. under
    # %matplotlib widget. Install it with the "interactive" extra:
    #     pip install "solphin[interactive]"
    # hides the figure “header” in JupyterLab
    fig.canvas.header_visible = False  # type: ignore[attr-defined]
    fig.canvas.footer_visible = False  # type: ignore[attr-defined]  # hides the footer
    fig.canvas.toolbar_visible = False  # type: ignore[attr-defined]  # hides the toolbar

    # wrapping that clears axes and redraws combined DB plots

    def plot_combined_wrapper(
            Egap: float, Tcell: float, spectrum_type: str, illuminance: float
    ) -> None:
        """Clear the axes and redraw the combined plots for the selected values."""
        for ax in fig.axes:
            ax.clear()
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")

        spectrum = db_fom.load_spectrum(spectrum_type=spectrum_type, target_lux=illuminance)
        photon_spectrum = db_fom.convert_spectrum(spectrum)

        plot_db_combined(photon_spectrum, Egap, Tcell, spectrum_type, target_lux=illuminance,
                         fig=fig, axes=axes, ax2=ax2)

    # Derived from the loader rather than restated, so a spectrum added to
    # db_fom.SPECTRUM_FILES appears in the dropdown automatically.
    spectra = list(db_fom.SPECTRUM_FILES)
    widget_layout = layout = widgets.Layout(width='800px')
    widget_style = {'description_width': '200px'}

    Tmid = Tmin + ((Tmax - Tmin) / 2)
    Tstart = 300 if Tmin <= 300 <= Tmax else Tmid

    Estart = Emin + ((Emax - Emin) / 2)

    temp_slider = widgets.IntSlider(value=Tstart, min=Tmin, max=Tmax, step=1, description="Operating Temperature (K)",
                                    layout=widget_layout, style=widget_style)
    gap_slider = widgets.FloatSlider(value=Estart, min=Emin, max=Emax, step=0.05, description="Band Gap (eV)",
                                     layout=widget_layout, style=widget_style)
    spectrum_drop = widgets.Dropdown(options=spectra, value=spectra[0], description="Optical Spectrum",
                                     style=widget_style)
    # Log scale: the useful indoor range spans three decades, from a dim
    # corridor to direct task lighting.
    lux_slider = widgets.FloatLogSlider(value=target_lux, base=10, min=1, max=5, step=0.05,
                                        description="Illuminance (lx)", readout_format=".0f",
                                        layout=widget_layout, style=widget_style)

    interact(plot_combined_wrapper, Egap=gap_slider, Tcell=temp_slider, spectrum_type=spectrum_drop,
             illuminance=lux_slider)
