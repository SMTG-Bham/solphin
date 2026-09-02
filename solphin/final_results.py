"""Combine the detailed-balance limit and Γₚᵥ into final relative-efficiency results."""

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, widgets
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from numpy.typing import NDArray

from solphin.db_fom import max_eff
from solphin.pv_fom import SAMPLED_RANGES, Final_equation, check_sampled_ranges

# The sweep arguments each vary one property, so their bounds are checked
# against that property's entry in the shared table rather than a second copy of
# the numbers.
_SWEEP_PROPERTIES = {
    "dop_range": "dop_density",
    "tau_range": "tau",
    "mu_range": "mu",
}

# Default sweeps span exactly the sampled range of the property they vary, taken
# from the table itself rather than restated, so a default can never fall
# outside the window the following functions then check it against.
_DEFAULT_DOP_RANGE = SAMPLED_RANGES["dop_density"][:2]
_DEFAULT_TAU_RANGE = SAMPLED_RANGES["tau"][:2]
_DEFAULT_MU_RANGE = SAMPLED_RANGES["mu"][:2]


def _exponent_grid(start: float, stop: float, step: float) -> NDArray:
    """Build an inclusive grid of base-10 exponents from start to stop.

    ``np.arange`` excludes its stop, which left the default sweeps a decade
    short of the bounds they document. Simply pushing the stop out by one step
    overshoots whenever the step does not divide the span, which would take the
    sweep past both the caller's maximum and the range sampled in Crovetto
    2024 table 1. So the endpoint is admitted only when it lands on the grid.

    Parameters
    ----------
    start : float
        Base-10 exponent of the first point.
    stop : float
        Base-10 exponent of the last point, included when the step reaches it
        exactly and otherwise the ceiling the grid stops below.
    step : float
        Spacing between successive exponents.

    Returns
    -------
    numpy.ndarray
        Exponents from ``start``, ascending, none of them above ``stop``.
    """
    exponents = np.arange(start, stop + step / 2, step)

    # A float grid can land a hair above the stop, so compare with a tolerance
    # small against the step rather than exactly.
    return exponents[exponents <= stop + step * 1e-9]


def _check_sweep_range(
        name: str, bounds: tuple[float, float], allow_out_of_range: bool
) -> None:
    """Check the endpoints of a sweep range against Crovetto 2024 table 1.

    A sweep steps one property across a range, so every point it visits has to
    be inside the sampled range for the same reason a single value does. The
    endpoints are checked before any of them is evaluated, so an unusable range
    fails once with the range named rather than once per sample from inside the
    figure of merit.

    Parameters
    ----------
    name : str
        Name of the sweep argument, one of ``"dop_range"``, ``"tau_range"`` or
        ``"mu_range"``.
    bounds : tuple of float
        Minimum and maximum of the sweep.
    allow_out_of_range : bool
        If True, an endpoint outside the table 1 range warns instead of
        raising.

    Raises
    ------
    ValueError
        If either endpoint lies outside the table 1 range of the property the
        sweep varies and ``allow_out_of_range`` is False.

    Warns
    -----
    UserWarning
        The same conditions, when ``allow_out_of_range`` is True.
    """
    prop = _SWEEP_PROPERTIES[name]

    for bound in bounds:
        check_sampled_ranges(allow_out_of_range, **{prop: bound})


# Calculating equation 33 from the FOM paper

def SQ_relative_FOM_PV_efficiency(
        E_gap: float,
        photon_spectrum: NDArray,
        alpha: float,
        tau: float,
        sigma: float,
        dos_mass: float,
        dop_density: float,
        epsilon: float,
        mu: float,
        Tcell: float,
        *,
        allow_out_of_range: bool = False,
) -> tuple[float, float, float]:
    """Calculate the Γₚᵥ efficiency and its value relative to the Shockley-Queisser limit.

    Implements equation 33 of Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``: photon
        energy in eV against photon flux in m⁻² s⁻¹ eV⁻¹.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.
    sigma : float
        Spectral dispersion of the absorption coefficient, dimensionless.
    dos_mass : float
        Density-of-states effective mass in units of m₀.
    dop_density : float
        Doping density in cm⁻³.
    epsilon : float
        Static dielectric constant, dimensionless.
    mu : float
        Charge carrier mobility in cm² V⁻¹ s⁻¹.
    Tcell : float
        Operating temperature of the cell in K.
    allow_out_of_range : bool, optional
        If True, a property outside its Crovetto 2024 table 1 sampled range
        warns instead of raising, and the efficiency is evaluated anyway.
        Default is False. Keyword-only.

    Returns
    -------
    SQ : float
        Shockley-Queisser limit efficiency in %.
    SQ_relative : float
        Figure-of-merit efficiency relative to the SQ limit, in %.
    FOM_efficiency : float
        Figure-of-merit photovoltaic efficiency in %.

    Raises
    ------
    ValueError
        If any property lies outside its Crovetto 2024 table 1 sampled range
        in :data:`~solphin.pv_fom.SAMPLED_RANGES` and ``allow_out_of_range``
        is False.
    """
    PV_FOM = Final_equation(
        E_gap, alpha, tau, sigma, dos_mass, dop_density, epsilon, mu,
        allow_out_of_range=allow_out_of_range,
    )

    k_1 = 3.3e-1
    k_2 = 9.06e-2
    k_3 = 2.48e-3

    FOM_Pv_235 = PV_FOM ** (-0.235)
    FOM_Pv_869 = PV_FOM ** 0.869
    FOM_Pv_362 = PV_FOM ** (-0.362)

    fraction = (k_1 * FOM_Pv_235) / (1 + k_2 * FOM_Pv_869)
    denom_bracket = 1 + (k_3 * FOM_Pv_362)

    SQ_eff = max_eff(E_gap, photon_spectrum, Tcell)
    SQ = SQ_eff * 100

    efficiency = SQ_eff / ((1 + fraction) * denom_bracket)

    FOM_efficiency = efficiency * 100

    SQ_relative = (FOM_efficiency / SQ) * 100

    return SQ, SQ_relative, FOM_efficiency


def plot_FOM(
        fig: Figure,
        axes: list[Axes],
        E_gap: float,
        photon_spectrum: NDArray,
        alpha: float,
        tau: float,
        sigma: float,
        dos_mass: float,
        dop_density: float,
        epsilon: float,
        mu: float,
        Tcell: float,
        dop_range: tuple[float, float] = _DEFAULT_DOP_RANGE,
        tau_range: tuple[float, float] = _DEFAULT_TAU_RANGE,
        mu_range: tuple[float, float] = _DEFAULT_MU_RANGE,
        *,
        allow_out_of_range: bool = False,
) -> None:
    """Plot the figure of merit against doping density, lifetime and mobility.

    Each panel varies one transport parameter while the other two stay fixed
    at their supplied values. All three sweeps are logarithmic, since each
    spans several decades.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure containing the axes for the plots.
    axes : list of Axes
        Three axes for the plots against dopant density, carrier lifetime
        and mobility, respectively.
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Carrier lifetime in s, held fixed when varying dopant density and
        mobility.
    sigma : float
        Spectral dispersion of the absorption coefficient, dimensionless.
    dos_mass : float
        Density-of-states effective mass in units of m₀.
    dop_density : float
        Dopant density in cm⁻³, held fixed when varying carrier lifetime
        and mobility.
    epsilon : float
        Static dielectric constant, dimensionless.
    mu : float
        Carrier mobility in cm² V⁻¹ s⁻¹, held fixed when varying dopant
        density and carrier lifetime.
    Tcell : float
        Operating temperature of the cell in K.
    dop_range : tuple of float, optional
        Minimum and maximum dopant densities evaluated, in cm⁻³. Default is
        the sampled range of Crovetto 2024 table 1.
    tau_range : tuple of float, optional
        Minimum and maximum carrier lifetimes evaluated, in s. Default is the
        sampled range of Crovetto 2024 table 1.
    mu_range : tuple of float, optional
        Minimum and maximum carrier mobilities evaluated, in
        cm² V⁻¹ s⁻¹. Default is the sampled range of Crovetto 2024 table 1.
    allow_out_of_range : bool, optional
        If True, a fixed property or a sweep endpoint outside its Crovetto
        2024 table 1 sampled range warns instead of raising, and the panels
        are drawn anyway. Default is False. Keyword-only.

    Raises
    ------
    ValueError
        If a sweep endpoint or a fixed property lies outside its table 1
        sampled range and ``allow_out_of_range`` is False.
    """
    # Checked before the first point is evaluated, so an unusable range fails
    # once naming the range rather than fifty times from inside the sweep.
    for name, bounds in (
            ("dop_range", dop_range),
            ("tau_range", tau_range),
            ("mu_range", mu_range),
    ):
        _check_sweep_range(name, bounds, allow_out_of_range)

    # For each of the three quantities, take the other two as fixed and iterate over a sensible range

    # Plot vs dopant density
    densities = np.logspace(np.log10(dop_range[0]), np.log10(dop_range[1]))
    dop_foms = [
        SQ_relative_FOM_PV_efficiency(E_gap=E_gap, photon_spectrum=photon_spectrum, alpha=alpha, tau=tau, sigma=sigma,
                                      dos_mass=dos_mass, dop_density=d, epsilon=epsilon, mu=mu, Tcell=Tcell,
                                      allow_out_of_range=allow_out_of_range)[-1] for d
        in densities]
    axes[0].plot(densities, dop_foms, "-", markersize=6)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Doping Density (cm$^{-3}$)")
    axes[0].set_ylabel("Figure of Merit")
    axes[0].set_title("Figure of Merit vs Doping Density \n" + r"($\mu$=" + str(mu) + r", $\tau$=" + f"{tau:.2e}" + ")")

    # Plot vs lifetime
    lifetimes = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]))
    lifetime_foms = [
        SQ_relative_FOM_PV_efficiency(E_gap=E_gap, photon_spectrum=photon_spectrum, alpha=alpha, tau=l, sigma=sigma,
                                      dos_mass=dos_mass, dop_density=dop_density, epsilon=epsilon, mu=mu, Tcell=Tcell,
                                      allow_out_of_range=allow_out_of_range)[
            -1] for l in lifetimes]
    axes[1].plot(lifetimes, lifetime_foms, "-", markersize=6)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Carrier Lifetime (s)")
    axes[1].set_title(
        "Figure of Merit vs Carrier Lifetime \n" + r"($\mu$=" + str(mu) + r", Density=" + f"{dop_density:.2e}" + ")")

    # Plot vs mobility
    mobilities = np.logspace(np.log10(mu_range[0]), np.log10(mu_range[1]))
    mob_foms = [
        SQ_relative_FOM_PV_efficiency(E_gap=E_gap, photon_spectrum=photon_spectrum, alpha=alpha, tau=tau, sigma=sigma,
                                      dos_mass=dos_mass, dop_density=dop_density, epsilon=epsilon, mu=m, Tcell=Tcell,
                                      allow_out_of_range=allow_out_of_range)[
            -1] for m in mobilities]
    axes[2].plot(mobilities, mob_foms, "-", markersize=6)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Carrier Mobility (cm$^2$V$^{-1}$s$^{-1}$)")
    axes[2].set_title(
        "Figure of Merit vs Carrier Mobility \n" + "(Density=" + f"{dop_density:.2e}" + r", $\tau$=" + f"{tau:.2e}" + ")")


def plot_final_result_interactive(
        E_gap: float,
        photon_spectrum: NDArray,
        alpha: float,
        tau: float,
        sigma: float,
        dos_mass: float,
        dop_density: float,
        epsilon: float,
        mu: float,
        Tcell: float,
        dop_range: tuple[float, float] = _DEFAULT_DOP_RANGE,
        tau_range: tuple[float, float] = _DEFAULT_TAU_RANGE,
        mu_range: tuple[float, float] = _DEFAULT_MU_RANGE,
        *,
        allow_out_of_range: bool = False,
) -> None:
    """Create an interactive figure-of-merit dashboard with parameter sliders.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.
    sigma : float
        Spectral dispersion of the absorption coefficient, dimensionless.
    dos_mass : float
        Density-of-states effective mass in units of m₀.
    dop_density : float
        Doping density in cm⁻³.
    epsilon : float
        Static dielectric constant, dimensionless.
    mu : float
        Charge carrier mobility in cm² V⁻¹ s⁻¹.
    Tcell : float
        Operating temperature of the cell in K.
    dop_range : tuple of float, optional
        Dopant density range explored by the slider, in cm⁻³.
        Default is the sampled range of Crovetto 2024 table 1.
    tau_range : tuple of float, optional
        Carrier lifetime range explored by the slider, in s.
        Default is the sampled range of Crovetto 2024 table 1.
    mu_range : tuple of float, optional
        Carrier mobility range explored by the slider, in cm² V⁻¹ s⁻¹.
        Default is the sampled range of Crovetto 2024 table 1.
    allow_out_of_range : bool, optional
        If True, a property or a slider bound outside its Crovetto 2024
        table 1 sampled range warns instead of raising. Default is False.
        Keyword-only.

    Raises
    ------
    ValueError
        If a slider bound or a fixed property lies outside its table 1
        sampled range and ``allow_out_of_range`` is False.
    """
    plt.close("all")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), dpi=120, constrained_layout=True)

    # These three attributes exist only on the ipympl canvas, i.e. under
    # %matplotlib widget. Install it with the "interactive" extra:
    #     pip install "solphin[interactive]"
    fig.canvas.header_visible = False  # type: ignore[attr-defined]  # hides the figure “header” in JupyterLab
    fig.canvas.footer_visible = False  # type: ignore[attr-defined]  # hides the footer
    fig.canvas.toolbar_visible = False  # type: ignore[attr-defined]  # hides the toolbar

    # wrapping that clears axes and redraws combined DB plots

    def plot_combined_wrapper(density: float, lifetime: float, mobility: float) -> None:
        """Clear the axes and redraw the plots for the selected slider values.

        Parameters
        ----------
        density : float
            Doping density in cm⁻³.
        lifetime : float
            Carrier lifetime in s.
        mobility : float
            Carrier mobility in cm² V⁻¹ s⁻¹.
        """
        for ax in fig.axes:
            ax.clear()
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")

        plot_FOM(
            fig=fig,
            axes=axes,
            E_gap=E_gap,
            photon_spectrum=photon_spectrum,
            alpha=alpha,
            tau=lifetime,
            sigma=sigma,
            dos_mass=dos_mass,
            dop_density=density,
            epsilon=epsilon,
            mu=mobility,
            Tcell=Tcell,
            dop_range=dop_range,
            tau_range=tau_range,
            mu_range=mu_range,
            allow_out_of_range=allow_out_of_range,
        )

    widget_layout = layout = widgets.Layout(width='800px')
    widget_style = {'description_width': '200px'}

    dopant_slider = widgets.FloatLogSlider(value=dop_density, min=np.log10(dop_range[0]), max=np.log10(dop_range[1]),
                                           step=0.1, description="Doping Density (cm⁻³)", layout=widget_layout,
                                           style=widget_style)
    lifetime_slider = widgets.FloatLogSlider(value=tau, min=np.log10(tau_range[0]), max=np.log10(tau_range[1]),
                                             step=0.01, description="Carrier Lifetime (s)", layout=widget_layout,
                                             style=widget_style)
    mobility_slider = widgets.FloatLogSlider(value=mu, min=np.log10(mu_range[0]), max=np.log10(mu_range[1]),
                                             step=0.1, description="Carrier Mobility (cm²V⁻¹s⁻¹)",
                                             layout=widget_layout, style=widget_style)

    interact(plot_combined_wrapper, density=dopant_slider, lifetime=lifetime_slider, mobility=mobility_slider)


def _clearlines(n: int) -> None:
    """Clear previously printed terminal lines with ANSI escape sequences.

    Parameters
    ----------
    n : int
        Number of terminal lines to move upward and clear.
    """
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)


def print_final_result_interactive(
        E_gap: float,
        photon_spectrum: NDArray,
        alpha: float,
        tau: float,
        sigma: float,
        dos_mass: float,
        dop_density: float,
        epsilon: float,
        mu: float,
        Tcell: float,
        dop_range: tuple[float, float] = _DEFAULT_DOP_RANGE,
        tau_range: tuple[float, float] = _DEFAULT_TAU_RANGE,
        mu_range: tuple[float, float] = _DEFAULT_MU_RANGE,
        *,
        allow_out_of_range: bool = False,
) -> None:
    """Interactively print the figure-of-merit efficiencies with parameter sliders.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.
    sigma : float
        Spectral dispersion of the absorption coefficient, dimensionless.
    dos_mass : float
        Density-of-states effective mass in units of m₀.
    dop_density : float
        Doping density in cm⁻³.
    epsilon : float
        Static dielectric constant, dimensionless.
    mu : float
        Charge carrier mobility in cm² V⁻¹ s⁻¹.
    Tcell : float
        Operating temperature of the cell in K.
    dop_range : tuple of float, optional
        Dopant density range explored by the slider, in cm⁻³.
        Default is the sampled range of Crovetto 2024 table 1.
    tau_range : tuple of float, optional
        Carrier lifetime range explored by the slider, in s.
        Default is the sampled range of Crovetto 2024 table 1.
    mu_range : tuple of float, optional
        Carrier mobility range explored by the slider, in cm² V⁻¹ s⁻¹.
        Default is the sampled range of Crovetto 2024 table 1.
    allow_out_of_range : bool, optional
        If True, a property or a slider bound outside its Crovetto 2024
        table 1 sampled range warns instead of raising. Default is False.
        Keyword-only.

    Raises
    ------
    ValueError
        If a slider bound or a fixed property lies outside its table 1
        sampled range and ``allow_out_of_range`` is False.
    """

    def print_fom(density: float, lifetime: float, mobility: float) -> None:
        """Print the figure-of-merit efficiencies for the selected slider values."""
        _clearlines(5)

        sq, fom_sq, eff = SQ_relative_FOM_PV_efficiency(E_gap, photon_spectrum, alpha, lifetime, sigma, dos_mass,
                                                        density, epsilon, mobility, Tcell,
                                                        allow_out_of_range=allow_out_of_range)
        print("")
        print(f"Photovoltaic Figure of Merit relative to the SQ limit: {fom_sq:.2f} %")
        print(f"Photovoltaic Figure of Merit total efficiency: {eff:.2f} %")
        print(f"SQ limit: {sq:.2f} %")

    # This wrapper drives the efficiency directly rather than through plot_FOM,
    # so the slider bounds are checked here instead.
    for name, bounds in (
            ("dop_range", dop_range),
            ("tau_range", tau_range),
            ("mu_range", mu_range),
    ):
        _check_sweep_range(name, bounds, allow_out_of_range)

    widget_layout = layout = widgets.Layout(width='800px')
    widget_style = {'description_width': '200px'}

    dopant_slider = widgets.FloatLogSlider(value=dop_density, min=np.log10(dop_range[0]), max=np.log10(dop_range[1]),
                                           step=0.1, description="Doping Density (cm⁻³)", layout=widget_layout,
                                           style=widget_style)
    lifetime_slider = widgets.FloatLogSlider(value=tau, min=np.log10(tau_range[0]), max=np.log10(tau_range[1]),
                                             step=0.01, description="Carrier Lifetime (s)", layout=widget_layout,
                                             style=widget_style)
    mobility_slider = widgets.FloatLogSlider(value=mu, min=np.log10(mu_range[0]), max=np.log10(mu_range[1]),
                                             step=0.1, description="Carrier Mobility (cm²V⁻¹s⁻¹)",
                                             layout=widget_layout, style=widget_style)

    interact(print_fom, density=dopant_slider, lifetime=lifetime_slider, mobility=mobility_slider)


def mobility_plot(
        E_gap: float,
        photon_spectrum: NDArray,
        alpha: float,
        sigma: float,
        dos_mass: float,
        epsilon: float,
        dop_density: float = 1e10,
        mob_min: float = -2,
        mob_max: float = 9,
        lifetime_min: float = -15,
        lifetime_max: float = 3,
        step: float = 1,
        Tcell: float = 300,
        *,
        allow_out_of_range: bool = False,
) -> None:
    """Plot the figure-of-merit efficiency against mobility, one line per lifetime.

    Mobility and lifetime are swept on logarithmic grids: the ``*_min`` and
    ``*_max`` arguments are base-10 exponents, not values, so the defaults
    cover 10⁻² to 10⁹ cm² V⁻¹ s⁻¹ and 10⁻¹⁵ to 10³ s.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    photon_spectrum : numpy.ndarray
        Converted photon flux spectrum from ``convert_spectrum``: photon
        energy in eV against photon flux in m⁻² s⁻¹ eV⁻¹.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    sigma : float
        Spectral dispersion of the absorption coefficient, dimensionless.
    dos_mass : float
        Density-of-states effective mass in units of m₀.
    epsilon : float
        Static dielectric constant, dimensionless.
    dop_density : float, optional
        Doping density in cm⁻³. Default is ``1e10``.
    mob_min : float, optional
        Base-10 exponent of the lowest mobility swept. Default is ``-2``.
    mob_max : float, optional
        Base-10 exponent of the highest mobility swept, included when
        ``step`` reaches it exactly. Default is ``9``.
    lifetime_min : float, optional
        Base-10 exponent of the shortest lifetime swept. Default is ``-15``.
    lifetime_max : float, optional
        Base-10 exponent of the longest lifetime swept, included when ``step``
        reaches it exactly. Default is ``3``.
    step : float, optional
        Spacing between successive exponents. Default is ``1``.
    Tcell : float, optional
        Operating temperature of the cell in K. Default is ``300``.
    allow_out_of_range : bool, optional
        If True, a property or a sweep endpoint outside its Crovetto 2024
        table 1 sampled range warns instead of raising, and the lines are
        drawn anyway. Default is False. Keyword-only.

    Raises
    ------
    ValueError
        If a sweep endpoint or a fixed property lies outside its table 1
        sampled range and ``allow_out_of_range`` is False.
    """
    # The sweeps are given as exponents, so they are converted before being
    # checked against the table, which holds values.
    _check_sweep_range("mu_range", (10.0 ** mob_min, 10.0 ** mob_max), allow_out_of_range)
    _check_sweep_range(
        "tau_range", (10.0 ** lifetime_min, 10.0 ** lifetime_max), allow_out_of_range
    )

    # Exponents
    mobility_exp = _exponent_grid(mob_min, mob_max, step)
    lifetime_exp = _exponent_grid(lifetime_min, lifetime_max, step)

    # Actual values: 1e-2, 1e-1, 1e0, ...
    mobility_values = 10.0 ** mobility_exp
    lifetime_values = 10.0 ** lifetime_exp

    # Larger figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colour represents actual lifetime
    cmap = plt.colormaps["viridis"]
    norm = LogNorm(
        vmin=lifetime_values.min(),
        vmax=lifetime_values.max()
    )

    # One line per lifetime
    for tau in lifetime_values:

        efficiency_list = []

        for mu in mobility_values:
            _, _, efficiency = SQ_relative_FOM_PV_efficiency(
                E_gap,
                photon_spectrum,
                alpha,
                tau=tau,
                sigma=sigma,
                dos_mass=dos_mass,
                dop_density=dop_density,
                epsilon=epsilon,
                mu=mu,
                Tcell=Tcell,
                allow_out_of_range=allow_out_of_range,
            )

            efficiency_list.append(efficiency)

        ax.plot(
            mobility_values,
            efficiency_list,
            color=cmap(norm(tau)),
            linewidth=2
        )

    # Logarithmic x-axis
    ax.set_xscale("log")

    # Colour bar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Lifetime")

    # Labels
    ax.set_xlabel("Mobility")
    ax.set_ylabel("PV efficiency")

    # Larger text
    ax.tick_params(axis="both", labelsize=12)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    plt.show()
