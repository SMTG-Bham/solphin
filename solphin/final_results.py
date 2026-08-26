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
from solphin.pv_fom import Final_equation


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

    Returns
    -------
    SQ : float
        Shockley-Queisser limit efficiency in %.
    SQ_relative : float
        Figure-of-merit efficiency relative to the SQ limit, in %.
    FOM_efficiency : float
        Figure-of-merit photovoltaic efficiency in %.
    """
    PV_FOM = Final_equation(E_gap, alpha, tau, sigma, dos_mass, dop_density, epsilon, mu)

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
        dop_range: tuple[float, float],
        tau_range: tuple[float, float],
        mu_range: tuple[float, float],
) -> None:
    """Plot the figure of merit against doping density, lifetime and mobility.

    Each panel varies one transport parameter while the other two stay fixed
    at their supplied values.

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
    dop_range : tuple of float
        Minimum and maximum dopant densities evaluated.
    tau_range : tuple of float
        Minimum and maximum carrier lifetimes evaluated.
    mu_range : tuple of float
        Minimum and maximum carrier mobilities evaluated.
    """
    # For each of the three quantities, take the other two as fixed and iterate over a sensible range

    # Plot vs dopant density
    densities = np.logspace(np.log10(dop_range[0]), np.log10(dop_range[1]))
    dop_foms = [
        SQ_relative_FOM_PV_efficiency(E_gap=E_gap, photon_spectrum=photon_spectrum, alpha=alpha, tau=tau, sigma=sigma,
                                      dos_mass=dos_mass, dop_density=d, epsilon=epsilon, mu=mu, Tcell=Tcell)[-1] for d
        in densities]
    axes[0].plot(densities, dop_foms, "-", markersize=6)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Doping Density (cm$^{-3}$)")
    axes[0].set_ylabel("Figure of Merit")
    axes[0].set_title("Figure of Merit vs Doping Density \n" + r"($\mu$=" + str(mu) + r", $\tau$=" + f"{tau:.2e}" + ")")

    # Plot vs lifetime
    lifetimes = np.linspace(tau_range[0], tau_range[1])
    lifetime_foms = [
        SQ_relative_FOM_PV_efficiency(E_gap=E_gap, photon_spectrum=photon_spectrum, alpha=alpha, tau=l, sigma=sigma,
                                      dos_mass=dos_mass, dop_density=dop_density, epsilon=epsilon, mu=mu, Tcell=Tcell)[
            -1] for l in lifetimes]
    axes[1].plot(lifetimes, lifetime_foms, "-", markersize=6)
    axes[1].set_xlabel("Carrier Lifetime (s)")
    axes[1].set_title(
        "Figure of Merit vs Carrier Lifetime \n" + r"($\mu$=" + str(mu) + r", Density=" + f"{dop_density:.2e}" + ")")

    # Plot vs mobility
    mobilities = np.linspace(mu_range[0], mu_range[1])
    mob_foms = [
        SQ_relative_FOM_PV_efficiency(E_gap=E_gap, photon_spectrum=photon_spectrum, alpha=alpha, tau=tau, sigma=sigma,
                                      dos_mass=dos_mass, dop_density=dop_density, epsilon=epsilon, mu=m, Tcell=Tcell)[
            -1] for m in mobilities]
    axes[2].plot(mobilities, mob_foms, "-", markersize=6)
    axes[2].set_xlabel("Carrier Mobility (cm$^2$V$^{-1}$s$^{-1}$)")
    axes[2].set_title(
        "Figure of Merit vs Carrier Mobility \n" + "(Density=" + f"{dop_density:.2e}" + r", $\tau$=" + f"{tau:.2e}" + ")")


def _get_step(quantity_range: tuple[float, float]) -> float:
    """Calculate the slider step size for a quantity range.

    Parameters
    ----------
    quantity_range : tuple of float
        Minimum and maximum values defining the range.

    Returns
    -------
    float
        Step size, one hundredth of the supplied range.
    """
    return (quantity_range[1] - quantity_range[0]) / 100


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
        dop_range: tuple[float, float] = (1e10, 1e18),
        tau_range: tuple[float, float] = (1e-15, 1e3),
        mu_range: tuple[float, float] = (1e-2, 1e9),
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
        Dopant density range explored by the slider.
        Default is ``(1e10, 1e18)``.
    tau_range : tuple of float, optional
        Carrier lifetime range explored by the slider.
        Default is ``(1e-15, 1e3)``.
    mu_range : tuple of float, optional
        Carrier mobility range explored by the slider.
        Default is ``(1e-2, 1e9)``.
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
        )

    widget_layout = layout = widgets.Layout(width='800px')
    widget_style = {'description_width': '200px'}

    dopant_slider = widgets.FloatLogSlider(value=dop_density, min=np.log10(dop_range[0]), max=np.log10(dop_range[1]),
                                           step=0.1, description="Doping Density (cm⁻³)", layout=widget_layout,
                                           style=widget_style)
    lifetime_slider = widgets.FloatLogSlider(value=tau, min=np.log10(tau_range[0]), max=np.log10(tau_range[1]),
                                             step=0.01, description="Carrier Lifetime (s)", layout=widget_layout,
                                             style=widget_style)
    mobility_slider = widgets.FloatSlider(value=mu, min=mu_range[0], max=mu_range[1], step=_get_step(mu_range),
                                          description="Carrier Mobility (cm²V⁻¹s⁻¹)", layout=widget_layout,
                                          style=widget_style)

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
        dop_range: tuple[float, float] = (1e10, 1e18),
        tau_range: tuple[float, float] = (1e-15, 1e3),
        mu_range: tuple[float, float] = (1e-2, 1e9),
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
        Dopant density range explored by the slider.
        Default is ``(1e10, 1e18)``.
    tau_range : tuple of float, optional
        Carrier lifetime range explored by the slider.
        Default is ``(1e-15, 1e3)``.
    mu_range : tuple of float, optional
        Carrier mobility range explored by the slider.
        Default is ``(1e-2, 1e9)``.
    """

    def print_fom(density: float, lifetime: float, mobility: float) -> None:
        """Print the figure-of-merit efficiencies for the selected slider values."""
        _clearlines(5)

        sq, fom_sq, eff = SQ_relative_FOM_PV_efficiency(E_gap, photon_spectrum, alpha, lifetime, sigma, dos_mass,
                                                        density, epsilon, mobility, Tcell)
        print("")
        print(f"Photovoltaic Figure of Merit relative to the SQ limit: {fom_sq:.2f} %")
        print(f"Photovoltaic Figure of Merit total efficiency: {eff:.2f} %")
        print(f"SQ limit: {sq:.2f} %")

    widget_layout = layout = widgets.Layout(width='800px')
    widget_style = {'description_width': '200px'}

    dopant_slider = widgets.FloatLogSlider(value=dop_density, min=np.log10(dop_range[0]), max=np.log10(dop_range[1]),
                                           step=0.1, description="Doping Density (cm⁻³)", layout=widget_layout,
                                           style=widget_style)
    lifetime_slider = widgets.FloatLogSlider(value=tau, min=np.log10(tau_range[0]), max=np.log10(tau_range[1]),
                                             step=0.01, description="Carrier Lifetime (s)", layout=widget_layout,
                                             style=widget_style)
    mobility_slider = widgets.FloatSlider(value=mu, min=mu_range[0], max=mu_range[1], step=_get_step(mu_range),
                                          description="Carrier Mobility (cm²V⁻¹s⁻¹)", layout=widget_layout,
                                          style=widget_style)

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
        Base-10 exponent of the highest mobility swept. Default is ``9``.
    lifetime_min : float, optional
        Base-10 exponent of the shortest lifetime swept. Default is ``-15``.
    lifetime_max : float, optional
        Base-10 exponent of the longest lifetime swept. Default is ``3``.
    step : float, optional
        Spacing between successive exponents. Default is ``1``.
    Tcell : float, optional
        Operating temperature of the cell in K. Default is ``300``.
    """
    # Exponents
    mobility_exp = np.arange(mob_min, mob_max, step)
    lifetime_exp = np.arange(lifetime_min, lifetime_max, step)

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
                Tcell=Tcell
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
