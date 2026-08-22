import logging
import warnings

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

logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings(action="ignore",
                        message="This figure was using a layout engine that is incompatible with subplots_adjust{1}.+")
warnings.filterwarnings(action="ignore", message="invalid value encountered in multiply{1}.+")


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
    ''' Calculates the final value for the photovoltaic figure of merit from Crovetto 2024 as a percentage of the Shockley Queisser efficiency limit.

    Parameters:
        E_gap(float): Optical Band Gap in eV
        photon_spectrum(numpy.ndarrray): Converted input spectrum from DB_FOM.convert_spectrum y: Number of photons (Np/m2/s/dE) x: Energy (eV)
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        sigma(float): Spectral dispersion of the absorption coefficient spectrum, unitless
        dos_mass(float): Density of States effective mass in m₀
        dop_density(float): Doping density in cm⁻³
        epsilon(float): Static dielectric constant, unitless
        mu(float): Charge carrier mobility in cm²V⁻¹s⁻¹
        Tcell(float): Operating temperature of the cell in K

    Returns:
        efficiency(float): Percentage photovoltaic figure of merit efficiency.
    '''

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
    """
    Plots the photovoltaic figure of merit as a function of key transport parameters.

    This function evaluates and plots the photovoltaic figure of merit while
    independently varying dopant density, carrier lifetime, and carrier
    mobility. For each plot, the other two transport parameters are held fixed
    at their supplied values.

    Parameters:
        fig(Figure): Matplotlib figure containing the axes used for the
            figure-of-merit plots.

        axes(list[Axes]): List of three Matplotlib axes used to plot the figure
            of merit against dopant density, carrier lifetime, and mobility,
            respectively.

        E_gap(float): Electronic band gap used in the photovoltaic efficiency
            calculation.

        photon_spectrum: Incident photon spectrum used in the photovoltaic
            efficiency calculation.

        alpha: Absorption coefficient data used in the photovoltaic efficiency
            calculation.

        tau(float): Carrier lifetime held fixed when varying dopant density and
            mobility.

        sigma: Carrier capture cross section used in the photovoltaic efficiency
            calculation.

        dos_mass(float): Density-of-states effective mass used in the
            photovoltaic efficiency calculation.

        dop_density(float): Dopant density held fixed when varying carrier
            lifetime and mobility.

        epsilon(float): Dielectric constant used in the photovoltaic efficiency
            calculation.

        mu(float): Carrier mobility held fixed when varying dopant density and
            carrier lifetime.

        Tcell(float): Cell temperature used in the photovoltaic efficiency
            calculation.

        dop_range(tuple[float, float]): Minimum and maximum dopant densities
            over which the figure of merit is evaluated.

        tau_range(tuple[float, float]): Minimum and maximum carrier lifetimes
            over which the figure of merit is evaluated.

        mu_range(tuple[float, float]): Minimum and maximum carrier mobilities
            over which the figure of merit is evaluated.

    Returns:
        None
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
    """
    Calculates the step size for a specified quantity range.

    This function divides the interval between the minimum and maximum values
    of the supplied range into 100 equal steps.

    Parameters:
        quantity_range(tuple): Minimum and maximum values defining the range,
            specified as (minimum, maximum).

    Returns:
        float: Step size corresponding to one hundredth of the supplied range.
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
    """
    Creates an interactive Figure of Merit visualization dashboard.

    Parameters:
        E_gap(float): Optical Band Gap in eV
        photon_spectrum(numpy.ndarrray): Converted input spectrum from DB_FOM.convert_spectrum y: Number of photons (Np/m2/s/dE) x: Energy (eV)
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        sigma(float): Spectral dispersion of the absorption coefficient spectrum, unitless
        dos_mass(float): Density of States effective mass in m₀
        dop_density(float): Doping density in cm⁻³
        epsilon(float): Static dielectric constant, unitless
        mu(float): Charge carrier mobility in cm²V⁻¹s⁻¹
        Tcell(float): Operating temperature of the cell in K
        dop_range(tuple[float,float]): Range of dopant densities to explore in the interactive plot (default: (1e8, 1e24))
        tau_range(tuple[float,float]): Range of carrier lifetimes to explore in the interactive plot (default: (1e-8, 1e-6))
        mu_range(tuple[float,float]): Range of carrier mobilities to explore in the interactive plot (default: (1, 300))

    Returns:
        None
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
        """
        Updates the combined figure using the selected transport parameters.

        This function clears the existing figure axes and redraws the
        figure-of-merit plots using the specified doping density, carrier
        lifetime, and mobility together with the fixed material and simulation
        parameters defined in the enclosing scope.

        Parameters:
            density(float): Doping density used for the figure-of-merit
                calculation.

            lifetime(float): Carrier lifetime used for the figure-of-merit
                calculation.

            mobility(float): Carrier mobility used for the figure-of-merit
                calculation.

        Returns:
            None
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
    """
    Clears a specified number of previously printed terminal lines.

    This function uses ANSI escape sequences to move the terminal cursor upward
    and clear each selected line. It is intended for updating or replacing
    previously printed command-line output.

    Parameters:
        n(int): Number of terminal lines to move upward and clear.

    Returns:
        None
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
    """
    Interactively writes figure of merit information.

    Parameters:
        E_gap(float): Optical Band Gap in eV
        photon_spectrum(numpy.ndarrray): Converted input spectrum from DB_FOM.convert_spectrum y: Number of photons (Np/m2/s/dE) x: Energy (eV)
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        sigma(float): Spectral dispersion of the absorption coefficient spectrum, unitless
        dos_mass(float): Density of States effective mass in m₀
        dop_density(float): Doping density in cm⁻³
        epsilon(float): Static dielectric constant, unitless
        mu(float): Charge carrier mobility in cm²V⁻¹s⁻¹
        Tcell(float): Operating temperature of the cell in K
        dop_range(tuple[float,float]): Range of dopant densities to explore in the interactive plot (default: (1e8, 1e24))
        tau_range(tuple[float,float]): Range of carrier lifetimes to explore in the interactive plot (default: (1e-8, 1e-6))
        mu_range(tuple[float,float]): Range of carrier mobilities to explore in the interactive plot (default: (1, 300))

    Returns:
        None
    """

    def print_fom(density: float, lifetime: float, mobility: float) -> None:
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
    """
    Plots the figure-of-merit efficiency against carrier mobility, one line per
    carrier lifetime.

    Mobility and lifetime are swept on logarithmic grids: the *_min and *_max
    arguments are base-10 exponents, not values, so the defaults cover
    1e-2 to 1e9 cm2V-1s-1 and 1e-15 to 1e3 s.

    Parameters:
        E_gap(float): Optical Band Gap in eV.
        photon_spectrum(np.ndarray): Converted photon spectrum from
            db_fom.convert_spectrum; column 0 energy (eV), column 1 photon
            flux (m-2 s-1 eV-1).
        alpha(float): Spectral average of incident light in cm-1.
        sigma(float): Spectral dispersion of the absorption coefficient
            spectrum, unitless.
        dos_mass(float): Density of States effective mass in m0.
        epsilon(float): Static dielectric constant, unitless.
        dop_density(float): Doping density in cm-3. Default is 1e10.
        mob_min(float): Base-10 exponent of the lowest mobility swept.
            Default is -2.
        mob_max(float): Base-10 exponent of the highest mobility swept.
            Default is 9.
        lifetime_min(float): Base-10 exponent of the shortest lifetime swept.
            Default is -15.
        lifetime_max(float): Base-10 exponent of the longest lifetime swept.
            Default is 3.
        step(float): Spacing between successive exponents. Default is 1.
        Tcell(float): Operating temperature of the cell in K. Default is 300.

    Returns:
        None
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
            efficiency = SQ_relative_FOM_PV_efficiency(
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
