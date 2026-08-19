from solphin.pv_fom import Final_equation
from solphin.db_fom import max_eff

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings(action="ignore",message="This figure was using a layout engine that is incompatible with subplots_adjust{1}.+")
warnings.filterwarnings(action="ignore",message="invalid value encountered in multiply{1}.+")

# Calculating equation 33 from the FOM paper

def SQ_relative_FOM_PV_efficiency(E_gap, photon_spectrum, alpha, tau, sigma, dos_mass, dop_density, epsilon, mu, Tcell):

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

    SQ_efficiency = SQ / ((1 + fraction) * denom_bracket)

    efficiency = (SQ / 100) * SQ_efficiency

    FOM = ((1 + fraction) * denom_bracket)

    return SQ, SQ_efficiency, efficiency, FOM


def plot_FOM(
        fig:Figure, 
        axes:list[Axes], 
        E_gap,
        photon_spectrum, 
        alpha, 
        tau, 
        sigma, 
        dos_mass, 
        dop_density, 
        epsilon, 
        mu, 
        Tcell,
        dop_range:tuple[float,float], 
        tau_range:tuple[float,float], 
        mu_range:tuple[float,float]):

    #For each of the three quantities, take the other two as fixed and iterate over a sensible range

    #Plot vs dopant density
    densities = np.logspace(np.log10(dop_range[0]), np.log10(dop_range[1]))
    dop_foms = [SQ_relative_FOM_PV_efficiency(E_gap=E_gap, photon_spectrum=photon_spectrum, alpha=alpha, tau=tau, sigma=sigma, dos_mass=dos_mass, dop_density=d, epsilon=epsilon, mu=mu, Tcell=Tcell)[-1] for d in densities]
    axes[0].plot(densities, dop_foms, "-", markersize=6)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Dopant Density (cm$^{-3}$)")
    axes[0].set_ylabel("Figure of Merit")
    axes[0].set_title("Figure of Merit vs Dopant Density \n"+r"($\mu$="+str(mu)+r", $\tau$=" + f"{tau:.2e}" + ")")

    #Plot vs lifetime
    lifetimes = np.linspace(tau_range[0], tau_range[1])
    lifetime_foms = [SQ_relative_FOM_PV_efficiency(E_gap=E_gap, photon_spectrum=photon_spectrum, alpha=alpha, tau=l, sigma=sigma, dos_mass=dos_mass, dop_density=dop_density, epsilon=epsilon, mu=mu, Tcell=Tcell)[-1] for l in lifetimes]
    axes[1].plot(lifetimes, lifetime_foms, "-", markersize=6)
    axes[1].set_xlabel("Carrier Lifetime (s)")
    axes[1].set_title("Figure of Merit vs Carrier Lifetime \n"+r"($\mu$="+str(mu)+r", Density=" + f"{dop_density:.2e}" + ")")

    #Plot vs mobility
    mobilities = np.linspace(mu_range[0], mu_range[1])
    mob_foms = [SQ_relative_FOM_PV_efficiency(E_gap=E_gap, photon_spectrum=photon_spectrum, alpha=alpha, tau=tau, sigma=sigma, dos_mass=dos_mass, dop_density=dop_density, epsilon=epsilon, mu=m, Tcell=Tcell)[-1] for m in mobilities]
    axes[2].plot(mobilities, mob_foms, "-", markersize=6)
    axes[2].set_xlabel("Carrier Mobility (cm$^2$V$^{-1}$s$^{-1}$)")
    axes[2].set_title("Figure of Merit vs Carrier Mobility \n" + "(Density=" + f"{dop_density:.2e}" + r", $\tau$="+ f"{tau:.2e}" + ")")

def _get_step(quantity_range):
    return (quantity_range[1] - quantity_range[0])/100

def plot_final_result_interactive(
        E_gap,
        photon_spectrum, 
        alpha, 
        tau, 
        sigma, 
        dos_mass, 
        dop_density, 
        epsilon, 
        mu, 
        Tcell,
        dop_range:tuple[float,float]=(1e8, 1e24), 
        tau_range:tuple[float,float]=(1e-8, 1e-6), 
        mu_range:tuple[float,float]=(1, 300)):
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
    fig, axes = plt.subplots(1,3, figsize=(12,3), dpi=120, constrained_layout=True)

    fig.canvas.header_visible = False      # hides the figure “header” in JupyterLab
    fig.canvas.footer_visible = False      # hides the footer
    fig.canvas.toolbar_visible = False     # hides the toolbar

    # wrapping that clears axes and redraws combined DB plots

    def plot_combined_wrapper(density, lifetime, mobility):

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

    widget_layout = layout=widgets.Layout(width='800px')
    widget_style = {'description_width': '200px'}

    dopant_slider   = widgets.FloatLogSlider(value=dop_density, min=np.log10(dop_range[0]), max=np.log10(dop_range[1]), step=0.1, description="Dopant Density (cm⁻³)",    layout=widget_layout, style=widget_style)
    lifetime_slider = widgets.FloatLogSlider(value=tau, min=np.log10(tau_range[0]), max=np.log10(tau_range[1]), step=0.01, description="Carrier Lifetime (s)", layout=widget_layout, style=widget_style)
    mobility_slider = widgets.FloatSlider(value=mu, min=mu_range[0], max=mu_range[1], step=_get_step(mu_range), description="Carrier Mobility (cm²V⁻¹s⁻¹)", layout=widget_layout, style=widget_style)

    interact(plot_combined_wrapper, density=dopant_slider, lifetime=lifetime_slider, mobility=mobility_slider)

def _clearlines(n):
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)

def print_final_result_interactive(
        E_gap,
        photon_spectrum, 
        alpha, 
        tau, 
        sigma, 
        dos_mass, 
        dop_density, 
        epsilon, 
        mu, 
        Tcell,
        dop_range:tuple[float,float]=(1e8, 1e24), 
        tau_range:tuple[float,float]=(1e-8, 1e-6), 
        mu_range:tuple[float,float]=(1, 300)):
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

    def print_fom(density, lifetime, mobility):

        _clearlines(5)

        sq, fom_sq, eff, FOM = SQ_relative_FOM_PV_efficiency(E_gap, photon_spectrum, alpha, lifetime, sigma, dos_mass, density, epsilon, mobility, Tcell)
        print("")
        print(f"Photovoltaic Figure of Merit: {FOM:.2f}")
        print(f"Photovoltaic Figure of Merit relative to the SQ limit: {fom_sq:.2f} %")
        print(f"Photovoltaic Figure of Merit total efficiency: {eff:.2f} %")
        print(f"SQ limit: {sq:.2f} %")

    widget_layout = layout=widgets.Layout(width='800px')
    widget_style = {'description_width': '200px'}

    dopant_slider   = widgets.FloatLogSlider(value=dop_density, min=np.log10(dop_range[0]), max=np.log10(dop_range[1]), step=0.1, description="Dopant Density (cm⁻³)",    layout=widget_layout, style=widget_style)
    lifetime_slider = widgets.FloatLogSlider(value=tau, min=np.log10(tau_range[0]), max=np.log10(tau_range[1]), step=0.01, description="Carrier Lifetime (s)", layout=widget_layout, style=widget_style)
    mobility_slider = widgets.FloatSlider(value=mu, min=mu_range[0], max=mu_range[1], step=_get_step(mu_range), description="Carrier Mobility (cm²V⁻¹s⁻¹)", layout=widget_layout, style=widget_style)

    interact(print_fom, density=dopant_slider, lifetime=lifetime_slider, mobility=mobility_slider)
