import solphin.db_fom as db_fom
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings(action="ignore",message="This figure was using a layout engine that is incompatible with subplots_adjust{1}.+")
warnings.filterwarnings(action="ignore",message="invalid value encountered in multiply{1}.+")


def photons_above_bandgap_plot(spectrum, Egap, ax=None):
    """
    Plots the number of photons above a given bandgap as a function of bandgap energy.

    This function evaluates and visualizes the integrated photon flux above a bandgap
    threshold across a spectral dataset. It can either plot on a provided Matplotlib
    axis or create a standalone figure.

    Parameters:
        spectrum(np.array): 2D array where:
            - column 0 is photon energy or wavelength value (as used by db_fom)
            - column 1 is spectral intensity or irradiance data
        Egap(float): bandgap energy in eV at which to evaluate and highlight
            the photon flux above the bandgap.
        ax(matplotlib.axes.Axes or None): optional Matplotlib axis object.
            If provided, the plot is drawn on this axis; otherwise a new figure
            is created.

    Returns:
        matplotlib.axes.Axes or None: returns the axis object if provided,
            otherwise displays the plot and returns None.
    """

    a = np.copy(spectrum)
    for row in a:
        # print row
        row[1] = db_fom._photons_above_bandgap(row[0], spectrum)

    canvas = ax if ax else plt
    
    canvas.plot(a[:, 0], a[:, 1], color='#231123')

    p_above_1_1 = db_fom._photons_above_bandgap(Egap, spectrum)
    canvas.plot([Egap], [p_above_1_1], 'ro')#, color='#FF6666')
    canvas.text(Egap+0.05, p_above_1_1, '{:.4}eV, {:.4}'.format(Egap, p_above_1_1))

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

def iv_curve_plot(spectrum, Egap, Tcell, power=False):
    """
    Plots the ideal current-voltage (IV) curve or power-voltage curve for a photovoltaic material.

    This function computes the voltage-dependent current density using a detailed-balance
    model and optionally converts it into output power. It visualizes either the IV curve
    or the power curve depending on the input flag.

    Parameters:
        spectrum(np.array): spectral dataset used for detailed-balance calculations.
            Format is expected to be compatible with db_fom functions.
        Egap(float): bandgap energy in eV of the material.
        Tcell(float): cell temperature in Kelvin.
        power(bool): if True, plots power vs voltage curve;
            if False, plots current density vs voltage curve.

    Returns:
        None
    """
    v_open = db_fom.voc(Egap, spectrum, Tcell)
    v = np.linspace(0, v_open)
    if power:
        p =  v * db_fom.current_density(Egap, spectrum, v, Tcell)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Power generated ($W$)')
        plt.title('Power Curve')
        plt.plot(v, p, color='#231123')
    else:
        i =  db_fom.current_density(Egap, spectrum, v, Tcell)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current density $J$ ($Am^{-2}$)')
        plt.title('IV Curve')
        plt.plot(v, i, color='#231123')

def iv_pv_curve_plot(spectrum, Egap, Tcell, power=False, ax1=None, ax2=None):
    """
    Plots the ideal current-voltage (IV) curve and power-voltage curve for a photovoltaic material.

    This function computes the voltage-dependent current density and power output using a
    detailed-balance photovoltaic model. It supports plotting either on existing Matplotlib
    axes or creating a new dual-axis figure.

    Parameters:
        spectrum(np.array): spectral dataset used for detailed-balance calculations.
            Format must be compatible with db_fom functions.
        Egap(float): bandgap energy in eV of the material.
        Tcell(float): cell temperature in Kelvin.
        power(bool): unused flag (kept for compatibility; power is always plotted).
        ax1(matplotlib.axes.Axes or None): primary axis for current density.
            If None, a new figure is created.
        ax2(matplotlib.axes.Axes or None): secondary axis for power curve.
            Used only when ax1 is provided.

    Returns:
        None
    """
    v_open = db_fom.voc(Egap, spectrum, Tcell)
    v = np.linspace(0, v_open)

    if ax1:
        ax1 = ax1
        ax2 = ax2
    else:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
    
    p =  v * db_fom.current_density(Egap, spectrum, v, Tcell)
    i =  db_fom.current_density(Egap, spectrum, v, Tcell)
    
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

def sq_limit_plot(spectrum, Egap, Tcell, ax=None):
    """
    Plots the Shockley–Queisser (SQ) efficiency limit as a function of bandgap energy.

    This function computes and visualizes the theoretical maximum photovoltaic efficiency
    for a given spectrum using a detailed-balance model. It highlights the efficiency at a
    specified bandgap energy and optionally plots on a provided Matplotlib axis.

    Parameters:
        spectrum(np.array): spectral dataset used for detailed-balance calculations.
            Format must be compatible with db_fom functions.
        Egap(float): bandgap energy in eV at which the SQ efficiency is highlighted.
        Tcell(float): cell temperature in Kelvin.
        ax(matplotlib.axes.Axes or None): optional Matplotlib axis object.
            If provided, the plot is drawn on this axis; otherwise a new figure
            is created.

    Returns:
        None or matplotlib.axes.Axes:
            Returns the axis object if provided; otherwise displays the plot.
    """
    # Plot the famous SQ limit
    a = np.copy(spectrum)

    # Not for whole array hack to remove divide by 0 errors
    for row in a[2:]:
        # print row
        row[1] = db_fom.max_eff(row[0], spectrum, Tcell)

    canvas = ax if ax else plt

    # Not plotting whole array becase some bad values happen
    canvas.plot(a[2:, 0], a[2:, 1])
    e_gap = Egap
    p_above_1_1 = db_fom.max_eff(e_gap, spectrum, Tcell)
    
    canvas.plot([e_gap], [p_above_1_1], 'ro')
    canvas.text(e_gap+0.05, p_above_1_1, '{:.4}eV, {:.4}'.format(e_gap, p_above_1_1))

    if ax:
        ax.set_xlabel('$E_{gap}$ (eV)')
        ax.set_ylabel('Max efficiency')
        ax.set_title('SQ Limit')
        return
    
    else:
        plt.xlabel('$E_{gap}$ (eV)')
        plt.ylabel('Max efficiency')
        plt.title('SQ Limit')

        fig = plt.gcf()
        fig.set_size_inches(3, 3)
        fig.set_dpi(150)


def plot_db_combined(spectrum, Egap, Tcell, spectrum_type, fig=None, axes=None, ax2=None):
    """
    Generates a combined detailed-balance photovoltaic analysis figure.

    This function creates a multi-panel plot summarizing key photovoltaic performance
    metrics for a given spectrum and material bandgap. It includes:
    (1) photons above bandgap,
    (2) ideal IV and power curves,
    (3) Shockley–Queisser efficiency limit.

    Parameters:
        spectrum(np.array): spectral dataset used for detailed-balance calculations.
        Egap(float): bandgap energy in eV.
        Tcell(float): cell temperature in Kelvin.
        spectrum_type(string): label for the spectrum used (e.g., AM1.5).
        fig(matplotlib.figure.Figure or None): optional existing figure object.
            If None, a new figure is created.
        axes(matplotlib.axes.Axes or None): optional array of axes for subplots.
            If None, a new 1x3 subplot grid is created.
        ax2(matplotlib.axes.Axes or None): optional secondary y-axis for IV plot
            power curve.

    Returns:
        None
    """

    if not fig:
        fig, axes = plt.subplots(1,3, figsize=(14,3), dpi=120)
        ax2 = axes[1].twinx()

    photons_above_bandgap_plot(spectrum, Egap, ax=axes[0])
    iv_pv_curve_plot(spectrum, Egap, Tcell, ax1=axes[1], ax2=ax2)
    sq_limit_plot(spectrum, Egap, Tcell, ax=axes[2])

    fig.subplots_adjust(wspace=0.5)
    fig.suptitle("$E_{gap}$" + f"= {Egap} eV, T = {Tcell} K, {spectrum_type} Spectrum", y=1.1)

def plot_db_combined_interactive(Tmin=1, Tmax=1000, Emin=0.1, Emax=3.1):
    """
    Creates an interactive photovoltaic detailed-balance visualization dashboard.

    This function builds an interactive Jupyter widget interface that allows users to
    explore photovoltaic performance metrics as a function of bandgap, temperature,
    and illumination spectrum. It dynamically updates combined plots including photon
    flux above bandgap, IV/power curves, and Shockley-Queisser efficiency limits.

    Parameters:
        Tmin(int): minimum allowed operating temperature in Kelvin.
        Tmax(int): maximum allowed operating temperature in Kelvin.
        Emin(float): minimum bandgap energy in eV.
        Emax(float): maximum bandgap energy in eV.

    Returns:
        None
    """
    plt.close("all")
    fig, axes = plt.subplots(1,3, figsize=(12,3), dpi=120, constrained_layout=True)
    ax2 = axes[1].twinx()

    fig.canvas.header_visible = False      # hides the figure “header” in JupyterLab
    fig.canvas.footer_visible = False      # hides the footer
    fig.canvas.toolbar_visible = False     # hides the toolbar

    # wrapping that clear axes and redraws combined DB plots

    def plot_combined_wrapper(Egap, Tcell, spectrum_type):

        for ax in fig.axes:
            ax.clear()
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")

        spectrum = db_fom.load_spectrum(spectrum_type=spectrum_type)
        photon_spectrum = db_fom.convert_spectrum(spectrum)
        
        plot_db_combined(photon_spectrum, Egap, Tcell, spectrum_type, fig=fig, axes=axes, ax2=ax2)



    spectra = ["AM1.5", "Fluorescent", "Blue LED", "Green LED", "Red LED", "White LED", "IR LED", "Photopic"]
    widget_layout = layout=widgets.Layout(width='800px')
    widget_style = {'description_width': '200px'}

    Tmid = Tmin+((Tmax-Tmin)/2)
    Tstart = 300 if Tmin <= 300 <= Tmax else Tmid
    
    Estart = Emin+((Emax-Emin)/2)

    temp_slider = widgets.IntSlider(value=Tstart, min=Tmin, max=Tmax, step=1, description="Operating Temperature (K)", layout=widget_layout, style=widget_style)
    gap_slider = widgets.FloatSlider(value=Estart, min=Emin, max=Emax, step=0.05, description="Band Gap (eV)", layout=widget_layout, style=widget_style)
    spectrum_drop = widgets.Dropdown(options=spectra, value=spectra[0], description="Optical Spectrum", style=widget_style)

    interact(plot_combined_wrapper, Egap=gap_slider, Tcell=temp_slider, spectrum_type=spectrum_drop)