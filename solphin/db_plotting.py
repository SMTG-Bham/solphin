import solphin.db_fom as db_fom
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(level=logging.INFO)

def photons_above_bandgap_plot(spectrum, Egap, ax=None):
    """Plot of photons above bandgap as a function of bandgap"""
    a = np.copy(spectrum)
    for row in a:
        # print row
        row[1] = db_fom.photons_above_bandgap(row[0], spectrum)

    canvas = ax if ax else plt
    
    canvas.plot(a[:, 0], a[:, 1], color='#231123')

    p_above_1_1 = db_fom.photons_above_bandgap(Egap, spectrum)
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
    """Plots the ideal IV curve, or the ideal power for a given material"""
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

def iv_curve_plot_2(spectrum, Egap, Tcell, power=False, ax1=None, ax2=None):
    """Plots the ideal IV curve, and the ideal power for a given material"""
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

    if not fig:
        fig, axes = plt.subplots(1,3, figsize=(14,3), dpi=120)
        ax2 = axes[1].twinx()

    photons_above_bandgap_plot(spectrum, Egap, ax=axes[0])
    iv_curve_plot_2(spectrum, Egap, Tcell, ax1=axes[1], ax2=ax2)
    sq_limit_plot(spectrum, Egap, Tcell, ax=axes[2])

    fig.subplots_adjust(wspace=0.5)
    fig.suptitle("$E_{gap}$" + f"= {Egap} eV, T = {Tcell} K, {spectrum_type} Spectrum", y=1.1)

def plot_db_combined_interactive(Tmin=1, Tmax=1000, Emin=0.1, Emax=3.1):
    plt.close("all")
    fig, axes = plt.subplots(1,3, figsize=(12,3), dpi=120, constrained_layout=True)
    ax2 = axes[1].twinx()

    fig.canvas.header_visible = False      # hides the figure “header” in JupyterLab
    fig.canvas.footer_visible = False      # hides the footer
    fig.canvas.toolbar_visible = False     # hides the toolbar

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