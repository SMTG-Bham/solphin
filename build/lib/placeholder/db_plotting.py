import placeholder.db_fom as db_fom
import numpy as np
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.INFO)

def photons_above_bandgap_plot(spectrum, Egap):
    """Plot of photons above bandgap as a function of bandgap"""
    a = np.copy(spectrum)
    for row in a:
        # print row
        row[1] = db_fom.photons_above_bandgap(row[0], spectrum)
    plt.plot(a[:, 0], a[:, 1])

    p_above_1_1 = db_fom.photons_above_bandgap(Egap, spectrum)
    plt.plot([Egap], [p_above_1_1], 'ro')
    plt.text(Egap+0.05, p_above_1_1, '{}eV, {:.4}'.format(Egap, p_above_1_1))

    plt.xlabel('$E_{gap}$ (eV)')
    plt.ylabel('# Photons $m^{-2}s^{-1}$')
    plt.title('Number of above-bandgap \nphotons as a function of bandgap')
    plt.show()

def iv_curve_plot(egap, spectrum, Tcell, power=False):
    """Plots the ideal IV curve, or the ideal power for a given material"""
    v_open = db_fom.voc(egap, spectrum, Tcell)
    v = np.linspace(0, v_open)
    if power:
        p =  v * db_fom.current_density(egap, spectrum, v, Tcell)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Power generated ($W$)')
        plt.title('Power Curve')
        plt.plot(v, p)
    else:
        i =  db_fom.current_density(egap, spectrum, v, Tcell)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current density $J$ ($Am^{-2}$)')
        plt.title('IV Curve')
        plt.plot(v, i)


def iv_curve_plot_2(egap, spectrum, Tcell, power=False):
    """Plots the ideal IV curve, and the ideal power for a given material"""
    v_open = db_fom.voc(egap, spectrum, Tcell)
    v = np.linspace(0, v_open)

    fig, ax1 = plt.subplots()
    p =  v * db_fom.current_density(egap, spectrum, v, Tcell)
    i =  db_fom.current_density(egap, spectrum, v, Tcell)
    
    ax1.plot(v, i)
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current density $J$ ($Am^{-2}$)')
    ax1.legend(['Current'], loc=2)
    
    ax2 = ax1.twinx()
    ax2.plot(v, p, color='orange')
    ax2.set_ylabel('Power generated ($W$)')
    ax2.legend(['Power'], loc=3)
    return


def sq_limit_plot(spectrum, Egap, Tcell):
    # Plot the famous SQ limit
    a = np.copy(spectrum)
    # Not for whole array hack to remove divide by 0 errors
    for row in a[2:]:
        # print row
        row[1] = db_fom.max_eff(row[0], spectrum, Tcell)
    # Not plotting whole array becase some bad values happen
    plt.plot(a[2:, 0], a[2:, 1])
    e_gap = Egap
    p_above_1_1 = db_fom.max_eff(e_gap, spectrum, Tcell)
    plt.plot([e_gap], [p_above_1_1], 'ro')
    plt.text(e_gap+0.05, p_above_1_1, '{}eV, {:.4}'.format(e_gap, p_above_1_1))

    plt.xlabel('$E_{gap}$ (eV)')
    plt.ylabel('Max efficiency')
    plt.title('SQ Limit')