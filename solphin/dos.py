from matplotlib import pyplot as plt
from sumo.cli.dosplot import dosplot
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
import numpy as np
from numpy.typing import NDArray

import warnings
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from pymatgen.io.vasp import Vasprun
from pymatgen.core.structure import Structure
from solphin.vasp_inputs import write_vasp_calculation
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin
import scipy.constants as sc
from scipy.constants import physical_constants as pc

from solphin.band_structure import get_band_structure

HBAR = sc.hbar   # J·s
M_E  = pc["atomic unit of mass"][0]   # kg
EV   = sc.e    # J per eV


def plot_dos(filename, xmin=-3, xmax=3, gaussian=0.05, save=False):
    """
    Plots the electronic density of states (DOS) from a calculation output file.

    This function generates a density of states plot using the provided DOS data
    file and visualizes it within a predefined plotting style. Optionally, the
    plot can also be saved as a PDF file.

    Parameters:
        filename(str): Path to the DOS data file to be plotted.

        xmin(float, optional): Minimum energy value shown on the x-axis in eV.
            Default is -3.

        xmax(float, optional): Maximum energy value shown on the x-axis in eV.
            Default is 3.

        gaussian(float, optional): Width of the Gaussian broadening applied to
            the DOS in eV. Default is 0.05.

        save(bool, optional): If True, saves the generated figure as
            "dos.pdf". Default is False.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(5,3), dpi=150)
    dosplot(filename=filename, xmin=xmin, xmax=xmax, gaussian=gaussian, plt=plt)

    if save:
        plt.savefig("dos.pdf")

    plt.show()
    return

'''
Density of states effect mass classes.
'''

@dataclass
class _DOSEffectiveMassResult:

    carrier: str
    m_eff_rel: float
    m_eff_si: float
    fit_quality: float 
    n_points: int
    E_edge: float
    energy_window: float
    fit_coefficient: float 

    @property 
    def E_c(self):
        return self.E_edge

    def __str__(self):

        edge_label = (
            "CBM"
            if self.carrier == "electrons"
            else "VBM"
        )

        sub = (
            "ₑ"
            if self.carrier == "electrons"
            else "ₕ"
        )

        lines = [
            "",
            "=" * 60,
            f" DOS Effective Mass ({self.carrier.capitalize()})",
            "=" * 60,
            f" Band edge ({edge_label}) : {self.E_edge:.6f} eV",
                        f"  Energy window     : {self.energy_window:.4f} eV",
            f"  Points used       : {self.n_points}",
            f"  Fit quality       : {self.fit_quality:.6f} R²",
            "",
            f"  DOS effective mass: {self.m_eff_rel:.6f} m{sub}",
            f"                    : {self.m_eff_si:.6e} kg",
            "=" * 60,
            "",
        ]

        return "\n".join(lines)

@dataclass

class DOSResult:

    fit_quality_e: Optional[float]
    fit_quality_h: Optional[float]
    cell_volume_m3: float
    carrier: str
    final_result: float
    em_electrons: Optional[_DOSEffectiveMassResult] = None
    em_holes: Optional[_DOSEffectiveMassResult] = None
    cbm: Optional[float] = None
    vbm: Optional[float] = None

    @property 
    def em_result(self) -> Optional[_DOSEffectiveMassResult]:

        if self.carrier == "electrons":
            return self.em_electrons

        return self.em_holes

    def __str__(self):
        return _format_dos_summary(self)

def _load_dos(
        dos_vasprun: str
):

    vr = Vasprun(
        dos_vasprun,
        parse_dos=True,
        parse_eigen=False
    )

    cdos = vr.complete_dos

    energies = np.asarray(
        cdos.energies,
        dtype=float
    )

    densities = np.zeros_like(
        energies,
        dtype=float
    )

    for spin_density in cdos.densities.values():
        densities += np.asarray(
            spin_density,
            dtype=float
        )

    return vr, cdos, energies, densities

def _get_band_edge(cdos, carrier, energies, energy_window, densities):

    cbm, vbm = cdos.get_cbm_vbm()

    if carrier == "electrons":

        E_edge = cbm

        mask = (
            (energies > E_edge)
            & (energies <= E_edge + energy_window)
        )

        delta_E_ev = (
            energies[mask] - E_edge
        )

    else:

        E_edge = vbm

        mask = (
            (energies < E_edge)
            & (energies >= E_edge - energy_window)
        )

        delta_E_ev = (
            E_edge - energies[mask]
        )

    dos = densities[mask]

    return E_edge, delta_E_ev, dos

def _clean_dos_values(delta_E_ev, dos, min_dos, energy_window):

    good = (
        np.isfinite(delta_E_ev)
        & np.isfinite(dos)
        & (delta_E_ev > 0)
        & (dos > min_dos)
    )

    delta_E_ev = delta_E_ev[good]
    dos = dos[good]

    if len(dos) < 3:
        raise ValueError(
            f"Only {len(dos)} usable DOS points lie within "
            f"{energy_window:.3f} eV of the band edge."
        )

    return delta_E_ev, dos

def _convert_dos(vr, delta_E_ev, dos):

    volume_m3 = (
        vr.final_structure.volume
        * 1.0e-30
        )

    delta_E_J = (
        delta_E_ev * EV
    )

    dos_si = (
        dos
        / EV
        / volume_m3
    )

    return delta_E_J, dos_si

def _check_fit(A, x, y):

    y_pred = (
        A * x
    )

    ss_res = np.sum(
        (y - y_pred)**2
    )

    ss_tot = np.sum(
        (y - np.mean(y))**2
    )

    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = 1.0

    return r2

def _calculate_DOS(delta_E_J, dos_si):

    x = np.sqrt(
        delta_E_J
    )

    y = dos_si

    denominator = np.dot(
        x,
        x,
    )

    if denominator == 0:
        raise ValueError(
            "Cannot fit DOS effective mass: zero energy spread."
        )

    A = (
        np.dot(x, y)
        / denominator
    )

    if A <= 0:
        raise ValueError(
            "Fitted DOS coefficient is non-positive."
        )

    m_eff_si = (
        HBAR**2
        / 2.0
        * (
            2.0
            * np.pi**2
            * A
        )**(2.0 / 3.0)
    )

    m_eff_rel = (
        m_eff_si / M_E
    )

    r2 = _check_fit(A, x, y)

    return m_eff_rel, m_eff_si, r2, y, A


def get_dos_effective_mass(
        dos_vasprun: str,
        carrier: str = "electrons",
        energy_window: float = 0.15,
        min_dos: float = 0.0,
) -> _DOSEffectiveMassResult:

    if carrier not in ("electrons", "holes"):
        raise ValueError(
            f"Carrier must be 'electrons' or 'holes' recieved: {carrier!r}"
        )

    vr, cdos, energies, densities = _load_dos(dos_vasprun)

    E_edge, delta_E_ev, dos = _get_band_edge(cdos, carrier, energies, energy_window, densities)

    delta_E_ev, dos = _clean_dos_values(delta_E_ev, dos, min_dos, energy_window)

    delta_E_J, dos_si = _convert_dos(vr, delta_E_ev, dos)

    m_eff_rel, m_eff_si, r2, y, A = _calculate_DOS(delta_E_J, dos_si)

    return _DOSEffectiveMassResult(
    carrier=carrier,

    m_eff_rel=float(m_eff_rel),
    m_eff_si=float(m_eff_si),

    fit_quality=float(r2),
    n_points=len(y),

    E_edge=float(E_edge),
    energy_window=float(energy_window),

    fit_coefficient=float(A),
    )






    