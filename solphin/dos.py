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