"""Density-of-states effective masses from VASP or CASTEP output, and DOS calculation setup.

The VASP paths read ``vasprun.xml``; the ``code="castep"`` paths read a
``<seed>.bands`` file, histogrammed into a DOS by sumo. The effective-mass
fit itself is shared, so its parabolic-band assumptions - and the gapped
material it presumes, since the CASTEP loader references energies to the
valence band maximum - apply to both codes alike.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.constants as sc
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.inputs import Kpoints
from scipy.constants import physical_constants as pc
from sumo.cli.dosplot import dosplot
from sumo.io.castep import read_bands_header
from sumo.io.castep import read_dos as castep_read_dos

from solphin.castep_inputs import write_castep_calculation
from solphin.pv_fom import SAMPLED_RANGES
from solphin.vasp_inputs import write_vasp_calculation

HBAR = sc.hbar  # J·s
M_E = pc["atomic unit of mass"][0]  # kg
EV = sc.e  # J per eV
BOHR_M = pc["Bohr radius"][0]  # m
MIN_DOS_FIT_R2 = 0.80
MIN_DOS_FIT_POINTS = 10


def plot_dos(
        filename: str | Path,
        xmin: float = -3,
        xmax: float = 3,
        gaussian: float = 0.05,
        save: bool = False,
        out_directory: str | Path = ".",
        code: str = "vasp",
) -> None:
    """Plot the electronic density of states from a calculation output file.

    Parameters
    ----------
    filename : str or Path
        Path to the DOS data file to plot: ``vasprun.xml`` for VASP, a
        ``<seed>.bands`` file for CASTEP (sibling ``.pdos_bin`` and ``.cell``
        files are picked up automatically when present).
    xmin : float, optional
        Minimum energy in eV shown on the x-axis. Default is ``-3``.
    xmax : float, optional
        Maximum energy in eV shown on the x-axis. Default is ``3``.
    gaussian : float, optional
        Width of the Gaussian broadening applied to the DOS in eV.
        Default is ``0.05``.
    save : bool, optional
        Save the figure as ``dos.png``. Default is ``False``.
    out_directory : str or Path, optional
        Directory the figure is written into when ``save`` is ``True``.
        Default is ``"."``, the current working directory.
    code : str, optional
        Which code produced the file, ``"vasp"`` or ``"castep"``. Default is
        ``"vasp"``.
    """
    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    dosplot(filename=filename, code=code, xmin=xmin, xmax=xmax, gaussian=gaussian, plt=plt)

    if save:
        plt.savefig(Path(out_directory) / "dos.png")

    plt.show()
    return


# --- Density-of-states effective-mass classes ---


@dataclass
class _DOSEffectiveMassResult:
    """Result of a DOS effective-mass fit for one carrier.

    Attributes
    ----------
    carrier : str
        Carrier type, ``"electrons"`` or ``"holes"``.
    m_eff_rel : float
        DOS effective mass relative to the free electron mass.
    m_eff_si : float
        DOS effective mass in kg.
    fit_quality : float
        Coefficient of determination (R²) of the fit.
    n_points : int
        Number of DOS points used in the fit.
    E_edge : float
        Energy of the fitted band edge in eV.
    energy_window : float
        Energy window used for the fit in eV.
    fit_coefficient : float
        Fitted coefficient relating the DOS to the square root of energy.
    """

    carrier: str
    m_eff_rel: float
    m_eff_si: float
    fit_quality: float
    n_points: int
    E_edge: float
    energy_window: float
    fit_coefficient: float

    @property
    def E_c(self) -> float:
        """Energy of the fitted band edge in eV, an alias of ``E_edge``."""
        return self.E_edge

    def __str__(self) -> str:
        """Return a formatted summary of the effective-mass fit."""
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
    """Combined DOS analysis result for a material.

    Attributes
    ----------
    fit_quality_e : float or None
        R² of the electron-edge fit; None if not calculated.
    fit_quality_h : float or None
        R² of the hole-edge fit; None if not calculated.
    cell_volume_m3 : float
        Unit-cell volume in m³.
    carrier : str
        Primary carrier type, ``"electrons"`` or ``"holes"``. Selects which
        band edge and which fit the summary and :attr:`em_result` report; it
        does not select :attr:`final_result`.
    final_result : float
        DOS effective mass entering Γₚᵥ, relative to the free electron mass:
        the geometric average √(mₑm_h) of supplementary equation (S6) of
        Crovetto 2024. Falls back to whichever single carrier fitted, with a
        warning, when only one edge could be fitted, and is the user-supplied
        mass when ``m_eff`` was given.
    em_electrons : _DOSEffectiveMassResult or None
        Electron fit result; None if not calculated.
    em_holes : _DOSEffectiveMassResult or None
        Hole fit result; None if not calculated.
    cbm : float or None
        Conduction band minimum in eV, referenced to VBM = 0.
    vbm : float or None
        Valence band maximum in eV, zero by construction.
    """

    fit_quality_e: float | None
    fit_quality_h: float | None
    cell_volume_m3: float
    carrier: str
    final_result: float
    em_electrons: _DOSEffectiveMassResult | None = None
    em_holes: _DOSEffectiveMassResult | None = None
    cbm: float | None = None
    vbm: float | None = None

    @property
    def em_result(self) -> _DOSEffectiveMassResult | None:
        """Fit result for the carrier named by :attr:`carrier`, or None."""
        if self.carrier == "electrons":
            return self.em_electrons

        return self.em_holes

    def __str__(self) -> str:
        """Return the formatted multi-line DOS summary."""
        return _format_dos_summary(self)


def _load_dos(
        dos_vasprun: str
) -> tuple[Vasprun, CompleteDos, NDArray, NDArray]:
    """Load the electronic density of states from a VASP calculation output file.

    The total DOS is obtained by summing the densities over all available
    spin channels.

    Parameters
    ----------
    dos_vasprun : str
        Path to the VASP ``vasprun.xml`` file with density of states data.

    Returns
    -------
    vr : Vasprun
        Parsed VASP calculation output.
    cdos : CompleteDos
        Complete density of states object with total and projected DOS.
    energies : numpy.ndarray
        Energy values corresponding to the DOS in eV.
    densities : numpy.ndarray
        Total electronic DOS summed over all spin channels.
    """
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


def _load_dos_castep(
        bands_file: str, bin_width: float = 0.01
) -> tuple[Dos, NDArray, NDArray, float]:
    """Load the electronic density of states from a CASTEP .bands file.

    sumo histograms the eigenvalues into raw state counts per bin with no
    spin-degeneracy factor, so two unit conversions here are load-bearing:
    dividing by the bin width turns counts into states per eV, and
    single-spin data is doubled to match the spin-summed convention of the
    VASP loader. The cell volume comes from the .bands header lattice, which
    is in Bohr.

    Parameters
    ----------
    bands_file : str
        Path to the CASTEP ``<seed>.bands`` file with the eigenvalue data.
    bin_width : float, optional
        Width of the DOS histogram bins in eV. Default is ``0.01``.

    Returns
    -------
    dos_obj : Dos
        Density of states with the Fermi level referenced to the valence
        band maximum for gapped systems.
    energies : numpy.ndarray
        Energy values corresponding to the DOS in eV.
    densities : numpy.ndarray
        Total electronic DOS in states per eV, summed over spin channels
        and including spin degeneracy.
    volume_m3 : float
        Unit-cell volume in m³.
    """
    dos_obj, _ = castep_read_dos(
        bands_file,
        bin_width=bin_width,
        efermi_to_vbm=True,
        total_only=True,
    )

    energies = np.asarray(dos_obj.energies, dtype=float)

    densities = np.zeros_like(energies, dtype=float)
    for spin_density in dos_obj.densities.values():
        densities += np.asarray(spin_density, dtype=float)

    # states per bin -> states per eV
    densities /= bin_width

    # CASTEP eigenvalues carry no spin degeneracy; VASP's spin-restricted
    # densities do, and the sqrt(E) fit is calibrated against that convention.
    if Spin.down not in dos_obj.densities:
        densities *= 2.0

    header = read_bands_header(bands_file)
    lattice_bohr = np.asarray(header["lattice_vectors"], dtype=float)
    volume_m3 = float(abs(np.linalg.det(lattice_bohr))) * BOHR_M ** 3

    return dos_obj, energies, densities, volume_m3


def _load_dos_data(
        filename: str, code: str = "vasp", bin_width: float = 0.01
) -> tuple[Dos, NDArray, NDArray, float]:
    """Load DOS data from either supported code onto one common contract.

    Parameters
    ----------
    filename : str
        Path to the DOS data file: ``vasprun.xml`` for VASP, a
        ``<seed>.bands`` file for CASTEP.
    code : str, optional
        Which code produced the file, ``"vasp"`` or ``"castep"``. Default is
        ``"vasp"``.
    bin_width : float, optional
        CASTEP only: histogram bin width in eV. Default is ``0.01``.

    Returns
    -------
    dos_obj : Dos
        Density of states object providing the band edges.
    energies : numpy.ndarray
        Energy values corresponding to the DOS in eV.
    densities : numpy.ndarray
        Spin-summed total DOS in states per eV.
    volume_m3 : float
        Unit-cell volume in m³.

    Raises
    ------
    ValueError
        If ``code`` is not ``"vasp"`` or ``"castep"``.
    """
    if code == "vasp":
        vr, cdos, energies, densities = _load_dos(filename)
        return cdos, energies, densities, vr.final_structure.volume * sc.angstrom ** 3
    if code == "castep":
        return _load_dos_castep(filename, bin_width=bin_width)
    raise ValueError(f"Unsupported code {code!r}; expected 'vasp' or 'castep'.")


def _get_band_edge(
        cdos: Dos,
        carrier: str,
        energies: NDArray,
        energy_window: float,
        densities: NDArray,
) -> tuple[float, NDArray, NDArray]:
    """Extract the electronic density of states near a selected band edge.

    Uses the conduction band minimum for electrons and the valence band
    maximum otherwise; energies are returned relative to the selected edge.

    Parameters
    ----------
    cdos : Dos
        Density of states object with the band edge information.
    carrier : str
        Charge carrier type; ``"electrons"`` selects the CBM, anything else
        the VBM.
    energies : numpy.ndarray
        Energy values corresponding to the DOS in eV.
    energy_window : float
        Energy range from the selected band edge over which the DOS is
        extracted, in eV.
    densities : numpy.ndarray
        Electronic density of states at the supplied energy values.

    Returns
    -------
    E_edge : float
        Energy of the selected band edge in eV.
    delta_E_ev : numpy.ndarray
        Energy values relative to the selected band edge in eV.
    dos : numpy.ndarray
        Electronic density of states within the selected energy window.
    """
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


def _clean_dos_values(
        delta_E_ev: NDArray,
        dos: NDArray,
        min_dos: float,
        energy_window: float,
) -> tuple[NDArray, NDArray]:
    """Clean and validate density of states data near a band edge.

    Removes non-finite values, non-positive relative energies and DOS values
    below the threshold.

    Parameters
    ----------
    delta_E_ev : numpy.ndarray
        Energy values relative to the selected band edge in eV.
    dos : numpy.ndarray
        Electronic density of states at the relative energy values.
    min_dos : float
        Minimum DOS value required for a data point to be retained.
    energy_window : float
        Energy range from the band edge used to select the DOS data, in eV.
        Only used when reporting insufficient valid data points.

    Returns
    -------
    delta_E_ev : numpy.ndarray
        Valid relative energy values in eV.
    dos : numpy.ndarray
        Valid electronic density of states values.

    Raises
    ------
    ValueError
        If fewer than three usable DOS data points remain after filtering.
    """
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


def _convert_dos(
        volume_m3: float, delta_E_ev: NDArray, dos: NDArray
) -> tuple[NDArray, NDArray]:
    """Convert density of states data from eV-and-cell units to SI units.

    Energies go from eV to J, and the DOS is normalised by the unit-cell
    volume.

    Parameters
    ----------
    volume_m3 : float
        Unit-cell volume in m³.
    delta_E_ev : numpy.ndarray
        Energy values relative to the selected band edge in eV.
    dos : numpy.ndarray
        Electronic density of states in states per eV.

    Returns
    -------
    delta_E_J : numpy.ndarray
        Energy values relative to the selected band edge in J.
    dos_si : numpy.ndarray
        Electronic density of states in states J⁻¹ m⁻³.
    """
    delta_E_J = (
            delta_E_ev * EV
    )

    dos_si = (
            dos
            / EV
            / volume_m3
    )

    return delta_E_J, dos_si


def _check_fit(A: float, x: NDArray, y: NDArray) -> float:
    """Evaluate the quality of a linear fit constrained through the origin.

    Parameters
    ----------
    A : float
        Fitted proportionality constant, the slope of the model ``y = A x``.
    x : numpy.ndarray
        Independent variable values used in the fit.
    y : numpy.ndarray
        Observed dependent variable values compared with the fitted model.

    Returns
    -------
    float
        Coefficient of determination (R²) of the fit; 1 is a perfect fit.
    """
    y_pred = (
            A * x
    )

    ss_res = np.sum(
        (y - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = 1.0

    return r2


def _calculate_DOS(
        delta_E_J: NDArray, dos_si: NDArray
) -> tuple[float, float, float, NDArray, float]:
    """Calculate the DOS effective mass from DOS data near a band edge.

    Fits the density of states to the square-root energy dependence expected
    for a three-dimensional parabolic band.

    Parameters
    ----------
    delta_E_J : numpy.ndarray
        Energy values relative to the selected band edge in J.
    dos_si : numpy.ndarray
        Electronic density of states in states J⁻¹ m⁻³.

    Returns
    -------
    m_eff_rel : float
        DOS effective mass relative to the free electron mass.
    m_eff_si : float
        DOS effective mass in kg.
    r2 : float
        Coefficient of determination (R²) of the DOS fit.
    y : numpy.ndarray
        DOS values used as the dependent variable in the fit.
    A : float
        Fitted coefficient relating the DOS to the square root of energy.

    Raises
    ------
    ValueError
        If the energy data have zero spread or the fitted DOS coefficient is
        non-positive.
    """
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
            HBAR ** 2
            / 2.0
            * (
                    2.0
                    * np.pi ** 2
                    * A
            ) ** (2.0 / 3.0)
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
        code: str = "vasp",
        bin_width: float = 0.01,
) -> _DOSEffectiveMassResult:
    """Calculate the density-of-states effective mass for electrons or holes.

    Extracts the DOS near the selected band edge from the calculation
    output, converts it to SI units, and fits the square-root energy
    dependence expected for a three-dimensional parabolic band.

    Parameters
    ----------
    dos_vasprun : str
        Path to the DOS data file: ``vasprun.xml`` for VASP, a
        ``<seed>.bands`` file for CASTEP.
    carrier : str, optional
        Charge carrier to calculate the effective mass for, either
        ``"electrons"`` or ``"holes"``. Default is ``"electrons"``.
    energy_window : float, optional
        Energy range from the selected band edge over which the DOS is
        fitted, in eV. Default is ``0.15``.
    min_dos : float, optional
        Minimum DOS value required for a data point to enter the fit.
        Default is ``0.0``.
    code : str, optional
        Which code produced the file, ``"vasp"`` or ``"castep"``. Default is
        ``"vasp"``.
    bin_width : float, optional
        CASTEP only: width of the DOS histogram bins in eV, the CASTEP
        analogue of VASP's NEDOS density. Ignored for VASP. Default is
        ``0.01``.

    Returns
    -------
    _DOSEffectiveMassResult
        The calculated DOS effective mass and fitting information.

    Raises
    ------
    ValueError
        If ``carrier`` is not ``"electrons"`` or ``"holes"``, if fewer than
        three usable DOS points remain after filtering, if the selected
        energy data have zero spread, or if the fitted DOS coefficient is
        non-positive.
    """
    if carrier not in ("electrons", "holes"):
        raise ValueError(
            f"Carrier must be 'electrons' or 'holes' recieved: {carrier!r}"
        )

    dos_obj, energies, densities, volume_m3 = _load_dos_data(
        dos_vasprun, code=code, bin_width=bin_width
    )

    E_edge, delta_E_ev, dos = _get_band_edge(dos_obj, carrier, energies, energy_window, densities)

    delta_E_ev, dos = _clean_dos_values(delta_E_ev, dos, min_dos, energy_window)

    delta_E_J, dos_si = _convert_dos(volume_m3, delta_E_ev, dos)

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


def test_dos_mass_windows(
        dos_vasprun: str,
        carrier: str = "electrons",
        windows: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30),
        min_dos: float = 0.0,
        code: str = "vasp",
        bin_width: float = 0.01,
) -> list[_DOSEffectiveMassResult]:
    """Test the sensitivity of the DOS effective mass to the fitting window.

    Failed fits are reported without interrupting the remaining windows.

    Parameters
    ----------
    dos_vasprun : str
        Path to the DOS data file: ``vasprun.xml`` for VASP, a
        ``<seed>.bands`` file for CASTEP.
    carrier : str, optional
        Charge carrier to calculate the effective mass for, either
        ``"electrons"`` or ``"holes"``. Default is ``"electrons"``.
    windows : tuple of float, optional
        Energy windows in eV to test the fit sensitivity over.
        Default is ``(0.05, 0.10, 0.15, 0.20, 0.30)``.
    min_dos : float, optional
        Minimum DOS value required for a data point to enter each fit.
        Default is ``0.0``.
    code : str, optional
        Which code produced the file, ``"vasp"`` or ``"castep"``. Default is
        ``"vasp"``.
    bin_width : float, optional
        CASTEP only: width of the DOS histogram bins in eV. Ignored for
        VASP. Default is ``0.01``.

    Returns
    -------
    list of _DOSEffectiveMassResult
        One result per successful fitting window; failed windows are
        omitted.

    Notes
    -----
    A convergence table is printed to standard output with the fitting
    window, effective mass relative to the free electron mass, R² value,
    and number of fitted DOS points for each successful calculation.
    """
    print("")
    print("=" * 78)

    print(
        f"  FOM DOS effective-mass convergence "
        f"({carrier})"
    )

    print("=" * 78)

    print(
        f"{'Window / eV':>12} "
        f"{'m_DOS / m_e':>15} "
        f"{'R²':>12} "
        f"{'N':>8}"
    )

    print("-" * 78)

    results = []

    for window in windows:

        try:

            result = get_dos_effective_mass(
                dos_vasprun=dos_vasprun,
                carrier=carrier,
                energy_window=window,
                min_dos=min_dos,
                code=code,
                bin_width=bin_width,
            )

            print(
                f"{window:12.4f} "
                f"{result.m_eff_rel:15.6f} "
                f"{result.fit_quality:12.6f} "
                f"{result.n_points:8d}"
            )

            results.append(
                result
            )

        except ValueError as exc:

            print(
                f"{window:12.4f} "
                f"FAILED: {exc}"
            )

    print("=" * 78)
    print("")

    return results


def _format_em_table(
        em: _DOSEffectiveMassResult,
        edge: float | None,
        is_dos_carrier: bool,
        fit: float | None,
) -> list[str]:
    """Format DOS effective-mass results for display in a summary table.

    Parameters
    ----------
    em : _DOSEffectiveMassResult
        DOS effective-mass result with the calculated mass and fit details.
    edge : float or None
        Energy of the band edge the effective mass was fitted at, in eV. If
        None, the edge is displayed as ``"N/A"``.
    is_dos_carrier : bool
        If True, mark this carrier as the primary one in the output.
    fit : float or None
        Fit quality as the coefficient of determination (R²). If None, the
        fit quality is displayed as ``"N/A"``.

    Returns
    -------
    list of str
        Formatted lines with the DOS effective-mass results.
    """
    edge_label = (
        "CBM"
        if em.carrier == "electrons"
        else "VBM"
    )

    sub = (
        "ₑ"
        if em.carrier == "electrons"
        else "ₕ"
    )

    marker = (
        "  ← primary carrier"
        if is_dos_carrier
        else ""
    )

    fit_text = (
        f"{fit:.6f}"
        if fit is not None
        else "N/A"
    )

    edge_text = (
        f"{edge:.3f} eV"
        if edge is not None
        else "N/A"
    )

    return [
        f"  {em.carrier.capitalize()} "
        f"(fitted at {edge_label}: {edge_text})",

        f"  {'DOS effective mass':<22}: "
        f"{em.m_eff_rel:.6f} m{sub}"
        f"  ({em.m_eff_si:.3e} kg)"
        f"{marker}",

        f"  {'Fit quality':<22}: "
        f"{fit_text} R²",

        f"  {'Points fitted':<22}: "
        f"{em.n_points}",

        f"  {'Energy window':<22}: "
        f"{em.energy_window:.4f} eV",

        "",
    ]


def _format_dos_summary(
        result: DOSResult,
) -> str:
    """Format a complete density-of-states result summary for display.

    Parameters
    ----------
    result : DOSResult
        DOS result with the carrier type, band-edge energies, cell volume,
        effective masses and fit-quality information.

    Returns
    -------
    str
        Formatted multi-line summary of the DOS calculation and available
        effective-mass results.
    """
    edge_label = (
        "CBM"
        if result.carrier == "electrons"
        else "VBM"
    )

    edge_value = (
        result.cbm
        if result.carrier == "electrons"
        else result.vbm
    )

    edge_text = (
        f"{edge_value:.3f} eV"
        if edge_value is not None
        else "N/A"
    )

    lines = [
        "",
        "=" * 60,
        "  DOS Result Summary",
        "=" * 60,
        f"  Primary carrier     : "
        f"{result.carrier.capitalize()}",
        f"  Band edge ({edge_label})     : "
        f"{edge_text}",
        f"  Cell volume         : "
        f"{result.cell_volume_m3:.3e} m³",
        "",
        "  ── FOM DOS Effective Masses "
        + "─" * 23,
    ]

    if result.em_electrons is not None:

        lines += _format_em_table(
            result.em_electrons,
            result.cbm,
            is_dos_carrier=(
                    result.carrier == "electrons"
            ),
            fit=result.fit_quality_e,
        )

    else:

        lines.append(
            "  Electrons: DOS effective mass not calculated."
        )

    lines.append("")

    if result.em_holes is not None:

        lines += _format_em_table(
            result.em_holes,
            result.vbm,
            is_dos_carrier=(
                    result.carrier == "holes"
            ),
            fit=result.fit_quality_h,
        )

    else:

        lines.append(
            "  Holes: DOS effective mass not calculated."
        )

    # The value that actually reaches the figure of merit, printed beside the
    # two masses it is formed from.
    lines += [
        "",
        f"  Γₚᵥ DOS mass √(mₑm_h) : {result.final_result:.6f} m₀",
        "=" * 60,
        "",
    ]

    return "\n".join(
        lines
    )


def print_dos_summary(
        result: DOSResult,
) -> None:
    """Print a formatted density-of-states result summary to standard output.

    Parameters
    ----------
    result : DOSResult
        DOS result with the density-of-states and effective-mass information
        to display.
    """
    print(
        _format_dos_summary(
            result
        )
    )


def _check_dos_fit_quality(
        result: _DOSEffectiveMassResult,
        dos_vasprun: str,
        windows: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40),
        min_dos: float = 0.0,
        code: str = "vasp",
        bin_width: float = 0.01,
) -> bool:
    """Check whether a DOS effective-mass fit is sufficiently well resolved.

    A poorly resolved fit prints a warning with recommendations and a
    fitting-window convergence test.

    Parameters
    ----------
    result : _DOSEffectiveMassResult
        DOS effective-mass result with the fitted mass, fit quality, number
        of fitted points and carrier type.
    dos_vasprun : str
        Path to the DOS data file, used for the fitting-window test when the
        fit is poorly resolved.
    windows : tuple of float, optional
        Energy windows in eV for the sensitivity test of a poorly resolved
        fit. Default is ``(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)``.
    min_dos : float, optional
        Minimum DOS value required for a data point to enter the
        fitting-window tests. Default is ``0.0``.
    code : str, optional
        Which code produced the file, ``"vasp"`` or ``"castep"``. Default is
        ``"vasp"``.
    bin_width : float, optional
        CASTEP only: width of the DOS histogram bins in eV. Ignored for
        VASP. Default is ``0.01``.

    Returns
    -------
    bool
        True if the fit passes the quality check; False if it is poorly
        resolved and a warning was issued.
    """
    poor_fit = (
            result.fit_quality < MIN_DOS_FIT_R2
            and result.n_points < MIN_DOS_FIT_POINTS
    )

    if not poor_fit:
        return True

    print("")
    print("!" * 70)
    print("  WARNING: DOS EFFECTIVE-MASS FIT IS POORLY RESOLVED")
    print("!" * 70)

    print(
        f"  Carrier            : {result.carrier}"
    )

    print(
        f"  Fitted mass        : "
        f"{result.m_eff_rel:.6f} m_e"
    )

    print(
        f"  Fit quality        : "
        f"R² = {result.fit_quality:.6f}"
    )

    print(
        f"  DOS points fitted  : "
        f"{result.n_points}"
    )

    print("")
    print(
        "  This DOS calculation does not contain enough "
        "well-resolved"
    )
    print(
        "  near-edge DOS data to determine the Crovetto "
        "DOS effective"
    )
    print(
        "  mass reliably."
    )

    print("")
    print(
        "  Consider either:"
    )
    print(
        "    1. Re-running the DOS calculation with a "
        "denser regular k-point grid."
    )
    print(
        "    2. Supplying an alternative effective-mass "
        "estimate if a denser"
    )
    print(
        "       hybrid-functional calculation is "
        "prohibitively expensive."
    )

    print("")
    print(
        "  The current fitted mass should therefore be "
        "treated as provisional."
    )

    print("")
    print(
        "  Fitting-window behaviour:"
    )

    test_dos_mass_windows(
        dos_vasprun=dos_vasprun,
        carrier=result.carrier,
        windows=windows,
        min_dos=min_dos,
        code=code,
        bin_width=bin_width,
    )

    return False


def compute_dos(
        dos_vasprun: str,
        m_eff: float | None = None,
        carrier: str = "electrons",
        energy_window: float = 0.15,
        min_dos: float = 0.0,
        code: str = "vasp",
        bin_width: float = 0.01,
) -> DOSResult:
    """Compute density-of-states information and the corresponding effective mass.

    Parses the calculation output, extracts the band edges, and fits the
    DOS effective masses for electrons and holes. The mass reported as the
    final result is their geometric average √(mₑm_h), which is the quantity
    supplementary equation (S6) of Crovetto 2024 defines as the DOS effective
    mass entering Γₚᵥ. A user-supplied effective mass, when given, is used as
    the final result instead of fitting.

    Parameters
    ----------
    dos_vasprun : str
        Path to the DOS data file: ``vasprun.xml`` for VASP, a
        ``<seed>.bands`` file for CASTEP.
    m_eff : float or None, optional
        User-supplied effective mass relative to the free electron mass.
        If provided, DOS effective-mass fitting is skipped. Default is None.
    carrier : str, optional
        Primary charge carrier, either ``"electrons"`` or ``"holes"``. Selects
        the band edge reported and the fit exposed as
        :attr:`DOSResult.em_result`; both masses are fitted either way, and
        both enter the geometric average. Default is ``"electrons"``.
    energy_window : float, optional
        Energy range from each band edge over which the effective-mass fits
        are performed, in eV. Default is ``0.15``.
    min_dos : float, optional
        Minimum DOS value required for a data point to enter the fits.
        Default is ``0.0``.
    code : str, optional
        Which code produced the file, ``"vasp"`` or ``"castep"``. Default is
        ``"vasp"``.
    bin_width : float, optional
        CASTEP only: width of the DOS histogram bins in eV, the CASTEP
        analogue of VASP's NEDOS density. Ignored for VASP. Default is
        ``0.01``.

    Returns
    -------
    DOSResult
        The cell volume, zero-referenced band-edge energies, the geometric
        average DOS effective mass of equation (S6), per-carrier fit results
        and fit-quality information.

    Raises
    ------
    ValueError
        If ``carrier`` is not ``"electrons"`` or ``"holes"``, or if neither
        carrier effective mass can be calculated when no user-supplied
        effective mass is provided.

    Warns
    -----
    UserWarning
        If the electron or hole DOS effective-mass calculation fails. The
        calculation for the other carrier is still attempted, and a single
        fitted mass is then reported in place of the geometric average. Also
        if the resulting mass falls outside the range sampled by Crovetto 2024
        table 1, which Γₚᵥ will refuse.
    """
    if carrier not in (
            "electrons",
            "holes",
    ):
        raise ValueError(
            "carrier must be "
            "'electrons' or 'holes'."
        )

    dos_obj, _, _, vol_m3 = _load_dos_data(
        dos_vasprun, code=code, bin_width=bin_width
    )

    cbm, vbm = (
        dos_obj.get_cbm_vbm()
    )

    # Reference energies to VBM = 0
    cbm_zeroed = (
            cbm - vbm
    )

    vbm_zeroed = 0.0

    em_electrons = None
    em_holes = None

    fit_quality_e = None
    fit_quality_h = None

    if m_eff is None:

        print(
            "  Computing electron "
            "FOM DOS effective mass..."
        )

        try:

            em_electrons = (
                get_dos_effective_mass(
                    dos_vasprun=dos_vasprun,
                    carrier="electrons",
                    energy_window=energy_window,
                    min_dos=min_dos,
                    code=code,
                    bin_width=bin_width,
                )
            )

            fit_quality_e = (
                em_electrons.fit_quality
            )

        except Exception as exc:

            warnings.warn(
                "Electron DOS effective-mass fit failed: "
                f"{exc}",
                UserWarning,
                stacklevel=2,
            )

        print(
            "  Computing hole "
            "FOM DOS effective mass..."
        )

        try:

            em_holes = (
                get_dos_effective_mass(
                    dos_vasprun=dos_vasprun,
                    carrier="holes",
                    energy_window=energy_window,
                    min_dos=min_dos,
                    code=code,
                    bin_width=bin_width,
                )
            )

            fit_quality_h = (
                em_holes.fit_quality
            )

        except Exception as exc:

            warnings.warn(
                "Hole DOS effective-mass fit failed: "
                f"{exc}",
                UserWarning,
                stacklevel=2,
            )

    if m_eff is not None:

        final_result = float(
            m_eff
        )

    elif em_electrons is not None and em_holes is not None:

        # Supplementary equation (S6) of Crovetto 2024: the mass entering Γₚᵥ is
        # the geometric average of the two DOS masses, because the quasi-Fermi
        # level splitting depends on the N_c N_v product, which goes as
        # (m_e m_h)^(3/2).
        final_result = float(
            np.sqrt(em_electrons.m_eff_rel * em_holes.m_eff_rel)
        )

    else:

        # At most one edge fitted. Each carrier is tested on its own rather than
        # as `em_electrons if em_holes is None else em_holes`: mypy narrows one
        # Optional at a time and cannot infer from the branch above that exactly
        # one of the pair is None, so that expression stays typed as possibly
        # None and its attribute reads are rejected.
        if em_electrons is not None:

            fitted, missing = em_electrons, "hole"

        elif em_holes is not None:

            fitted, missing = em_holes, "electron"

        else:

            raise ValueError(
                "Neither the electron nor the hole DOS effective mass could be "
                "calculated, so the geometric average of Crovetto 2024 equation "
                "(S6) has no ingredients."
            )

        # One edge fitted, so the geometric average is unavailable. Falling back
        # to the carrier that did fit keeps a usable number, but it is not the
        # quantity Γₚᵥ was fitted against, so say so.
        warnings.warn(
            f"The {missing} DOS effective-mass fit is unavailable, so the "
            "geometric average √(mₑm_h) of Crovetto 2024 equation (S6) cannot "
            f"be formed; falling back to the {fitted.carrier} mass "
            f"{fitted.m_eff_rel:.6f} m₀ alone.",
            UserWarning,
            stacklevel=2,
        )

        final_result = (
            fitted.m_eff_rel
        )

    minimum, maximum, _ = SAMPLED_RANGES["dos_mass"]

    if not minimum <= final_result <= maximum:

        warnings.warn(
            f"DOS effective mass {final_result:.6f} m₀ is outside the "
            f"{minimum} - {maximum} m₀ range sampled by Crovetto 2024 table 1;"
            " it is a property of the supplied density of states, but the Γₚᵥ"
            " figure of merit will refuse it unless"
            " allow_out_of_range=True is passed.",
            UserWarning,
            stacklevel=2,
        )

    result = DOSResult(
        fit_quality_e=fit_quality_e,
        fit_quality_h=fit_quality_h,

        cell_volume_m3=vol_m3,

        carrier=carrier,

        final_result=final_result,

        em_electrons=em_electrons,
        em_holes=em_holes,

        cbm=cbm_zeroed,
        vbm=vbm_zeroed,
    )

    if m_eff is None:

        if carrier == "electrons":
            selected_em = em_electrons
        else:
            selected_em = em_holes

        if selected_em is not None:
            _check_dos_fit_quality(
                result=selected_em,
                dos_vasprun=dos_vasprun,
                min_dos=min_dos,
                code=code,
                bin_width=bin_width,
            )

    return result


# --- Density-of-states file generation ---


def _local_kpoint_offsets(
        k0_frac: NDArray,
        mesh: tuple[int, int, int],
        delta: float,
) -> NDArray:
    """Build a uniform grid of fractional k-points around a band-edge k-point.

    Parameters
    ----------
    k0_frac : numpy.ndarray
        Fractional reciprocal-space coordinates of the central k-point.
    mesh : tuple of int
        Number of k-points along each reciprocal direction, as
        ``(nx, ny, nz)``.
    delta : float
        Maximum fractional reciprocal-space displacement from the central
        k-point along each direction.

    Returns
    -------
    numpy.ndarray
        The grid points in fractional coordinates, shape (nx*ny*nz, 3).
    """
    k0_frac = np.asarray(k0_frac, dtype=float)

    nx, ny, nz = mesh

    xs = np.linspace(-delta, delta, nx)
    ys = np.linspace(-delta, delta, ny)
    zs = np.linspace(-delta, delta, nz)

    pts = []

    for dx in xs:
        for dy in ys:
            for dz in zs:
                pts.append(
                    k0_frac + np.array([dx, dy, dz])
                )

    return np.asarray(pts)


def _generate_local_kpoints(
        k0_frac: NDArray,
        mesh: tuple[int, int, int],
        delta: float,
) -> Kpoints:
    """Generate a local reciprocal-space k-point mesh around a band-edge k-point.

    Builds a uniform three-dimensional grid centred on the given fractional
    coordinate, in reciprocal-coordinate mode with equal weights.

    Parameters
    ----------
    k0_frac : numpy.ndarray
        Fractional reciprocal-space coordinates of the central k-point.
    mesh : tuple of int
        Number of k-points along each reciprocal direction, as
        ``(nx, ny, nz)``.
    delta : float
        Maximum fractional reciprocal-space displacement from the central
        k-point along each direction.

    Returns
    -------
    Kpoints
        VASP KPOINTS object with the generated local mesh.
    """
    pts_grid = _local_kpoint_offsets(k0_frac, mesh, delta)

    return Kpoints(
        comment="Local k-mesh around band edge",
        style=Kpoints.supported_modes.Reciprocal,
        num_kpts=len(pts_grid),
        kpts=pts_grid.tolist(),
        kpts_weights=[1.0] * len(pts_grid),
    )


def write_local_kpoints(
        folder: str | Path,
        k0_frac: NDArray,
        mesh: tuple[int, int, int],
        delta: float,
) -> None:
    """Generate and write a dense local VASP KPOINTS file around a band edge.

    Parameters
    ----------
    folder : str or Path
        Calculation folder the generated ``KPOINTS`` file is written into.
    k0_frac : numpy.ndarray
        Fractional reciprocal-space coordinates of the central k-point,
        typically the CBM, VBM, or a relevant direct band-gap location.
    mesh : tuple of int
        Number of k-points along each reciprocal direction, as
        ``(nx, ny, nz)``.
    delta : float
        Maximum fractional reciprocal-space displacement from the central
        k-point along each direction.
    """
    kp = _generate_local_kpoints(k0_frac, mesh, delta)
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    kp.write_file(f"{folder_path}/KPOINTS")


def write_eff_mass(
        k0_frac: NDArray,
        structure: Structure,
        functional: str,
        encut: int,
        folder: str = "eff_mass",
        mesh: tuple[int, int, int] = (5, 5, 5),
        delta: float = 0.01,
        code: str = "vasp",
) -> None:
    """Write a calculation setup for an effective-mass calculation.

    Generates a dense local k-point mesh around the band-edge k-point and
    prepares the calculation with effective-mass settings: INCAR tags plus a
    KPOINTS mesh for VASP, or a spectral band-structure task with a
    ``spectral_kpoint_list`` for CASTEP.

    Parameters
    ----------
    k0_frac : numpy.ndarray
        Fractional reciprocal-space coordinates of the band-edge k-point.
    structure : Structure
        Crystal structure used to generate the input files.
    functional : str
        Calculation recipe or exchange-correlation functional.
    encut : int
        Plane-wave cutoff energy for the calculation in eV.
    folder : str, optional
        Folder the input files are written into. Default is ``"eff_mass"``.
    mesh : tuple of int, optional
        Number of k-points along each reciprocal direction, as
        ``(nx, ny, nz)``. Default is ``(5, 5, 5)``.
    delta : float, optional
        Maximum fractional reciprocal-space displacement from the central
        k-point along each direction. Default is ``0.01``.
    code : str, optional
        Which code to write inputs for, ``"vasp"`` or ``"castep"``. Default
        is ``"vasp"``.

    Raises
    ------
    ValueError
        If ``code`` is not ``"vasp"`` or ``"castep"``.
    """
    if code == "castep":
        rows = [
            [f"{coordinate:.8f}" for coordinate in point]
            for point in _local_kpoint_offsets(k0_frac, mesh, delta)
        ]
        write_castep_calculation(
            structure=structure,
            recipe=functional,
            out_dir=folder,
            patches=["eff_mass"],
            user_param_settings={"cut_off_energy": encut},
            user_cell_blocks={"spectral_kpoint_list": rows},
        )
        return
    if code != "vasp":
        raise ValueError(f"Unsupported code {code!r}; expected 'vasp' or 'castep'.")

    kp = _generate_local_kpoints(
        k0_frac=k0_frac,
        mesh=mesh,
        delta=delta,
    )

    write_vasp_calculation(
        structure=structure,
        recipe=functional,
        out_dir=folder,
        patches=["eff_mass"],
        user_incar_settings={"ENCUT": encut, "ISYM": 0, "ICHARG": 0},
        user_kpoints_settings=kp,
    )
