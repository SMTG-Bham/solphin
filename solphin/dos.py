from matplotlib import pyplot as plt
from sumo.cli.dosplot import dosplot
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
import numpy as np
from numpy.typing import NDArray

import warnings
from dataclasses import dataclass
from typing import Optional

from pymatgen.io.vasp import Vasprun
from pymatgen.core.structure import Structure
from vasp_inputs import write_vasp_calculation
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

def _generate_local_kpoints(
        k0_frac:NDArray,
        mesh:tuple,
        delta:float
    ):

    """
    Creates a dense k-mesh around the cbm/vbm single k-point.

    Parameters:
        k0_frac(array): Fractional coordinates of the CBM/VBM and/or most direct bandgap.
        mesh(tuple): Number of k-points in each direction.
        delta(float): Maximum displacement from the k-point centre in fractional reciprocal coordinates
    """

    nx, ny, nz = mesh 

    xs = np.linspace(-delta, delta, nx)
    ys = np.linspace(-delta, delta, ny)
    zs = np.linspace(-delta, delta, nz)

    pts = []

    for dx in xs:
        for dy in ys:
            for dz in zs:
                pts.append(k0_frac + np.array([dx,dy,dz]))

    return np.array(pts)

def write_local_kpoints(
        folder:str,
        k0_frac:NDArray,
        mesh:tuple,
        delta:float
        ):
    
    """
    Generates the denser KPOINTS VASP file for the kpoints around the CBM/VBM.

    Parameters:
        folder(string): location of the calculation folder.
        k0_frac(array): Fractional coordinates of the CBM/VBM and/or most direct bandgap.
        mesh(tuple): Number of k-points in each direction.
        delta(float): Maximum displacement from the k-point centre in fractional reciprocal coordinates
    """

    kpoints = _generate_local_kpoints(k0_frac, mesh, delta)

    kp = Kpoints(
        comment="Local k-mesh for effective mass",
        style=Kpoints.supported_modes.Reciprocal,
        num_kpts=len(kpoints),
        kpts=kpoints.tolist(),
        kpts_weights=[1]*len(kpoints),)
    
    kp.write_file(f"{folder}/KPOINTS")

def write_eff_mass(k0_frac:NDArray,
                   structure:Structure,
                   functional:str,
                   encut:int,
                   folder:str="eff_mass",
                    mesh:tuple=(5,5,5),
                    delta:float=0.01):
    
    write_local_kpoints(folder, k0_frac, mesh, delta)

    write_vasp_calculation(
    structure=structure, 
    recipe=functional, 
    out_dir=folder, 
    patches=["eff_mass"], 
    user_incar_settings={"ENCUT": encut})


# Density of states tables and original methodolody. 

@dataclass
class _SegmentMass:

    label:       str    # e.g. "Γ→X"
    m_eff_rel:   float  # m* / m_e
    m_eff_si:    float  # kg
    fit_quality: float  # R²
    n_points:    int    # k-points used in fit


@dataclass
class _EffectiveMassResult:

    m_eff_rel:   float       # harmonic mean m* / m_e
    m_eff_si:    float       # harmonic mean m* in kg
    segments:    list        # list[SegmentMass]
    E_edge:      float       # CBM (electrons) or VBM (holes) energy in eV
    k_edge:      np.ndarray  # edge k-point, Cartesian 1/m
    carrier:     str         # "electrons" or "holes"

    @property
    def E_c(self):

        return self.E_edge

    def __str__(self):
        edge_label = "CBM" if self.carrier == "electrons" else "VBM"
        lines = [
            "",
            "=" * 60,
            f"  Effective Mass Results ({self.carrier.capitalize()})",
            "=" * 60,
            f"  Carrier type        : {self.carrier.capitalize()}",
            f"  Band edge ({edge_label})     : {self.E_edge:.4f} eV",
            "",
            "  Anisotropic masses (per k-path segment):",
            f"  {'Segment':<18} {'m* (mₑ)':>10} {'R²':>8} {'N pts':>7}",
            "  " + "-" * 47,
        ]

        lines += [
            "  " + "-" * 47,
            f"  {'Harmonic mean':<18} {self.m_eff_rel:>10.4f}",
            "=" * 60,
        ]

        return "\n".join(lines)


@dataclass
class DOSResult:

    fit_quality_e :    Optional[float]   # mean R² for the DOS carrier
    fit_quality_h :    Optional[float]
    cell_volume_m3:    float             # m³
    carrier:           str               # "electrons" or "holes" — used in DOS
    final_result:      float   # always computed
    em_electrons:      Optional[_EffectiveMassResult] = None   # always computed 
    em_holes:          Optional[_EffectiveMassResult] = None   # always computed
    cbm:               Optional[_EffectiveMassResult] = None   # always computed
    vbm:               Optional[_EffectiveMassResult] = None   # always computed
    

    @property
    def em_result(self) -> Optional[_EffectiveMassResult]:

        if self.carrier == "electrons":
            return self.em_electrons
        return self.em_holes

    def __str__(self):
        return _format_dos_summary(self)

def _detect_spin_channel(bs: BandStructure, edge_info: dict) -> Spin:
    """
    Detects the spin channel associated with a band structure edge state.

    This function determines which spin channel should be used when analyzing
    a band edge (e.g., VBM or CBM) from a spin-polarized band structure. For
    non-spin-polarized calculations, the function defaults to `Spin.up`.

    Parameters:
        bs(BandStructure): Band structure object containing band eigenvalues
            indexed by spin channel.

        edge_info(dict): Dictionary containing band edge metadata, typically
            returned by pymatgen band structure analysis methods. The dictionary
            must contain a `"band_index"` entry with spin-resolved band indices.

    Returns:
        Spin: The detected spin channel associated with the band edge.
            Returns:
                - `Spin.up` for non-spin-polarized calculations or when the
                edge exists in the spin-up channel.
                - `Spin.down` otherwise.
    """

    if len(bs.bands) == 1:
        return Spin.up
    if Spin.up in edge_info["band_index"] and edge_info["band_index"][Spin.up]:
        return Spin.up
    return Spin.down

def _fit_segment_mass(
    kpoints_cart: np.ndarray,
    energies:     np.ndarray,
    edge_kidx:    int,
    E_edge:       float,
    label:        str,
    n_points:     int,
    carrier:      str = "electrons",
) -> _SegmentMass:
    """
    Fits a parabolic effective mass from a band edge segment.

    This function performs a local quadratic (parabolic) approximation of the
    band dispersion around a specified band edge by fitting energy versus
    k-point distance. The curvature is used to compute the effective mass for
    either electrons or holes.

    A symmetric window of k-points around the band edge is selected, and the
    dispersion is expressed as a function of squared दूरी in reciprocal space.
    A linear fit is then applied to extract the curvature.

    Parameters:
        kpoints_cart(np.ndarray): Array of k-points in Cartesian coordinates.

        energies(np.ndarray): Band energies (in eV) corresponding to each
            k-point in `kpoints_cart`.

        edge_kidx(int): Index of the k-point corresponding to the band edge
            (e.g., VBM or CBM).

        E_edge(float): Energy of the band edge in eV.

        label(str): Identifier for the k-point segment or high-symmetry line.

        n_points(int): Number of k-points to include on each side of the band
            edge for fitting.

        carrier(str, optional): Type of carrier being modeled. Must be either:
            - "electrons" (default)
            - "holes"

            For holes, the energy difference is flipped to ensure a positive
            curvature.

    Returns:
        SegmentMass: Object containing effective mass results, including:
            - label(str): Segment identifier
            - m_eff_rel(float): Effective mass in units of electron mass
            - m_eff_si(float): Effective mass in SI units (kg)
            - fit_quality(float): R² goodness-of-fit for the linear regression
            - n_points(int): Number of points used in the fit

    Notes:
        The method:
            - Uses a local window around the band edge
            - Fits E vs |k - k_edge|² using linear regression
            - Computes effective mass from band curvature:
                m* = ħ² / (2 * slope)
            - Rejects fits with fewer than 2 valid points or non-positive slope
    """

    # Check what these are exactly, where they come from and if there is a smoother way of getting it. 

    k_edge = kpoints_cart[edge_kidx]

    i_lo  = max(0, edge_kidx - n_points)
    i_hi  = min(len(kpoints_cart), edge_kidx + n_points + 1)

    seg_k = kpoints_cart[i_lo:i_hi]
    seg_E = energies[i_lo:i_hi]

    dk_sq = np.sum((seg_k - k_edge) ** 2, axis=1)

    # Flip sign for holes so dE opens upward and slope is positive
    if carrier == "holes":
        dE = (E_edge - seg_E) * EV
    else:
        dE = (seg_E - E_edge) * EV

    mask   = dk_sq > 0
    dk_sq  = dk_sq[mask]
    dE     = dE[mask]
    n_used = len(dk_sq)

    if n_used < 2:
        return _SegmentMass(
            label=label, m_eff_rel=np.nan, m_eff_si=np.nan,
            fit_quality=np.nan, n_points=n_used
        )

    slope, intercept = np.polyfit(dk_sq, dE, 1)

    if slope <= 0:

        return _SegmentMass(
            label=label, m_eff_rel=np.nan, m_eff_si=np.nan,
            fit_quality=np.nan, n_points=n_used,
        )

    m_eff_si  = HBAR**2 / (2.0 * slope)
    m_eff_rel = m_eff_si / M_E

    dE_pred = slope * dk_sq + intercept
    ss_res  = np.sum((dE - dE_pred) ** 2)
    ss_tot  = np.sum((dE - np.mean(dE)) ** 2)
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return _SegmentMass(
        label=label, m_eff_rel=m_eff_rel, m_eff_si=m_eff_si,
        fit_quality=r2, n_points=n_used, 
    )


def get_effective_mass(
    source:   "str | BandStructure",
    n_points: int = 5,
    carrier:  str = "electrons",
) -> _EffectiveMassResult:
    """
    Computes the effective mass from a band structure using parabolic fitting.

    This function extracts the band edge (CBM or VBM) from a VASP band structure
    and estimates the carrier effective mass by fitting the local dispersion to
    a parabolic model. It supports both spin-polarized and non-spin-polarized
    calculations, as well as SOC band structures (when detected during loading).

    For line-mode band structures, the effective mass is computed along each
    high-symmetry segment. For non-line-mode structures, an isotropic fit is
    performed.

    Parameters:
        source(str | BandStructure): Input band structure source. Can be either:
            - Path to a VASP calculation directory or vasprun.xml file, or
            - Preloaded BandStructure object.

        n_points(int, optional): Number of k-points on each side of the band
            edge used for local parabolic fitting. Default is 5.

        carrier(str, optional): Type of charge carrier to analyze. Must be:
            - "electrons" (CBM, default)
            - "holes" (VBM)

    Returns:
        EffectiveMassResult: Object containing:
            - m_eff_rel(float): Average effective mass in units of electron mass
            - m_eff_si(float): Effective mass in SI units (kg)
            - segments(list): List of segment-level effective mass fits
            - E_edge(float): Band edge energy in eV
            - k_edge(np.ndarray): k-point (Cartesian coordinates) of band edge
            - carrier(str): Carrier type used in computation

    Raises:
        ValueError:
            - If an invalid carrier type is provided.
            - If the material is metallic (zero or negative band gap).
            - If no valid k-path segments contain the band edge.
            - If all segment fits fail to produce a valid effective mass.

    Notes:
        The method:
            - Detects SOC band structures and restricts analysis to Spin.up
            - Identifies CBM/VBM from band structure metadata
            - Handles degenerate band edges (SOC splitting)
            - Performs local parabolic fits via `_fit_segment_mass`
            - Uses harmonic averaging of segment effective masses
            - Requires sufficient k-path sampling around band extrema

        For line-mode band structures:
            - Each high-symmetry segment is fit independently

        For non-line-mode band structures:
            - A single isotropic fit is performed
    """

    if carrier not in ("electrons", "holes"):
        raise ValueError(f"carrier must be 'electrons' or 'holes', got '{carrier}'.")

    if isinstance(source, (BandStructure, BandStructureSymmLine)):
        bs     = source
        is_soc = False   # can't determine without vasprun — assume not SOC
    else:
        bs = get_band_structure(source)
        
    if is_soc:
        print("  SOC band structure detected (LSORBIT=.TRUE.) — using Spin.up channel.")

    try:
        gap = bs.get_band_gap()["energy"]
    except Exception:
        gap = None

    if gap is not None and gap <= 0.0:
        if bs.is_metal():
            raise ValueError(
                "Band structure indicates a metal — parabolic effective mass "
                "is not applicable."
            )


    edge_info = bs.get_cbm() if carrier == "electrons" else bs.get_vbm()
    E_edge    = edge_info["energy"]
    spin      = _detect_spin_channel(bs, edge_info)

    edge_band        = edge_info["band_index"][spin][0]
    edge_kidx_global = edge_info["kpoint_index"][0]
    kpoints_cart_all = np.array([kp.cart_coords for kp in bs.kpoints]) * 1e10
    k_edge           = kpoints_cart_all[edge_kidx_global]

    # Collect all bands degenerate at the edge (handles SOC splitting)
    def _near_edge(band_idx: int) -> bool:
        e = np.array(bs.bands[spin][band_idx])
        return (np.min(e) <= E_edge + 0.01 if carrier == "electrons"
                else np.max(e) >= E_edge - 0.01)

    edge_bands = [i for i in range(bs.bands[spin].shape[0]) if _near_edge(i)]
    if not edge_bands:
        edge_bands = [edge_band]

    segment_masses = []

    if isinstance(bs, BandStructureSymmLine):
        for seg in bs.branches:
            i_lo     = seg["start_index"]
            i_hi     = seg["end_index"] + 1
            label    = seg["name"]
            pretty   = label.replace("GAMMA", "Γ").replace("|", "").replace("-", "→")
            seg_kpts = kpoints_cart_all[i_lo:i_hi]

            for band_idx in edge_bands:
                seg_energies = np.array(bs.bands[spin][band_idx][i_lo:i_hi])

                if carrier == "electrons":
                    if np.min(seg_energies) > E_edge + 0.05:
                        continue
                    local_edge = int(np.argmin(seg_energies))
                else:
                    if np.max(seg_energies) < E_edge - 0.05:
                        continue
                    local_edge = int(np.argmax(seg_energies))

                seg_label = (
                    f"{pretty} (b{band_idx})" if len(edge_bands) > 1 else pretty
                )

                sm = _fit_segment_mass(
                    kpoints_cart = seg_kpts,
                    energies     = seg_energies,
                    edge_kidx    = local_edge,
                    E_edge       = E_edge,
                    label        = seg_label,
                    n_points     = n_points,
                    carrier      = carrier,
                )
                segment_masses.append(sm)


    else:
        band_all   = np.array(bs.bands[spin][edge_band])
        local_edge = (int(np.argmin(band_all)) if carrier == "electrons"
                      else int(np.argmax(band_all)))
        sm = _fit_segment_mass(
            kpoints_cart = kpoints_cart_all,
            energies     = band_all,
            edge_kidx    = local_edge,
            E_edge       = E_edge,
            label        = "isotropic",
            n_points     = n_points,
            carrier      = carrier,
        )
        segment_masses.append(sm)

    if not segment_masses:
        raise ValueError(
            "No k-path segments contained the band edge. "
            "Check that the band structure covers the CBM/VBM k-point."
        )


    valid = [s.m_eff_rel for s in segment_masses if np.isfinite(s.m_eff_rel)]

    if not valid:
        raise ValueError("All segment fits failed. Cannot compute an average m*.")

    m_eff_rel_avg = np.cbrt(np.prod(valid))
    m_eff_si_avg  = m_eff_rel_avg * M_E

    return _EffectiveMassResult(
        m_eff_rel = m_eff_rel_avg,
        m_eff_si  = m_eff_si_avg,
        segments  = segment_masses,
        E_edge    = E_edge,
        k_edge    = k_edge,
        carrier   = carrier,
    )


def _format_em_table(em: _EffectiveMassResult, edge: float, is_dos_carrier: bool, fit: float) -> list:
    """
    Formats a human-readable summary table for effective mass results.

    This function generates a structured list of formatted strings summarizing
    effective mass calculations for either electrons or holes. It includes the
    band edge type, effective mass in both relative and SI units, and the fit
    quality of the underlying parabolic approximation.

    Parameters:
        em(EffectiveMassResult): Effective mass result object containing
            computed masses and metadata for a given carrier type.

        edge(float): Energy (in eV) of the relevant band edge (CBM or VBM).

        is_dos_carrier(bool): Flag indicating whether the effective mass was
            derived from a DOS-based method, used to annotate the output.

        fit(float): R² value representing the quality of the effective mass
            fit.

    Returns:
        list: List of formatted strings representing a readable summary table.

    Notes:
        The output includes:
            - Carrier type (electron or hole)
            - Band edge reference (CBM or VBM)
            - Harmonic mean effective mass in units of mₑ or mₕ
            - Effective mass in SI units (kg)
            - Fit quality (R²)
            - Optional annotation if DOS-derived mass was used

        The returned list is intended for direct printing or logging.
    """
    
    edge_label  = "CBM" if em.carrier == "electrons" else "VBM"
    sub = "ₑ" if em.carrier == "electrons" else "ₕ"

    carrier_str = em.carrier.capitalize()
    dos_marker  = "  ← dos_mass" if is_dos_carrier else ""

    lines = [
        f"  {carrier_str} (fitted at {edge_label}: {edge:.3f} eV)",
        f"  {'Harmonic mean m*':<20}: {em.m_eff_rel:.3f} m{sub}"
        f"  ({em.m_eff_si:.3e} kg){dos_marker}",
        f"  Fit quality {em.carrier}   : {fit:.3f} R²"
        "",
    ]

    return lines


def _format_dos_summary(result: "DOSResult") -> str:
    """
    Formats a complete human-readable summary of DOS-based effective mass results.

    This function builds a structured text report summarizing density-of-states
    (DOS) derived effective mass calculations, including band edge positions,
    cell volume, and both electron and hole effective masses when available.

    Parameters:
        result(DOSResult): Object containing DOS-derived physical properties and
            effective mass results. Expected fields include:
                - carrier(str): Primary carrier type ("electrons" or "holes")
                - cbm(float): Conduction band minimum energy (eV)
                - vbm(float): Valence band maximum energy (eV)
                - cell_volume_m3(float): Unit cell volume in m³
                - em_electrons(EffectiveMassResult | None): Electron effective mass
                - em_holes(EffectiveMassResult | None): Hole effective mass
                - fit_quality_e(float): Fit quality for electrons
                - fit_quality_h(float): Fit quality for holes

    Returns:
        str: Multi-line formatted summary string suitable for printing or logging.

    Notes:
        The output includes:
            - Primary carrier type
            - Band edge energies (CBM/VBM)
            - Unit cell volume
            - Electron and hole effective masses (if available)
            - Fit quality indicators for band-structure-derived masses
            - Clear separation between electron and hole sections

        If effective mass data is not available for a carrier type, a fallback
        message is displayed indicating that values were supplied directly
        without band structure fitting.
    """

    edge_label = "CBM" if result.carrier == "electrons" else "VBM"
    edge_value = result.cbm if result.carrier == "electrons" else result.vbm

    lines = [
        "",
        "=" * 60,
        "  DOS Result Summary",
        "=" * 60,
        f"  Primary carrier     : {result.carrier.capitalize()}",
        f"  Band edge ({edge_label})     : {edge_value:.3f} eV",
        f"  Cell volume         : {result.cell_volume_m3:.3e} m³",
        "",
        "  ── Effective Masses " + "─" * 38,
    ]

    # Electrons block
    if result.em_electrons is not None:
        lines += _format_em_table(
            result.em_electrons,
            result.cbm,
            is_dos_carrier=(result.carrier == "electrons"),
            fit=result.fit_quality_e
        )
    else:
        lines.append("  Electrons : m* supplied directly — no band structure fit.")

    lines.append("")

    # Holes block
    if result.em_holes is not None:
        lines += _format_em_table(
            result.em_holes,
            result.vbm,
            is_dos_carrier=(result.carrier == "holes"),
            fit=result.fit_quality_h
        )
    else:
        lines.append("  Holes : mₕ* supplied directly — no band structure fit.")

    lines += ["=" * 60, ""]
    return "\n".join(lines)



def print_dos_summary(result: "DOSResult") -> None:
    """
    Prints a formatted summary of DOS-based effective mass results.

    This function outputs a human-readable report of density-of-states (DOS)
    analysis results by printing a structured summary generated from
    `_format_dos_summary`.

    Parameters:
        result(DOSResult): Object containing DOS-derived properties and
            effective mass results, including band edges, carrier type,
            and optional electron/hole effective mass data.

    Returns:
        None

    Notes:
        This is a convenience wrapper around `_format_dos_summary` and is
        intended for direct console output.
    """

    print(_format_dos_summary(result))

def compute_dos(
    dos_vasprun:  str,
    bs_vasprun:   Optional[str] = None,
    bs_directory: Optional[str] = None,
    m_eff:        Optional[float] = None,
    splits:       int = 1,
    n_fit_points: int = 5,
    carrier:      str = "electrons",
) -> DOSResult:
    
    """
    Computes DOS-based electronic properties and optional band-structure
    effective masses.

    This function analyzes a VASP DOS calculation and optionally combines it
    with band structure information to compute electron and hole effective
    masses. It supports three mutually exclusive input modes for effective
    mass evaluation:

        - Band structure vasprun file
        - Band structure directory (split or single)
        - Pre-supplied effective mass value

    The function also computes band edges, unit cell volume, and collects
    fit quality metrics from band structure-derived effective masses.

    Parameters:
        dos_vasprun(str): Path to a vasprun.xml file containing DOS data.

        bs_vasprun(Optional[str]): Path to a band structure vasprun.xml file.
            Mutually exclusive with `bs_directory` and `m_eff`.

        bs_directory(Optional[str]): Path to a directory containing band
            structure outputs (including possible split runs).
            Mutually exclusive with `bs_vasprun` and `m_eff`.

        m_eff(Optional[float]): Precomputed effective mass value supplied
            directly. Mutually exclusive with band structure inputs.

        sigma(float, optional): Gaussian smearing width used in DOS analysis.
            Default is 0.05.

        n_fit_points(int, optional): Number of k-points used on each side of
            the band edge for effective mass fitting. Default is 5.

        carrier(str, optional): Type of carrier to analyze:
            - "electrons" (default)
            - "holes"

    Returns:
        DOSResult: Object containing:
            - fit_quality_e(float | None): Electron fit quality (R²)
            - fit_quality_h(float | None): Hole fit quality (R²)
            - cell_volume_m3(float): Unit cell volume in m³
            - carrier(str): Selected carrier type
            - final_result(float | None): Final effective mass result
            - em_electrons(EffectiveMassResult | None): Electron effective mass
            - em_holes(EffectiveMassResult | None): Hole effective mass
            - cbm(float): Conduction band minimum (shifted to VBM reference)
            - vbm(float): Valence band maximum (set to zero reference)

    Notes:
        The returned band edges are shifted such that VBM = 0 eV.
    """
  
    if carrier not in ("electrons", "holes"):
        raise ValueError(f"carrier must be 'electrons' or 'holes', got '{carrier}'.")

    n_sources = sum(x is not None for x in [bs_vasprun, bs_directory, m_eff])
    if n_sources == 0:
        raise ValueError("Provide exactly one of: bs_vasprun, bs_directory, or m_eff.")
    if n_sources > 1:
        raise ValueError(
            "Provide exactly one of bs_vasprun, bs_directory, or m_eff — not multiple."
        )

    vr       = Vasprun(dos_vasprun, parse_dos=True, parse_eigen=False)
    cdos     = vr.complete_dos
    vol_m3   = vr.final_structure.volume * 1e-30

    em_electrons = None
    em_holes     = None
    fit_quality_electrons  = None
    fit_quality_holes = None
    cbm = None
    vbm = None

    if bs_vasprun is not None or bs_directory is not None:
        source = bs_directory if bs_directory is not None else bs_vasprun

        # Load the band structure once, pass the object to avoid re-parsing
        bs, _ = get_band_structure(source, splits= splits)

        print("  Computing electron effective mass (CBM)...")
        try:
            em_electrons = get_effective_mass(bs, n_points=n_fit_points, carrier="electrons")
            valid_segs  = [s for s in em_electrons.segments if np.isfinite(s.fit_quality)]
            fit_quality_electrons = np.mean([s.fit_quality for s in valid_segs]) if valid_segs else None
            cbm = em_electrons.E_edge

        except Exception as e:
            warnings.warn(f"Electron m* fit failed: {e}", UserWarning, stacklevel=2)

        print("  Computing hole effective mass (VBM)...")
        try:
            em_holes = get_effective_mass(bs, n_points=n_fit_points, carrier="holes")
            valid_segs  = [s for s in em_holes.segments if np.isfinite(s.fit_quality)]
            fit_quality_holes = np.mean([s.fit_quality for s in valid_segs]) if valid_segs else None
            vbm = em_holes.E_edge

        except Exception as e:
            warnings.warn(f"Hole m* fit failed: {e}", UserWarning, stacklevel=2)

    else:
        # m_eff supplied directly — single carrier only
        cbm, vbm  = cdos.get_cbm_vbm()

    vbm_zeroed = vbm - vbm
    cbm_zeroed = cbm - vbm


    return DOSResult(
        fit_quality_e     = fit_quality_electrons,
        fit_quality_h     = fit_quality_holes,
        cell_volume_m3    = vol_m3,
        carrier           = carrier,
        final_result      = em_electrons.m_eff_rel, 
        em_electrons      = em_electrons,
        em_holes          = em_holes,
        cbm               = cbm_zeroed,
        vbm               = vbm_zeroed
        
    )
