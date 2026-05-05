import matplotlib
from matplotlib import pyplot as plt
from sumo.cli.dosplot import dosplot
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

from solphin import dos_plotting
from pymatgen.io.vasp import Vasprun
import os, glob, numpy as np

def plot_dos(filename, xmin=-3, xmax=3, gaussian=0.05, save=False):
    fig, ax = plt.subplots(figsize=(5,3), dpi=150)
    dosplot(filename=filename, xmin=xmin, xmax=xmax, gaussian=gaussian, plt=plt)

    if save:
        plt.savefig("dos.pdf")

    plt.show()
    return

"""
effective_mass.py
=================
Compute the free-electron density of states and effective mass at the CBM
and VBM for any material, with support for:

  - GGA/LDA band structures (single vasprun.xml)
  - Hybrid (HSE/PBE0) band structures split across split-01/, split-02/, ...
  - SOC (LSORBIT = .TRUE.) non-collinear calculations
  - Electron effective masses (fitted at CBM, positive curvature)
  - Hole effective masses (fitted at VBM, negative curvature)
  - Both carriers always computed and reported
  - Anisotropic effective masses per k-path segment
  - Harmonic-mean scalar average (physically correct for transport)
  - DOS/optics vaspruns with an externally supplied m*

The calculation type (GGA, hybrid splits, SOC) is detected automatically.
Both electron and hole masses are always computed when a band structure
is provided — the `carrier` argument controls which m* is used in the
free-electron DOS equation.

Usage
-----
# Compute DOS using electron m* (default), report both carrier masses:
result = compute_dos(
    dos_vasprun  = "path/to/dos/vasprun.xml",
    bs_directory = "path/to/band/",
    carrier      = "electrons",
)
print_dos_summary(result)

# Compute DOS using hole m*:
result = compute_dos(
    dos_vasprun  = "path/to/dos/vasprun.xml",
    bs_directory = "path/to/band/",
    carrier      = "holes",
)
print_dos_summary(result)
"""

import warnings
import glob
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34    # J·s
M_E  = 9.1093837015e-31   # kg
EV   = 1.602176634e-19    # J per eV

# Sanity thresholds
_M_EFF_MAX = 10.0   # m_e  — warn above this
_R2_MIN    = 0.90   # warn below this


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SegmentMass:
    """Effective mass fitted along a single k-path segment."""
    label:       str    # e.g. "Γ→X"
    m_eff_rel:   float  # m* / m_e
    m_eff_si:    float  # kg
    fit_quality: float  # R²
    n_points:    int    # k-points used in fit
    warnings:    list   = field(default_factory=list)


@dataclass
class EffectiveMassResult:
    """Full anisotropic + averaged effective mass result for one carrier type."""
    m_eff_rel:   float       # harmonic mean m* / m_e
    m_eff_si:    float       # harmonic mean m* in kg
    segments:    list        # list[SegmentMass]
    E_edge:      float       # CBM (electrons) or VBM (holes) energy in eV
    k_edge:      np.ndarray  # edge k-point, Cartesian 1/m
    carrier:     str         # "electrons" or "holes"
    warnings:    list        = field(default_factory=list)

    @property
    def E_c(self):
        """Backwards-compatible alias for E_edge."""
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
        for seg in self.segments:
            warn_flag = " ⚠" if seg.warnings else ""
            if np.isfinite(seg.m_eff_rel):
                lines.append(
                    f"  {seg.label:<18} {seg.m_eff_rel:>10.4f} "
                    f"{seg.fit_quality:>8.4f} {seg.n_points:>7}{warn_flag}"
                )
            else:
                lines.append(
                    f"  {seg.label:<18} {'N/A':>10} {'N/A':>8} "
                    f"{seg.n_points:>7}{warn_flag}"
                )
        lines += [
            "  " + "-" * 47,
            f"  {'Harmonic mean':<18} {self.m_eff_rel:>10.4f}",
            "=" * 60,
        ]
        if self.warnings:
            lines.append("\n  Warnings:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)


@dataclass
class DOSResult:
    """
    Combined VASP + free-electron DOS result.

    Always contains both electron and hole effective mass results when a
    band structure was provided. The `carrier` field indicates which m* was
    used in the free-electron DOS equation.
    """
    energies:          np.ndarray
    vasp_dos:          np.ndarray        # states / eV / cell
    free_electron_dos: np.ndarray        # states / eV / m³
    E_c:               float             # band edge used in DOS equation (eV)
    m_eff_rel:         float             # m* / m_e used in DOS equation
    m_eff_si:          float             # kg
    fit_quality:       Optional[float]   # mean R² for the DOS carrier
    cell_volume_m3:    float             # m³
    carrier:           str               # "electrons" or "holes" — used in DOS
    em_electrons:      Optional[EffectiveMassResult] = None   # always computed
    em_holes:          Optional[EffectiveMassResult] = None   # always computed
    warnings:          list              = field(default_factory=list)

    @property
    def em_result(self) -> Optional[EffectiveMassResult]:
        """Return the EffectiveMassResult for the active carrier (DOS carrier)."""
        if self.carrier == "electrons":
            return self.em_electrons
        return self.em_holes

    def __str__(self):
        return _format_dos_summary(self)


# ---------------------------------------------------------------------------
# SOC helpers
# ---------------------------------------------------------------------------

def _is_soc(bs: BandStructure) -> bool:
    """Return True if the band structure is from a SOC (non-collinear) calc."""
    return len(bs.bands) == 1 and not bs.is_spin_polarized


def _detect_spin_channel(bs: BandStructure, edge_info: dict) -> Spin:
    """
    Return the spin channel containing the band edge.

    For SOC pymatgen uses Spin.up regardless of physical spin.
    For collinear spin-polarised calculations picks the channel with the edge.
    """
    if len(bs.bands) == 1:
        return Spin.up
    if Spin.up in edge_info["band_index"] and edge_info["band_index"][Spin.up]:
        return Spin.up
    return Spin.down


# ---------------------------------------------------------------------------
# Fermi level helper
# ---------------------------------------------------------------------------

def _get_efermi(vr: Vasprun, path: str) -> float:
    """
    Return the Fermi level from a parsed Vasprun, falling back to a
    second parse with parse_dos=True if vr.efermi is None.

    Parameters
    ----------
    vr : Vasprun
        Already-parsed Vasprun object (parse_dos=False).
    path : str
        Path to the vasprun.xml, used for the fallback parse.

    Returns
    -------
    float
        Fermi level in eV.
    """
    efermi = vr.efermi
    if efermi is not None:
        return efermi

    try:
        vr_dos = Vasprun(path, parse_dos=True, parse_eigen=False)
        efermi = vr_dos.efermi
    except Exception:
        pass

    if efermi is not None:
        return efermi

    raise ValueError(
        f"Could not determine Fermi level from '{path}'. "
        "Check that the vasprun.xml is complete and not truncated."
    )


# ---------------------------------------------------------------------------
# Split-folder helpers
# ---------------------------------------------------------------------------

def _find_split_vaspruns(directory: str) -> Tuple[list, bool]:
    """
    Search `directory` for band structure vasprun(s).

    Looks for split_*/vasprun.xml first (hybrid). Falls back to a single
    vasprun.xml in the directory (GGA/SOC), but only if a KPOINTS file
    is present alongside it — prevents accidentally using a DOS vasprun as
    a band structure when both point to the same folder.

    Returns
    -------
    paths : list[str]
    is_split : bool
    """
    pattern = os.path.join(directory, "split-*", "vasprun.xml")
    splits  = sorted(glob.glob(pattern))

    if splits:
        return splits, True

    single       = os.path.join(directory, "vasprun.xml")
    kpoints_file = os.path.join(directory, "KPOINTS")

    if os.path.isfile(single):
        if not os.path.isfile(kpoints_file):
            raise FileNotFoundError(
                f"Found vasprun.xml in '{directory}' but no KPOINTS file alongside it. "
                f"If this is a DOS/optics vasprun, use m_eff instead of bs_directory. "
                f"If this is a GGA band structure, ensure the KPOINTS file is present."
            )
        return [single], False

    raise FileNotFoundError(
        f"No split-*/vasprun.xml or vasprun.xml found in '{directory}'. "
        f"Check that the path is correct."
    )


def _parse_single_bs(path: str) -> BandStructureSymmLine:
    """Parse a single band structure vasprun.xml into a BandStructureSymmLine."""
    vr           = Vasprun(path, parse_dos=False, parse_eigen=True)
    kpoints_file = os.path.join(os.path.dirname(os.path.abspath(path)), "KPOINTS")
    efermi       = _get_efermi(vr, path)
    return vr.get_band_structure(
        kpoints_filename = kpoints_file if os.path.isfile(kpoints_file) else None,
        efermi           = efermi,
        line_mode        = True,
    )


def _merge_split_band_structures(vaspruns: list) -> BandStructureSymmLine:
    """
    Parse and concatenate split band structure vaspruns into a single
    BandStructureSymmLine.

    Band counts are harmonised to the minimum across all splits to handle
    SOC calculations where VASP occasionally writes an extra band in some
    segments.
    """
    parsed = []
    for path in vaspruns:
        vr           = Vasprun(path, parse_dos=False, parse_eigen=True)
        kpoints_file = os.path.join(os.path.dirname(path), "KPOINTS")

        if not os.path.isfile(kpoints_file):
            raise FileNotFoundError(
                f"KPOINTS file not found alongside {path}. "
                f"Expected at {kpoints_file}."
            )

        efermi = _get_efermi(vr, path)
        bs     = vr.get_band_structure(
            kpoints_filename = kpoints_file,
            efermi           = efermi,
            line_mode        = True,
        )
        parsed.append(bs)

    if len(parsed) == 1:
        return parsed[0]

    ref = parsed[0]

    # Harmonise band counts across splits (SOC often has ±1 band mismatch)
    n_bands_min = {}
    for spin in ref.bands:
        counts = [bs.bands[spin].shape[0] for bs in parsed]
        n_bands_min[spin] = min(counts)
        if len(set(counts)) > 1:
            warnings.warn(
                f"Band count mismatch across splits {counts} "
                f"(spin={spin}) — truncating all to {n_bands_min[spin]} bands. "
                "This is expected for SOC calculations.",
                UserWarning,
                stacklevel=2,
            )

    all_kpoints = []
    all_labels  = {}
    kpt_offset  = 0
    all_bands   = {spin: [] for spin in ref.bands}

    for bs in parsed:
        for spin in all_bands:
            all_bands[spin].append(bs.bands[spin][:n_bands_min[spin], :])

        all_kpoints.extend(bs.kpoints)

        for label, kpt in bs.labels_dict.items():
            for i, kp in enumerate(bs.kpoints):
                if np.allclose(kp.frac_coords, kpt.frac_coords, atol=1e-4):
                    all_labels[label] = i + kpt_offset
                    break

        kpt_offset += len(bs.kpoints)

    for spin in all_bands:
        all_bands[spin] = np.hstack(all_bands[spin])

    # BandStructureSymmLine expects fractional coordinate arrays, not Kpoint objects
    kpoints_frac = [kp.frac_coords for kp in all_kpoints]
    labels_dict  = {
        label: all_kpoints[idx].frac_coords
        for label, idx in all_labels.items()
    }

    return BandStructureSymmLine(
        kpoints     = kpoints_frac,
        eigenvals   = all_bands,
        lattice     = ref.lattice_rec,
        efermi      = ref.efermi,
        labels_dict = labels_dict,
        structure   = ref.structure,
    )


def _load_band_structure(source: str) -> BandStructureSymmLine:
    """
    Load a band structure from a directory or file path, auto-detecting
    whether it is a split hybrid, single GGA, or SOC calculation.

    Parameters
    ----------
    source : str
        Directory containing split-*/vasprun.xml or a single vasprun.xml,
        or a direct path to a vasprun.xml file.

    Returns
    -------
    BandStructureSymmLine
    """
    if os.path.isdir(source):
        paths, is_split = _find_split_vaspruns(source)
        if is_split:
            print(f"  Found {len(paths)} split folder(s) — merging (hybrid calc).")
            return _merge_split_band_structures(paths)
        else:
            print(f"  No splits found — single vasprun (GGA/SOC calc).")
            return _parse_single_bs(paths[0])
    elif os.path.isfile(source):
        return _parse_single_bs(source)
    else:
        raise FileNotFoundError(f"Source not found: '{source}'")


# ---------------------------------------------------------------------------
# Effective mass fitting
# ---------------------------------------------------------------------------

def _fit_segment_mass(
    kpoints_cart: np.ndarray,
    energies:     np.ndarray,
    edge_kidx:    int,
    E_edge:       float,
    label:        str,
    n_points:     int,
    carrier:      str = "electrons",
) -> SegmentMass:
    """
    Fit a parabola to one k-path segment near the band edge.

    For electrons: E(k) = E_c + (ħ²/2m*)|k - k_c|²  (positive curvature)
    For holes:     E(k) = E_v - (ħ²/2m*)|k - k_v|²  (negative curvature)

    In both cases dE is arranged to be positive so the slope is positive,
    giving m* = ħ²/2·slope > 0.

    Parameters
    ----------
    kpoints_cart : np.ndarray, shape (N, 3)
        Cartesian k-points in 1/m.
    energies : np.ndarray, shape (N,)
        Band energies in eV.
    edge_kidx : int
        Index of the band edge k-point within this segment.
    E_edge : float
        CBM (electrons) or VBM (holes) energy in eV.
    label : str
        Segment label, e.g. "Γ→X".
    n_points : int
        k-points either side of the edge to include in fit.
    carrier : str
        "electrons" or "holes".

    Returns
    -------
    SegmentMass
    """
    warns  = []
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
        msg = f"Segment {label}: fewer than 2 fit points — skipping."
        warns.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=4)
        return SegmentMass(
            label=label, m_eff_rel=np.nan, m_eff_si=np.nan,
            fit_quality=np.nan, n_points=n_used, warnings=warns,
        )

    slope, intercept = np.polyfit(dk_sq, dE, 1)

    if slope <= 0:
        direction = "upward" if carrier == "electrons" else "downward"
        msg = (
            f"Segment {label}: slope = {slope:.3e} — band does not curve "
            f"{direction} at this point. Skipping."
        )
        warns.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=4)
        return SegmentMass(
            label=label, m_eff_rel=np.nan, m_eff_si=np.nan,
            fit_quality=np.nan, n_points=n_used, warnings=warns,
        )

    m_eff_si  = HBAR**2 / (2.0 * slope)
    m_eff_rel = m_eff_si / M_E

    dE_pred = slope * dk_sq + intercept
    ss_res  = np.sum((dE - dE_pred) ** 2)
    ss_tot  = np.sum((dE - np.mean(dE)) ** 2)
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    if r2 < _R2_MIN:
        msg = (
            f"Segment {label}: poor parabolic fit (R² = {r2:.4f}). "
            "Band may be non-parabolic in this direction."
        )
        warns.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=4)

    if m_eff_rel > _M_EFF_MAX:
        msg = f"Segment {label}: large m* = {m_eff_rel:.2f} mₑ — treat with caution."
        warns.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=4)

    return SegmentMass(
        label=label, m_eff_rel=m_eff_rel, m_eff_si=m_eff_si,
        fit_quality=r2, n_points=n_used, warnings=warns,
    )


def get_effective_mass(
    source:   "str | BandStructure",
    n_points: int = 5,
    carrier:  str = "electrons",
) -> EffectiveMassResult:
    """
    Compute anisotropic and averaged effective masses at the CBM or VBM.

    Supports all VASP band structure calculation types — the type is detected
    automatically from the directory contents.

    Parameters
    ----------
    source : str or BandStructure
        - Directory path (GGA/hybrid/SOC — auto-detected)
        - Path to a single vasprun.xml
        - Pre-parsed BandStructure / BandStructureSymmLine object
    n_points : int
        k-points either side of the band edge per segment fit.
    carrier : str
        "electrons" — fit at CBM (n-type).
        "holes"     — fit at VBM (p-type).

    Returns
    -------
    EffectiveMassResult
    """
    if carrier not in ("electrons", "holes"):
        raise ValueError(f"carrier must be 'electrons' or 'holes', got '{carrier}'.")

    warns = []

    # ------------------------------------------------------------------
    # 1. Obtain the band structure
    # ------------------------------------------------------------------
    if isinstance(source, (BandStructure, BandStructureSymmLine)):
        bs = source
    else:
        bs = _load_band_structure(source)

    # ------------------------------------------------------------------
    # 2. SOC detection and metal check
    # ------------------------------------------------------------------
    if _is_soc(bs):
        print("  SOC band structure detected — using Spin.up channel (non-collinear).")

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
        else:
            msg = (
                "Band gap is zero or could not be determined but system is "
                "not flagged as a metal. Proceeding with caution."
            )
            warns.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

    # ------------------------------------------------------------------
    # 3. Locate band edge
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. Anisotropic fit — one mass per k-path segment per degenerate band
    # ------------------------------------------------------------------
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
                warns.extend(sm.warnings)

    else:
        msg = (
            "Band structure is not line-mode — fitting a single isotropic "
            "mass. Supply a line-mode band structure for anisotropic results."
        )
        warns.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

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
        warns.extend(sm.warnings)

    if not segment_masses:
        raise ValueError(
            "No k-path segments contained the band edge. "
            "Check that the band structure covers the CBM/VBM k-point."
        )

    # ------------------------------------------------------------------
    # 5. Harmonic mean scalar average
    # ------------------------------------------------------------------
    valid = [s for s in segment_masses if np.isfinite(s.m_eff_rel) and s.m_eff_rel > 0]

    if not valid:
        raise ValueError("All segment fits failed. Cannot compute an average m*.")

    m_eff_rel_avg = 1.0 / np.mean([1.0 / s.m_eff_rel for s in valid])
    m_eff_si_avg  = m_eff_rel_avg * M_E

    return EffectiveMassResult(
        m_eff_rel = m_eff_rel_avg,
        m_eff_si  = m_eff_si_avg,
        segments  = segment_masses,
        E_edge    = E_edge,
        k_edge    = k_edge,
        carrier   = carrier,
        warnings  = warns,
    )


# ---------------------------------------------------------------------------
# Free-electron DOS
# ---------------------------------------------------------------------------

def free_electron_dos(
    energies: np.ndarray,
    E_c:      float,
    m_eff:    float = 1.0,
) -> np.ndarray:
    """
    Compute the free-electron (parabolic band) density of states:

        DOS(E) = (1/2π²)(2 m* mₑ / ħ²)^(3/2) (E - E_c)^(1/2)

    This is always evaluated from E_c (CBM) upward regardless of carrier type,
    since the formula describes the conduction band DOS. For holes, the m*
    fed into this equation is the hole effective mass, giving the equivalent
    valence band DOS by symmetry.

    Parameters
    ----------
    energies : np.ndarray
        Energy values in eV.
    E_c : float
        Band edge (CBM) in eV.
    m_eff : float
        Effective mass relative to mₑ (electron or hole).

    Returns
    -------
    np.ndarray
        DOS in states / eV / m³, zero below E_c.
    """
    prefactor = (1.0 / (2.0 * np.pi**2)) * (2.0 * m_eff * M_E / HBAR**2) ** 1.5
    dE        = np.maximum(energies - E_c, 0.0) * EV
    return prefactor * np.sqrt(dE) * EV


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _format_em_table(em: EffectiveMassResult, is_dos_carrier: bool) -> list:
    """Return formatted lines for one EffectiveMassResult block."""
    edge_label  = "CBM" if em.carrier == "electrons" else "VBM"
    carrier_str = em.carrier.capitalize()
    dos_marker  = "  ← used in DOS equation" if is_dos_carrier else ""

    lines = [
        f"  {carrier_str} (fitted at {edge_label} = {em.E_edge:.4f} eV)",
        f"  {'Harmonic mean m*':<20}: {em.m_eff_rel:.4f} mₑ"
        f"  ({em.m_eff_si:.4e} kg){dos_marker}",
    ]

    valid_segs = [s for s in em.segments if np.isfinite(s.fit_quality)]
    if valid_segs:
        mean_r2 = np.mean([s.fit_quality for s in valid_segs])
        lines.append(f"  {'Mean fit quality R²':<20}: {mean_r2:.4f}")

    lines += [
        "",
        f"  {'Segment':<20} {'m* (mₑ)':>10} {'R²':>8} {'N pts':>7}",
        "  " + "-" * 47,
    ]

    for seg in em.segments:
        warn_flag = "  ⚠" if seg.warnings else ""
        if np.isfinite(seg.m_eff_rel):
            lines.append(
                f"  {seg.label:<20} {seg.m_eff_rel:>10.4f} "
                f"{seg.fit_quality:>8.4f} {seg.n_points:>7}{warn_flag}"
            )
        else:
            lines.append(
                f"  {seg.label:<20} {'N/A':>10} {'N/A':>8} "
                f"{seg.n_points:>7}{warn_flag}"
            )

    lines.append(f"  {'Harmonic mean':<20} {em.m_eff_rel:>10.4f}")

    if em.warnings:
        for w in em.warnings:
            lines.append(f"  ⚠ {w}")

    return lines


def _format_dos_summary(result: "DOSResult") -> str:
    """Return a formatted string summary of a DOSResult."""
    edge_label = "CBM" if result.carrier == "electrons" else "VBM"

    above_edge  = result.energies[result.energies >= result.E_c]
    active_range = (
        f"{result.E_c:.4f} → {above_edge[-1]:.4f} eV"
        if len(above_edge) else "N/A"
    )

    fe_max   = np.max(result.free_electron_dos)
    vasp_max = np.max(result.vasp_dos)

    lines = [
        "",
        "=" * 60,
        "  DOS Result Summary",
        "=" * 60,
        f"  Primary carrier     : {result.carrier.capitalize()}",
        f"  Band edge ({edge_label})     : {result.E_c:.4f} eV",
        f"  Cell volume         : {result.cell_volume_m3:.4e} m³",
        f"  Energy range        : {result.energies[0]:.4f} → "
                                 f"{result.energies[-1]:.4f} eV",
        f"  Active DOS range    : {active_range}",
        f"  DOS grid points     : {len(result.energies)}",
        "",
        "  ── Effective Masses " + "─" * 39,
    ]

    # Electrons block
    if result.em_electrons is not None:
        lines += _format_em_table(
            result.em_electrons,
            is_dos_carrier=(result.carrier == "electrons"),
        )
    else:
        lines.append("  Electrons : m* supplied directly — no band structure fit.")

    lines.append("")

    # Holes block
    if result.em_holes is not None:
        lines += _format_em_table(
            result.em_holes,
            is_dos_carrier=(result.carrier == "holes"),
        )
    else:
        lines.append("  Holes : m* supplied directly — no band structure fit.")

    lines += [
        "",
        "  ── Free-Electron DOS " + "─" * 37,
        f"  Equation            : DOS(E) = (1/2π²)(2m*mₑ/ħ²)^(3/2)(E-E_c)^(1/2)",
        f"  m* used             : {result.m_eff_rel:.4f} mₑ  "
                                 f"({result.carrier} harmonic mean)",
        f"  E_c used            : {result.E_c:.4f} eV  ({edge_label})",
        f"  Units               : states / eV / m³",
        f"  Peak value          : {fe_max:.4e} states/eV/m³",
        f"  VASP DOS peak       : {vasp_max:.4e} states/eV/cell",
        f"  Normalisation note  : divide free-electron DOS by cell volume",
        f"                        ({result.cell_volume_m3:.3e} m³) to compare",
        f"                        directly with VASP DOS in states/eV/cell",
    ]

    if result.warnings:
        lines += ["", "  ── Warnings " + "─" * 46]
        for w in result.warnings:
            lines.append(f"  ⚠ {w}")

    lines += ["=" * 60, ""]
    return "\n".join(lines)


def print_dos_summary(result: "DOSResult") -> None:
    """
    Print a formatted summary of a DOSResult.

    Shows both electron and hole effective masses, clearly marks which m*
    was used in the free-electron DOS equation, and gives normalisation
    guidance for overlaying the free-electron and VASP DOS.

    Parameters
    ----------
    result : DOSResult
        Output of compute_dos().
    """
    print(_format_dos_summary(result))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_dos(
    dos_vasprun:  str,
    bs_vasprun:   Optional[str] = None,
    bs_directory: Optional[str] = None,
    m_eff:        Optional[float] = None,
    sigma:        float = 0.05,
    n_fit_points: int = 5,
    carrier:      str = "electrons",
) -> DOSResult:
    """
    Compute the VASP total DOS and free-electron DOS for any material.

    Both electron and hole effective masses are always computed when a band
    structure is provided. The `carrier` argument controls which m* is
    substituted into the free-electron DOS equation.

    Exactly one of `bs_vasprun`, `bs_directory`, or `m_eff` must be supplied.
    The calculation type (GGA, hybrid, SOC) is detected automatically.

      bs_directory  Path to the band structure directory:
                    - GGA/SOC: directory containing vasprun.xml + KPOINTS
                    - Hybrid:  directory containing split-01/, split-02/, ...

      bs_vasprun    Shorthand: path directly to a single band structure vasprun.xml.

      m_eff         Effective mass in units of mₑ, supplied directly.
                    Only one carrier mass can be supplied this way — use
                    a band structure for both carriers simultaneously.

    Parameters
    ----------
    dos_vasprun : str
        Path to the DOS/optics/band vasprun.xml (energies + VASP DOS).
    bs_vasprun : str, optional
        Path to a single band structure vasprun.xml.
    bs_directory : str, optional
        Path to the band structure directory.
    m_eff : float, optional
        Effective mass in units of mₑ (single carrier only).
    sigma : float
        Gaussian smearing in eV applied to the VASP DOS (0 = none).
    n_fit_points : int
        k-points either side of band edge used in each segment fit.
    carrier : str
        "electrons" (default) — use electron m* in DOS equation, n-type.
        "holes"               — use hole m* in DOS equation, p-type.
        Both masses are always computed from the band structure regardless.

    Returns
    -------
    DOSResult
        Contains energies, VASP DOS, free-electron DOS, and both electron
        and hole EffectiveMassResult objects.
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

    warns = []

    # ------------------------------------------------------------------
    # Parse DOS vasprun
    # ------------------------------------------------------------------
    vr       = Vasprun(dos_vasprun, parse_dos=True, parse_eigen=False)
    cdos     = vr.complete_dos
    energies = cdos.energies
    vasp_dos = cdos.get_densities()
    vol_m3   = vr.final_structure.volume * 1e-30

    if sigma > 0:
        de       = energies[1] - energies[0]
        vasp_dos = gaussian_filter1d(vasp_dos, sigma=sigma / de)

    # ------------------------------------------------------------------
    # Effective masses — always compute both carriers from band structure
    # ------------------------------------------------------------------
    em_electrons = None
    em_holes     = None
    fit_quality  = None

    if bs_vasprun is not None or bs_directory is not None:
        source = bs_directory if bs_directory is not None else bs_vasprun

        # Load the band structure once, pass the object to avoid re-parsing
        bs = _load_band_structure(source)

        print("  Computing electron effective mass (CBM)...")
        try:
            em_electrons = get_effective_mass(bs, n_points=n_fit_points, carrier="electrons")
            warns.extend(em_electrons.warnings)
        except Exception as e:
            warnings.warn(f"Electron m* fit failed: {e}", UserWarning, stacklevel=2)

        print("  Computing hole effective mass (VBM)...")
        try:
            em_holes = get_effective_mass(bs, n_points=n_fit_points, carrier="holes")
            warns.extend(em_holes.warnings)
        except Exception as e:
            warnings.warn(f"Hole m* fit failed: {e}", UserWarning, stacklevel=2)

        # Select the active carrier result for the DOS equation
        em_active = em_electrons if carrier == "electrons" else em_holes
        if em_active is None:
            raise ValueError(
                f"Effective mass fit for '{carrier}' failed — "
                "cannot compute free-electron DOS. See warnings above."
            )

        m_eff_rel   = em_active.m_eff_rel
        m_eff_si    = em_active.m_eff_si
        E_c         = em_active.E_edge
        valid_segs  = [s for s in em_active.segments if np.isfinite(s.fit_quality)]
        fit_quality = np.mean([s.fit_quality for s in valid_segs]) if valid_segs else None

    else:
        # m_eff supplied directly — single carrier only
        m_eff_rel = m_eff
        m_eff_si  = m_eff * M_E
        cbm, vbm  = cdos.get_cbm_vbm()
        E_c       = cbm if carrier == "electrons" else vbm
        if cdos.get_gap() == 0.0:
            msg = "System appears metallic — free-electron DOS formula may not be meaningful."
            warns.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

    # ------------------------------------------------------------------
    # Free-electron DOS — always evaluated from CBM upward
    # ------------------------------------------------------------------
    cbm_energy = E_c if carrier == "electrons" else cdos.get_cbm_vbm()[0]
    fe_dos     = free_electron_dos(energies, cbm_energy, m_eff=m_eff_rel)

    return DOSResult(
        energies          = energies,
        vasp_dos          = vasp_dos,
        free_electron_dos = fe_dos,
        E_c               = E_c,
        m_eff_rel         = m_eff_rel,
        m_eff_si          = m_eff_si,
        fit_quality       = fit_quality,
        cell_volume_m3    = vol_m3,
        carrier           = carrier,
        em_electrons      = em_electrons,
        em_holes          = em_holes,
        warnings          = warns,
    )
