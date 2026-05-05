import matplotlib
from matplotlib import pyplot as plt
from sumo.cli.dosplot import dosplot
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

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
for any material, with support for:

  - Hybrid (HSE/PBE0) band structures split across split_01/, split_02/, ...
  - Anisotropic effective masses per k-path segment
  - Harmonic-mean scalar average (physically correct for transport)
  - DOS/optics vaspruns with an externally supplied m*

Usage
-----
# From a split hybrid band structure:
result = compute_dos(
    dos_vasprun  = "path/to/dos/vasprun.xml",
    bs_directory = "path/to/band/",   # contains split_01/, split_02/, ...
)

# From a single standard band structure vasprun:
result = compute_dos(
    dos_vasprun = "path/to/dos/vasprun.xml",
    bs_vasprun  = "path/to/band/vasprun.xml",
)

# With a manually supplied m*:
result = compute_dos(
    dos_vasprun = "path/to/dos/vasprun.xml",
    m_eff       = 0.3,
)

print(result)   # summary table
"""

import warnings
import glob
import os
from dataclasses import dataclass, field
from typing import Optional

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
    """Full anisotropic + averaged effective mass result."""
    # Scalar average (harmonic mean over segments — transport-relevant)
    m_eff_rel:   float
    m_eff_si:    float
    # Per-segment anisotropic masses
    segments:    list        # list[SegmentMass]
    E_c:         float       # CBM energy in eV
    k_c:         np.ndarray  # CBM k-point, Cartesian 1/m
    warnings:    list        = field(default_factory=list)

    def __str__(self):
        lines = [
            "",
            "=" * 55,
            "  Effective Mass Results",
            "=" * 55,
            f"  CBM energy          : {self.E_c:.4f} eV",
            "",
            "  Anisotropic masses (per k-path segment):",
            f"  {'Segment':<12} {'m* (mₑ)':>10} {'R²':>8} {'N pts':>7}",
            "  " + "-" * 43,
        ]
        for seg in self.segments:
            warn_flag = " ⚠" if seg.warnings else ""
            if np.isfinite(seg.m_eff_rel):
                lines.append(
                    f"  {seg.label:<12} {seg.m_eff_rel:>10.4f} "
                    f"{seg.fit_quality:>8.4f} {seg.n_points:>7}{warn_flag}"
                )
            else:
                lines.append(
                    f"  {seg.label:<12} {'N/A':>10} {'N/A':>8} {seg.n_points:>7}{warn_flag}"
                )
        lines += [
            "  " + "-" * 43,
            f"  {'Harmonic mean':<12} {self.m_eff_rel:>10.4f}",
            "=" * 55,
        ]
        if self.warnings:
            lines.append("\n  Warnings:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)


@dataclass
class DOSResult:
    """Combined VASP + free-electron DOS result."""
    energies:          np.ndarray
    vasp_dos:          np.ndarray        # states / eV / cell
    free_electron_dos: np.ndarray        # states / eV / m³
    E_c:               float             # CBM in eV
    m_eff_rel:         float             # m* / m_e used
    m_eff_si:          float             # kg
    fit_quality:       Optional[float]   # mean R² over segments, or None
    cell_volume_m3:    float             # m³
    em_result:         Optional[EffectiveMassResult] = None
    warnings:          list              = field(default_factory=list)

    def __str__(self):
        lines = [
            "",
            "=" * 55,
            "  DOS Result Summary",
            "=" * 55,
            f"  CBM energy          : {self.E_c:.4f} eV",
            f"  Effective mass m*   : {self.m_eff_rel:.4f} mₑ",
            f"  Fit quality (R²)    : "
                + (f"{self.fit_quality:.4f}" if self.fit_quality is not None
                   else "N/A (m* supplied directly)"),
            f"  Cell volume         : {self.cell_volume_m3:.4e} m³",
            f"  Energy range        : {self.energies[0]:.3f} → {self.energies[-1]:.3f} eV",
            f"  DOS grid points     : {len(self.energies)}",
            "=" * 55,
        ]
        if self.em_result is not None:
            lines.append(str(self.em_result))
        if self.warnings:
            lines.append("\n  Warnings:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Split-folder helpers
# ---------------------------------------------------------------------------

def _find_split_vaspruns(directory: str) -> list:
    """
    Search `directory` for split_*/vasprun.xml in sorted order.
    Never returns the root vasprun.xml — splits only.
    Raises FileNotFoundError if no splits are found.

    Parameters
    ----------
    directory : str
        Root directory of the band structure calculation.

    Returns
    -------
    list[str]
        Sorted list of vasprun.xml paths to merge.
    """
    pattern = os.path.join(directory, "split-*", "vasprun.xml")
    splits  = sorted(glob.glob(pattern))

    if not splits:
        raise FileNotFoundError(
            f"No split-*/vasprun.xml found in '{directory}'. "
            f"For a single vasprun, use bs_vasprun instead of bs_directory."
        )

    return splits


def _merge_split_band_structures(vaspruns: list) -> BandStructureSymmLine:
    """
    Parse and concatenate split band structure vaspruns into a single
    BandStructureSymmLine, preserving k-path ordering across splits.

    Each split vasprun must have a KPOINTS file in the same directory.

    Parameters
    ----------
    vaspruns : list[str]
        Ordered list of vasprun.xml paths (split_01, split_02, ...).

    Returns
    -------
    BandStructureSymmLine
        Merged band structure.
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
        bs = vr.get_band_structure(
            kpoints_filename = kpoints_file,
            efermi           = vr.efermi,
            line_mode        = True,
        )
        parsed.append(bs)

    if len(parsed) == 1:
        return parsed[0]

    ref         = parsed[0]
    all_kpoints = []
    all_labels  = {}
    kpt_offset  = 0
    all_bands   = {spin: [] for spin in ref.bands}

    for bs in parsed:
        for spin in all_bands:
            all_bands[spin].append(bs.bands[spin])   # (n_bands, n_kpts)

        all_kpoints.extend(bs.kpoints)

        for label, kpt in bs.labels_dict.items():
            for i, kp in enumerate(bs.kpoints):
                if np.allclose(kp.frac_coords, kpt.frac_coords, atol=1e-4):
                    all_labels[label] = i + kpt_offset
                    break

        kpt_offset += len(bs.kpoints)

    for spin in all_bands:
        all_bands[spin] = np.hstack(all_bands[spin])   # (n_bands, total_kpts)

    labels_dict = {label: all_kpoints[idx] for label, idx in all_labels.items()}

    return BandStructureSymmLine(
        kpoints     = all_kpoints,
        eigenvals   = all_bands,
        lattice     = ref.lattice_rec,
        efermi      = ref.efermi,
        labels_dict = labels_dict,
        structure   = ref.structure,
    )


# ---------------------------------------------------------------------------
# Effective mass fitting
# ---------------------------------------------------------------------------

def _fit_segment_mass(
    kpoints_cart: np.ndarray,
    energies:     np.ndarray,
    cbm_kidx:     int,
    E_c:          float,
    label:        str,
    n_points:     int,
) -> SegmentMass:
    """
    Fit a parabola to one k-path segment near the CBM.

        E(k) = E_c + (ħ²/2m*)|k - k_c|²

    Parameters
    ----------
    kpoints_cart : np.ndarray, shape (N, 3)
        Cartesian k-points in 1/m for this segment.
    energies : np.ndarray, shape (N,)
        Band energies in eV for this segment.
    cbm_kidx : int
        Index of the CBM k-point within this segment.
    E_c : float
        CBM energy in eV.
    label : str
        Human-readable segment label, e.g. "Γ→X".
    n_points : int
        Number of k-points either side of the CBM to fit.

    Returns
    -------
    SegmentMass
    """
    warns = []
    k_c   = kpoints_cart[cbm_kidx]

    i_lo  = max(0, cbm_kidx - n_points)
    i_hi  = min(len(kpoints_cart), cbm_kidx + n_points + 1)

    seg_k = kpoints_cart[i_lo:i_hi]
    seg_E = energies[i_lo:i_hi]

    dk_sq = np.sum((seg_k - k_c) ** 2, axis=1)
    dE    = (seg_E - E_c) * EV   # convert to J

    # Exclude the CBM itself (dk = 0)
    mask  = dk_sq > 0
    dk_sq = dk_sq[mask]
    dE    = dE[mask]
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
        msg = (
            f"Segment {label}: negative slope ({slope:.3e}) — "
            "band curves downward at this point, not a CBM. Skipping."
        )
        warns.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=4)
        return SegmentMass(
            label=label, m_eff_rel=np.nan, m_eff_si=np.nan,
            fit_quality=np.nan, n_points=n_used, warnings=warns,
        )

    m_eff_si  = HBAR**2 / (2.0 * slope)
    m_eff_rel = m_eff_si / M_E

    # R²
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
        label       = label,
        m_eff_rel   = m_eff_rel,
        m_eff_si    = m_eff_si,
        fit_quality = r2,
        n_points    = n_used,
        warnings    = warns,
    )


def get_effective_mass(
    source:   "str | BandStructure",
    n_points: int = 5,
) -> EffectiveMassResult:
    """
    Compute anisotropic and averaged effective masses at the CBM.

    Supports:
    - A path to a directory containing split_01/, split_02/, ... (hybrid calcs)
    - A path to a single band structure vasprun.xml (standard calcs)
    - A pre-parsed pymatgen BandStructure / BandStructureSymmLine object

    Parameters
    ----------
    source : str or BandStructure
        - Directory path containing split_*/vasprun.xml  (hybrid)
        - Path to a single vasprun.xml  (standard)
        - Pre-parsed BandStructure object
    n_points : int
        k-points either side of the CBM to include per segment fit.

    Returns
    -------
    EffectiveMassResult
        Anisotropic per-segment masses and harmonic-mean scalar average.
    """
    warns = []

    # ------------------------------------------------------------------
    # 1. Obtain the band structure
    # ------------------------------------------------------------------
    if isinstance(source, (BandStructure, BandStructureSymmLine)):
        bs = source

    elif os.path.isdir(source):
        split_paths = _find_split_vaspruns(source)
        print(f"  Found {len(split_paths)} split folder(s) — merging band structures.")
        bs = _merge_split_band_structures(split_paths)

    elif os.path.isfile(source):
        vr           = Vasprun(source, parse_dos=False, parse_eigen=True)
        kpoints_file = os.path.join(os.path.dirname(os.path.abspath(source)), "KPOINTS")
        bs = vr.get_band_structure(
            kpoints_filename = kpoints_file if os.path.isfile(kpoints_file) else None,
            efermi           = vr.efermi,
            line_mode        = True,
        )

    else:
        raise FileNotFoundError(f"Source not found: '{source}'")

    if bs.is_metal():
        raise ValueError(
            "Band structure indicates a metal — parabolic effective mass "
            "and the free-electron DOS formula are not applicable."
        )

    # ------------------------------------------------------------------
    # 2. Locate CBM
    # ------------------------------------------------------------------
    cbm_info = bs.get_cbm()
    E_c      = cbm_info["energy"]

    if Spin.up in cbm_info["band_index"] and cbm_info["band_index"][Spin.up]:
        spin     = Spin.up
        cbm_band = cbm_info["band_index"][Spin.up][0]
    else:
        spin     = Spin.down
        cbm_band = cbm_info["band_index"][Spin.down][0]

    cbm_kidx_global  = cbm_info["kpoint_index"][0]
    kpoints_cart_all = np.array([kp.cart_coords for kp in bs.kpoints]) * 1e10  # 1/Å → 1/m
    k_c              = kpoints_cart_all[cbm_kidx_global]
    band_all         = np.array(bs.bands[spin][cbm_band])

    # ------------------------------------------------------------------
    # 3. Anisotropic fit — one mass per k-path segment
    # ------------------------------------------------------------------
    segment_masses = []

    if isinstance(bs, BandStructureSymmLine):
        for seg in bs.branches:
            i_lo  = seg["start_index"]
            i_hi  = seg["end_index"] + 1   # inclusive
            label = seg["name"]

            seg_energies = band_all[i_lo:i_hi]

            # Only fit segments that contain or are very close to the CBM
            if not np.any(seg_energies <= E_c + 0.5):
                continue

            local_cbm = int(np.argmin(np.abs(seg_energies - E_c)))
            pretty    = label.replace("GAMMA", "Γ").replace("|", "").replace("-", "→")

            sm = _fit_segment_mass(
                kpoints_cart = kpoints_cart_all[i_lo:i_hi],
                energies     = seg_energies,
                cbm_kidx     = local_cbm,
                E_c          = E_c,
                label        = pretty,
                n_points     = n_points,
            )
            segment_masses.append(sm)
            warns.extend(sm.warnings)

    else:
        # Non-line-mode fallback: single global fit
        msg = (
            "Band structure is not line-mode — fitting a single isotropic "
            "mass from the nearest k-points. Supply a line-mode band structure "
            "for anisotropic results."
        )
        warns.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

        sm = _fit_segment_mass(
            kpoints_cart = kpoints_cart_all,
            energies     = band_all,
            cbm_kidx     = cbm_kidx_global,
            E_c          = E_c,
            label        = "isotropic",
            n_points     = n_points,
        )
        segment_masses.append(sm)
        warns.extend(sm.warnings)

    if not segment_masses:
        raise ValueError(
            "No k-path segments contained the CBM. "
            "Check that the band structure covers the CBM k-point."
        )

    # ------------------------------------------------------------------
    # 4. Harmonic mean scalar average (transport effective mass)
    #    1/m*_avg = (1/N) Σ 1/m*_i   (valid fits only)
    # ------------------------------------------------------------------
    valid = [s for s in segment_masses if np.isfinite(s.m_eff_rel) and s.m_eff_rel > 0]

    if not valid:
        raise ValueError(
            "All segment fits failed. Cannot compute an average effective mass. "
            "Check that the band structure is line-mode and covers the CBM."
        )

    m_eff_rel_avg = 1.0 / np.mean([1.0 / s.m_eff_rel for s in valid])
    m_eff_si_avg  = m_eff_rel_avg * M_E

    return EffectiveMassResult(
        m_eff_rel = m_eff_rel_avg,
        m_eff_si  = m_eff_si_avg,
        segments  = segment_masses,
        E_c       = E_c,
        k_c       = k_c,
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

    Parameters
    ----------
    energies : np.ndarray
        Energy values in eV.
    E_c : float
        Conduction band minimum in eV.
    m_eff : float
        Effective mass relative to mₑ.

    Returns
    -------
    np.ndarray
        DOS in states / eV / m³, zero below E_c.
    """
    prefactor = (1.0 / (2.0 * np.pi**2)) * (2.0 * m_eff * M_E / HBAR**2) ** 1.5
    dE        = np.maximum(energies - E_c, 0.0) * EV
    return prefactor * np.sqrt(dE) * EV


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
) -> DOSResult:
    """
    Compute the VASP total DOS and free-electron DOS for any material.

    Exactly one of `bs_vasprun`, `bs_directory`, or `m_eff` must be supplied:

      bs_directory  Path to directory containing split_01/, split_02/, ...
                    Use this for hybrid (HSE/PBE0) band structures.
                    Any number of split_* folders is supported.
                    The root vasprun.xml in this directory is never used
                    for the band structure — only the split subfolders are.

      bs_vasprun    Path to a single line-mode band structure vasprun.xml.
                    Use this for standard (GGA/LDA) band structures.
                    A KPOINTS file must exist in the same directory.

      m_eff         Effective mass in units of mₑ, supplied directly.
                    Use this for DOS or optics vaspruns where no band
                    structure is available. E_c is taken from the DOS CBM.

    Parameters
    ----------
    dos_vasprun : str
        Path to the DOS/optics/band vasprun.xml (energies + VASP DOS).
    bs_vasprun : str, optional
        Path to a single band structure vasprun.xml.
    bs_directory : str, optional
        Path to a directory of split band structure folders (hybrid calcs).
    m_eff : float, optional
        Effective mass in units of mₑ.
    sigma : float
        Gaussian smearing in eV applied to the VASP DOS (0 = none).
    n_fit_points : int
        k-points either side of CBM used in each segment fit.

    Returns
    -------
    DOSResult
        Contains energies, VASP DOS, free-electron DOS, m*, and full
        anisotropic breakdown (if fitted from a band structure).
    """
    n_sources = sum(x is not None for x in [bs_vasprun, bs_directory, m_eff])
    if n_sources == 0:
        raise ValueError(
            "Provide exactly one of: bs_vasprun, bs_directory, or m_eff."
        )
    if n_sources > 1:
        raise ValueError(
            "Provide exactly one of bs_vasprun, bs_directory, or m_eff — not multiple."
        )

    warns = []

    # ------------------------------------------------------------------
    # Parse DOS vasprun — never call get_band_structure on this
    # ------------------------------------------------------------------
    vr       = Vasprun(dos_vasprun, parse_dos=True, parse_eigen=False)
    cdos     = vr.complete_dos
    energies = cdos.energies
    vasp_dos = cdos.get_densities()
    vol_m3   = vr.final_structure.volume * 1e-30   # Å³ → m³

    if sigma > 0:
        de       = energies[1] - energies[0]
        vasp_dos = gaussian_filter1d(vasp_dos, sigma=sigma / de)

    # ------------------------------------------------------------------
    # Effective mass + E_c
    # ------------------------------------------------------------------
    em_result   = None
    fit_quality = None

    if bs_vasprun is not None or bs_directory is not None:
        source    = bs_directory if bs_directory is not None else bs_vasprun
        em_result = get_effective_mass(source, n_points=n_fit_points)
        m_eff_rel = em_result.m_eff_rel
        m_eff_si  = em_result.m_eff_si
        E_c       = em_result.E_c
        valid_segs  = [s for s in em_result.segments if np.isfinite(s.fit_quality)]
        fit_quality = np.mean([s.fit_quality for s in valid_segs]) if valid_segs else None
        warns.extend(em_result.warnings)

    else:
        # m_eff supplied directly — get E_c from the DOS, no band structure needed
        m_eff_rel  = m_eff
        m_eff_si   = m_eff * M_E
        cbm, vbm   = cdos.get_cbm_vbm()
        E_c        = cbm
        if cdos.get_gap() == 0.0:
            msg = "System appears metallic — free-electron DOS formula may not be meaningful."
            warns.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

    # ------------------------------------------------------------------
    # Free-electron DOS
    # ------------------------------------------------------------------
    fe_dos = free_electron_dos(energies, E_c, m_eff=m_eff_rel)

    return DOSResult(
        energies          = energies,
        vasp_dos          = vasp_dos,
        free_electron_dos = fe_dos,
        E_c               = E_c,
        m_eff_rel         = m_eff_rel,
        m_eff_si          = m_eff_si,
        fit_quality       = fit_quality,
        cell_volume_m3    = vol_m3,
        em_result         = em_result,
        warnings          = warns,
    )
