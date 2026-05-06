from matplotlib import pyplot as plt
from sumo.cli.dosplot import dosplot
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
from pymatgen.io.vasp import Vasprun
import os, glob, numpy as np

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

from scipy.ndimage import gaussian_filter1d
from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin
import scipy.constants as sc
from scipy.constants import physical_constants as pc


HBAR = sc.hbar   # J·s
M_E  = pc["atomic unit of mass"][0]   # kg
EV   = sc.e    # J per eV


def plot_dos(filename, xmin=-3, xmax=3, gaussian=0.05, save=False):
    fig, ax = plt.subplots(figsize=(5,3), dpi=150)
    dosplot(filename=filename, xmin=xmin, xmax=xmax, gaussian=gaussian, plt=plt)

    if save:
        plt.savefig("dos.pdf")

    plt.show()
    return


@dataclass
class SegmentMass:

    label:       str    # e.g. "Γ→X"
    m_eff_rel:   float  # m* / m_e
    m_eff_si:    float  # kg
    fit_quality: float  # R²
    n_points:    int    # k-points used in fit


@dataclass
class EffectiveMassResult:

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

    @property
    def em_result(self) -> Optional[EffectiveMassResult]:

        if self.carrier == "electrons":
            return self.em_electrons
        return self.em_holes

    def __str__(self):
        return _format_dos_summary(self)


def _is_soc_vasprun(vr: Vasprun) -> bool:

    try:
        return bool(vr.incar.get("LSORBIT", False))
    except Exception:
        return False


def _detect_spin_channel(bs: BandStructure, edge_info: dict) -> Spin:

    if len(bs.bands) == 1:
        return Spin.up
    if Spin.up in edge_info["band_index"] and edge_info["band_index"][Spin.up]:
        return Spin.up
    return Spin.down


def _get_efermi(vr: Vasprun, path: str) -> float:

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


def _find_split_vaspruns(directory: str) -> Tuple[list, bool]:

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
    )


def _parse_single_bs(path: str) -> tuple:

    vr           = Vasprun(path, parse_dos=False, parse_eigen=True)
    kpoints_file = os.path.join(os.path.dirname(os.path.abspath(path)), "KPOINTS")
    efermi       = _get_efermi(vr, path)
    bs = vr.get_band_structure(
        kpoints_filename = kpoints_file if os.path.isfile(kpoints_file) else None,
        efermi           = efermi,
        line_mode        = True,
    )
    return bs, _is_soc_vasprun(vr)


def parse_split_bs(vaspruns: list) -> tuple:

    parsed = []
    is_soc = False
    for path in vaspruns:
        vr           = Vasprun(path, parse_dos=False, parse_eigen=True)
        kpoints_file = os.path.join(os.path.dirname(path), "KPOINTS")

        if not os.path.isfile(kpoints_file):
            raise FileNotFoundError(
                f"KPOINTS file not found alongside {path}. "
                f"Expected at {kpoints_file}."
            )

        is_soc = is_soc or _is_soc_vasprun(vr)
        efermi = _get_efermi(vr, path)
        bs     = vr.get_band_structure(
            kpoints_filename = kpoints_file,
            efermi           = efermi,
            line_mode        = True,
        )
        parsed.append(bs)

    ref = parsed[0]

    if len(parsed) == 1:
        return parsed[0], is_soc, ref
    
    return parsed, is_soc, ref

def band_match(ref, parsed):

    n_bands_min = {}
    for spin in ref.bands:
        counts = [bs.bands[spin].shape[0] for bs in parsed]
        n_bands_min[spin] = min(counts)

    return n_bands_min

def combine_band_kpoints(ref, parsed, n_bands_min):

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

    return all_kpoints, all_labels, all_bands

def convert_kpoints(all_kpoints, all_labels):

    kpoints_frac = [kp.frac_coords for kp in all_kpoints]
    labels_dict  = {
        label: all_kpoints[idx].frac_coords
        for label, idx in all_labels.items()
    }

    return kpoints_frac, labels_dict


def _merge_split_band_structures(vaspruns: list) -> tuple:

    parsed, is_soc, ref = parse_split_bs(vaspruns)

    n_bands_min = band_match(ref, parsed)

    all_kpoints, all_labels, all_bands = combine_band_kpoints(ref, parsed, n_bands_min)

    kpoints_frac, labels_dict = convert_kpoints(all_kpoints, all_labels)

    bs_merged = BandStructureSymmLine(
        kpoints     = kpoints_frac,
        eigenvals   = all_bands,
        lattice     = ref.lattice_rec,
        efermi      = ref.efermi,
        labels_dict = labels_dict,
        structure   = ref.structure,
    )
    return bs_merged, is_soc


def _load_band_structure(source: str) -> tuple:

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


def _fit_segment_mass(
    kpoints_cart: np.ndarray,
    energies:     np.ndarray,
    edge_kidx:    int,
    E_edge:       float,
    label:        str,
    n_points:     int,
    carrier:      str = "electrons",
) -> SegmentMass:

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
        return SegmentMass(
            label=label, m_eff_rel=np.nan, m_eff_si=np.nan,
            fit_quality=np.nan, n_points=n_used
        )

    slope, intercept = np.polyfit(dk_sq, dE, 1)

    if slope <= 0:

        return SegmentMass(
            label=label, m_eff_rel=np.nan, m_eff_si=np.nan,
            fit_quality=np.nan, n_points=n_used,
        )

    m_eff_si  = HBAR**2 / (2.0 * slope)
    m_eff_rel = m_eff_si / M_E

    dE_pred = slope * dk_sq + intercept
    ss_res  = np.sum((dE - dE_pred) ** 2)
    ss_tot  = np.sum((dE - np.mean(dE)) ** 2)
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return SegmentMass(
        label=label, m_eff_rel=m_eff_rel, m_eff_si=m_eff_si,
        fit_quality=r2, n_points=n_used, 
    )


def get_effective_mass(
    source:   "str | BandStructure",
    n_points: int = 5,
    carrier:  str = "electrons",
) -> EffectiveMassResult:

    if carrier not in ("electrons", "holes"):
        raise ValueError(f"carrier must be 'electrons' or 'holes', got '{carrier}'.")

    if isinstance(source, (BandStructure, BandStructureSymmLine)):
        bs     = source
        is_soc = False   # can't determine without vasprun — assume not SOC
    else:
        bs, is_soc = _load_band_structure(source)
        
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
    )

def free_electron_dos(
    energies: np.ndarray,
    E_c:      float,
    m_eff:    float = 1.0,
) -> np.ndarray:

    prefactor = (1.0 / (2.0 * np.pi**2)) * (2.0 * m_eff * M_E / HBAR**2) ** 1.5
    dE        = np.maximum(energies - E_c, 0.0) * EV
    return prefactor * np.sqrt(dE) * EV


def _format_em_table(em: EffectiveMassResult, is_dos_carrier: bool) -> list:
    
    edge_label  = "CBM" if em.carrier == "electrons" else "VBM"
    sub = "ₑ" if em.carrier == "electrons" else "ₕ"

    carrier_str = em.carrier.capitalize()
    dos_marker  = "  ← used in DOS equation" if is_dos_carrier else ""

    lines = [
        f"  {carrier_str} (fitted at {edge_label} = {em.E_edge:.4f} eV)",
        f"  {'Harmonic mean m*':<20}: {em.m_eff_rel:.4f} m{sub}"
        f"  ({em.m_eff_si:.4e} kg){dos_marker}",
    ]

    valid_segs = [s for s in em.segments if np.isfinite(s.fit_quality)]
    if valid_segs:
        mean_r2 = np.mean([s.fit_quality for s in valid_segs])
        lines.append(f"  {'Mean fit quality R²':<20}: {mean_r2:.4f}")

    lines += [
        "",
        f"{'Segment':<20} {'m*':>10} (m{sub}) {'R²':>8} {'N pts':>7}",
        "  " + "-" * 47,
    ]

    lines.append(f"  {'Harmonic mean':<20} {em.m_eff_rel:>10.4f}")

    return lines


def _format_dos_summary(result: "DOSResult") -> str:

    edge_label = "CBM" if result.carrier == "electrons" else "VBM"

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
        lines.append("  Holes : mₕ* supplied directly — no band structure fit.")

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

    lines += ["=" * 60, ""]
    return "\n".join(lines)


def print_dos_summary(result: "DOSResult") -> None:

    print(_format_dos_summary(result))


def compute_dos(
    dos_vasprun:  str,
    bs_vasprun:   Optional[str] = None,
    bs_directory: Optional[str] = None,
    m_eff:        Optional[float] = None,
    sigma:        float = 0.05,
    n_fit_points: int = 5,
    carrier:      str = "electrons",
) -> DOSResult:
  
    if carrier not in ("electrons", "holes"):
        raise ValueError(f"carrier must be 'electrons' or 'holes', got '{carrier}'.")

    n_sources = sum(x is not None for x in [bs_vasprun, bs_directory, m_eff])
    if n_sources == 0:
        raise ValueError("Provide exactly one of: bs_vasprun, bs_directory, or m_eff.")
    if n_sources > 1:
        raise ValueError(
            "Provide exactly one of bs_vasprun, bs_directory, or m_eff — not multiple."
        )


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
        bs, _ = _load_band_structure(source)

        print("  Computing electron effective mass (CBM)...")
        try:
            em_electrons = get_effective_mass(bs, n_points=n_fit_points, carrier="electrons")
        except Exception as e:
            warnings.warn(f"Electron m* fit failed: {e}", UserWarning, stacklevel=2)

        print("  Computing hole effective mass (VBM)...")
        try:
            em_holes = get_effective_mass(bs, n_points=n_fit_points, carrier="holes")
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
    )
