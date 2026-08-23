"""The tutorial's own call sequence, end to end, against the committed data.

full_workflow_tutorial.ipynb records the numbers it produced when it was last
run. This walks the same chain and requires the same answers, so the notebook
and the code are pinned to each other in both directions - a regression anywhere
in the pipeline shows up here, and so does a change that quietly makes the
tutorial's printed output wrong.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import REPO_ROOT, requires_potcars
from numpy.typing import NDArray
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine

import solphin.band_structure as band_structure
import solphin.db_fom as db_fom
import solphin.dos as dos
import solphin.final_results as final_results
import solphin.optics as optics
import solphin.spectral as spectral
import solphin.vasp_inputs as vasp_inputs
from solphin.dos import DOSResult

# tutorial cell 41: "Set default values from Crovetto et al."
TAU = 1e-6  # s
DOPING_DENSITY = 1e10  # cm^-3
MU = 1e6  # cm^2 V^-1 s^-1
TCELL = 300  # K

# Every solphin entry point that writes to a caller-supplied directory. Used by
# the static notebook check below.
_WRITERS = (
    "write_vasp_calculation",
    "write_band_structure_calculation",
    "write_eff_mass",
    "generate_n_real",
    "generate_absorption",
)


def _tree_digest(root: Path) -> str:
    """Fingerprint every file under `root` by name, size and mtime.

    Size and mtime rather than content, so that a file rewritten with byte-identical
    contents still trips the guard - the point is that nothing was written at all.
    """
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            h.update(path.relative_to(root).as_posix().encode())
            h.update(str(path.stat().st_size).encode())
            h.update(str(path.stat().st_mtime_ns).encode())
    return h.hexdigest()


def test_full_workflow_reproduces_tutorial(
        opt_dir: Path, dos_vasprun: Path, band_dir: Path, am15: NDArray
) -> None:
    """Cells 21, 25, 27, 29, 34, 39 and 42 in the order the notebook runs them."""
    # cell 21 - DOS effective mass
    dos_mass = dos.compute_dos(
        dos_vasprun=str(dos_vasprun), carrier="electrons", energy_window=0.1
    ).final_result

    # cell 25 - high-frequency dielectric constant
    eps_inf, _, _, _, _ = optics.calc_dielectric(filename=str(opt_dir / "vasprun.xml"))

    # cell 27 - direct gap from the recombined band structure
    bs = band_structure.get_band_structure(str(band_dir), 7)
    E_gap_direct = bs.get_direct_band_gap()

    # cells 29 and 34 - the illumination spectrum, raw and converted
    spectrum = db_fom.load_spectrum(spectrum_type="AM1.5")
    photon_spectrum = db_fom.convert_spectrum(spectrum)

    # cell 39 - absorption descriptors (needs the committed absorption.dat)
    spectral_average, spectral_dispersion = spectral.generate_spectral_parameters(
        str(opt_dir), spectrum, E_gap=E_gap_direct
    )

    # cell 42 - the headline figure of merit
    sq, fom_sq, eff = final_results.SQ_relative_FOM_PV_efficiency(
        E_gap_direct, photon_spectrum, spectral_average, TAU, spectral_dispersion,
        dos_mass, DOPING_DENSITY, eps_inf, MU, TCELL,
    )

    # The three numbers printed in the notebook's own recorded output.
    assert round(sq, 2) == 33.51
    assert round(fom_sq, 2) == 80.28
    assert round(eff, 2) == 26.90


def test_workflow_intermediate_values(
        dos_result: DOSResult, dielectric: tuple[float, NDArray, NDArray, NDArray, NDArray],
        band_structure_obj: BandStructureSymmLine
) -> None:
    """The inputs cell 42 is fed, each pinned where the notebook reports it."""
    assert dos_result.final_result == pytest.approx(0.0727540, rel=1e-6)
    assert dielectric[0] == pytest.approx(6.109433, rel=1e-6)
    assert band_structure_obj.get_direct_band_gap() == pytest.approx(1.3901, rel=1e-4)


def test_workflow_argument_order(photon_spectrum: NDArray) -> None:
    """Ten positional arguments, three of them interchangeable-looking floats.

    dos_mass, dop_density and epsilon sit 6th, 7th and 8th. Transposing any two
    must change the answer - if it did not, a caller could get the order wrong
    and never find out.
    """
    # Heterogeneous on purpose: this test unpacks .values() positionally to
    # prove the parameter order matters, so the annotation cannot be narrower.
    baseline: dict[str, Any] = dict(
        E_gap=1.3901, photon_spectrum=photon_spectrum, alpha=128846.05, tau=TAU,
        sigma=1.5663, dos_mass=0.0727540, dop_density=DOPING_DENSITY,
        epsilon=6.109433, mu=MU, Tcell=TCELL,
    )
    correct = final_results.SQ_relative_FOM_PV_efficiency(*baseline.values())

    swapped = dict(baseline)
    swapped["dos_mass"], swapped["epsilon"] = baseline["epsilon"], baseline["dos_mass"]
    transposed = final_results.SQ_relative_FOM_PV_efficiency(*swapped.values())

    assert correct[2] != pytest.approx(transposed[2], rel=1e-6)


def test_spectral_chain_requires_generated_dat(tmp_opt_dir: Path, am15: NDArray) -> None:
    """generate_spectral_parameters reads absorption.dat off disk rather than taking an array.

    The notebook satisfies this with the committed absorption.dat, so cell 30
    writing its regenerated copy into the workdir does not break the chain. For
    any other directory, the .dat has to be generated before this is called.
    """
    assert not (tmp_opt_dir / "absorption.dat").exists()

    with pytest.raises(OSError):
        spectral.generate_spectral_parameters(str(tmp_opt_dir), am15, E_gap=1.3901)

    optics.generate_absorption(str(tmp_opt_dir))

    average, dispersion = spectral.generate_spectral_parameters(
        str(tmp_opt_dir), am15, E_gap=1.3901
    )

    assert average == pytest.approx(128846.05, rel=1e-6)
    assert dispersion == pytest.approx(1.5663474, rel=1e-6)


def test_workflow_does_not_touch_tracked_data(
        tutorial_data: Path, opt_dir: Path, dos_vasprun: Path, am15: NDArray
) -> None:
    """Nothing in the analysis half may write into the committed fixture tree."""
    before = _tree_digest(tutorial_data)

    optics.calc_dielectric(filename=str(opt_dir / "vasprun.xml"))
    spectral.generate_spectral_parameters(str(opt_dir), am15, E_gap=1.3901)
    dos.compute_dos(dos_vasprun=str(dos_vasprun), energy_window=0.1)

    assert _tree_digest(tutorial_data) == before


@requires_potcars
def test_generation_half_does_not_touch_tracked_data(
        tutorial_data: Path, relax_dir: Path, opt_dir: Path, tmp_path: Path
) -> None:
    """Cells 7, 12, 17 and 20 write only into the directory they are handed.

    Every one of them used to default into `tutorial/Cu2GeS3`, overwriting the
    committed inputs the analysis half then reads - cell 7 clobbered the very
    POSCAR cell 6 had just read. They now take a workdir path instead.

    Kept separate from the analysis-half guard above rather than merged into it,
    because this one needs POTCARs and would otherwise take that guard out of CI.
    """
    before = _tree_digest(tutorial_data)
    structure = vasp_inputs.read_structure_pmg(relax_dir / "CONTCAR")

    # cell 7
    vasp_inputs.write_vasp_calculation(
        structure=structure, recipe="HSE06", out_dir=tmp_path / "Relax",
        patches=["relax_cell", "tight_relax"],
        user_incar_settings={"KSPACING": 0.2, "ENCUT": 450},
    )
    # cell 12
    vasp_inputs.write_vasp_calculation(
        structure=structure, recipe="HSE06", out_dir=tmp_path / "OPT_hybrid",
        patches=["optics"], user_incar_settings={"KSPACING": 0.2, "ENCUT": 450},
    )
    # cells 14 and 17. Two splits, not the notebook's five: this is a test of
    # where the files land, and each split costs a POTCAR build.
    canonical, kpath = band_structure.generate_band_structure_path(
        structure=structure, definition="bradcrack"
    )
    band_structure.write_band_structure_calculation(
        structure=canonical, kpath=kpath, band_directory=tmp_path / "BAND_SP_HDFT",
        functional="HSE06", splits=2, scf_charge=None,
        scf_kpoints=str(opt_dir / "IBZKPT"), user_incar_settings={"ENCUT": 450},
    )
    # cell 20
    dos.write_eff_mass(
        k0_frac=np.array([0.0, 0.0, 0.0]), structure=structure,
        functional="HSE06", encut=450, folder=str(tmp_path / "eff_mass"),
    )

    assert _tree_digest(tutorial_data) == before
    for name in ("Relax", "OPT_hybrid", "BAND_SP_HDFT", "eff_mass"):
        assert (tmp_path / name).is_dir(), f"{name} was not written to tmp_path"


def test_notebook_generation_cells_target_the_workdir() -> None:
    """Every notebook cell that writes names a workdir path, not the tracked tree.

    A string check on the committed JSON, because it is the only form of this
    guard that runs in CI - the real thing needs POTCARs. Extend `_WRITERS` when
    a new writing entry point is added.
    """
    notebook = json.loads(
        (REPO_ROOT / "tutorial" / "full_workflow_tutorial.ipynb").read_text()
    )

    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(name in source for name in _WRITERS):
            assert "workdir" in source or "_out" in source, (
                f"cell {i} writes but names no workdir destination"
            )
