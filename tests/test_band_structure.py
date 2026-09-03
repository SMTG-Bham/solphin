"""High-symmetry paths and band-structure assembly.

The k-point count is deliberately never asserted. The notebook and a fresh run
against the committed CONTCAR agree on 239, but the number is a function of the
``density`` argument and of sumo's path definition, neither of which this
package pins, so it is not a stable contract.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine

import castep_fixtures
import solphin.band_structure as band_structure
import solphin.castep_inputs as castep_inputs
import solphin.vasp_inputs as vasp_inputs


@pytest.fixture(scope="module")
def relaxed_structure(relax_dir: Path) -> Structure:
    """Provide the relaxed Cu2GeS3 structure read from the committed CONTCAR."""
    return vasp_inputs.read_structure_pmg(relax_dir / "CONTCAR")


@pytest.fixture(scope="module")
def castep_scf_inputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A written CASTEP SCF input set with a dummy .check, for band-setup tests."""
    structure = Structure(Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]])
    out_dir = tmp_path_factory.mktemp("castep_scf")

    cell_file = castep_inputs.write_castep_calculation(structure, "PBE", out_dir)
    cell_file.with_suffix(".check").write_bytes(b"not a real checkpoint")
    return cell_file


# --- k-path generation -----------------------------------------------------


def test_generate_band_structure_path_returns_structure_and_path(
        relaxed_structure: Structure
) -> None:
    """The generated path has 3-vector k-points, matching labels, and labelled endpoints."""
    canonical, (kpoints, labels) = band_structure.generate_band_structure_path(
        structure=relaxed_structure, definition="bradcrack"
    )

    assert canonical.composition.reduced_formula == "Cu2GeS3"
    assert np.asarray(kpoints).shape[1] == 3
    assert len(labels) == len(kpoints)
    # Endpoints of a high-symmetry path are always labelled.
    assert labels[0] and labels[-1]


def test_generate_band_structure_path_is_deterministic(relaxed_structure: Structure) -> None:
    """Two runs over the same structure give identical k-points and labels."""
    first = band_structure.generate_band_structure_path(structure=relaxed_structure)
    second = band_structure.generate_band_structure_path(structure=relaxed_structure)

    np.testing.assert_allclose(np.asarray(first[1][0]), np.asarray(second[1][0]))
    assert first[1][1] == second[1][1]


# --- assembling the split calculations -------------------------------------


def test_get_band_structure_returns_symmline(band_structure_obj: BandStructureSymmLine) -> None:
    """The reconstruction yields a BandStructureSymmLine with a Fermi level and branches."""
    assert isinstance(band_structure_obj, BandStructureSymmLine)
    assert band_structure_obj.efermi is not None
    assert len(band_structure_obj.branches) > 0


def test_get_band_structure_reference_gap(band_structure_obj: BandStructureSymmLine) -> None:
    """Cu2GeS3 has a 1.39 eV direct gap at Gamma in the committed calculation."""
    gap = band_structure_obj.get_band_gap()

    assert gap["energy"] == pytest.approx(1.3901, rel=1e-4)
    assert gap["direct"] is True
    assert gap["transition"] == r"\Gamma-\Gamma"


def test_direct_gap_at_least_fundamental(band_structure_obj: BandStructureSymmLine) -> None:
    """The smallest vertical transition can never be below the global minimum gap."""
    fundamental = band_structure_obj.get_band_gap()["energy"]

    assert band_structure_obj.get_direct_band_gap() >= fundamental - 1e-9


def test_plot_band_structure_smoke(band_structure_obj: BandStructureSymmLine) -> None:
    """Plotting the band structure produces a matplotlib figure."""
    import matplotlib.pyplot as plt

    band_structure.plot_band_structure(band_structure_obj, plt, ymin=-6, ymax=6)

    assert plt.get_fignums()


# --- CASTEP ----------------------------------------------------------------


@pytest.fixture(scope="module")
def castep_band_structure_obj(castep_band_dir: Path) -> BandStructureSymmLine:
    """The reconstructed band structure of the synthetic Gamma-X-M fixture."""
    return band_structure.get_band_structure(str(castep_band_dir), 1, code="castep")


def test_castep_band_fixtures_regenerate_byte_identically(castep_band_dir: Path) -> None:
    """The committed toy.bands/toy.cell pair is exactly what castep_fixtures emits."""
    assert (castep_band_dir / "toy.bands").read_text() == castep_fixtures.band_bands_text()
    assert (castep_band_dir / "toy.cell").read_text() == castep_fixtures.band_cell_text()


def test_castep_get_band_structure_known_gap(
        castep_band_structure_obj: BandStructureSymmLine,
) -> None:
    """The cosine-band fixture has a 1.5 eV direct gap at Gamma by construction."""
    assert isinstance(castep_band_structure_obj, BandStructureSymmLine)

    gap = castep_band_structure_obj.get_band_gap()

    assert gap["energy"] == pytest.approx(1.5, abs=1e-6)
    assert gap["direct"] is True
    assert gap["transition"] == r"\Gamma-\Gamma"


def test_castep_get_band_structure_reads_labels_from_cell(
        castep_band_structure_obj: BandStructureSymmLine,
) -> None:
    """The high-symmetry labels come from the sibling .cell's k-point block."""
    labels = {
        kpoint.label
        for kpoint in castep_band_structure_obj.kpoints
        if kpoint.label is not None
    }

    assert labels == {r"\Gamma", "X", "M"}


def test_castep_get_band_structure_split_layout(
        castep_band_dir: Path, tmp_path: Path
) -> None:
    """Asking for splits reads split-NN folders and reproduces the single-file result."""
    # One folder per split: splits=1 takes the single-file branch, so the split
    # layout needs two to be exercised at all. This laid down one folder and
    # asked for two, which passed only while the count was ignored.
    for name in ("split-01", "split-02"):
        split_dir = tmp_path / name
        split_dir.mkdir()
        shutil.copyfile(castep_band_dir / "toy.bands", split_dir / "toy.bands")

    bs = band_structure.get_band_structure(str(tmp_path), 2, code="castep")

    assert bs.get_band_gap()["energy"] == pytest.approx(1.5, abs=1e-6)


def test_castep_get_band_structure_missing_bands_raises(tmp_path: Path) -> None:
    """An empty directory raises FileNotFoundError for both layouts."""
    with pytest.raises(FileNotFoundError):
        band_structure.get_band_structure(str(tmp_path), 1, code="castep")

    with pytest.raises(FileNotFoundError):
        band_structure.get_band_structure(str(tmp_path), 2, code="castep")


def test_castep_get_band_structure_ambiguous_bands_raises(
        castep_band_dir: Path, tmp_path: Path
) -> None:
    """Two .bands files in a single-calculation directory raise, naming both."""
    for seed in ("a", "b"):
        shutil.copyfile(castep_band_dir / "toy.bands", tmp_path / f"{seed}.bands")

    with pytest.raises(ValueError, match="a.bands.*b.bands"):
        band_structure.get_band_structure(str(tmp_path), 1, code="castep")


def test_get_band_structure_unknown_code_raises(castep_band_dir: Path) -> None:
    """An unsupported code name raises instead of falling back to VASP."""
    with pytest.raises(ValueError, match="banana"):
        band_structure.get_band_structure(str(castep_band_dir), 1, code="banana")


def test_castep_cell_for_requires_kpoint_block(
        castep_band_dir: Path, tmp_path: Path
) -> None:
    """Only a .cell with a k-point path/list block is offered to sumo.

    sumo's labels_from_cell calls sys.exit on a cell without one, so this
    guard is what keeps a plain SCF cell from killing the interpreter.
    """
    assert band_structure._castep_cell_for(castep_band_dir / "toy.bands") is not None

    bare = tmp_path / "toy.bands"
    bare.write_text("")
    assert band_structure._castep_cell_for(bare) is None

    (tmp_path / "toy.cell").write_text(
        "%block lattice_cart\n5 0 0\n0 5 0\n0 0 5\n%endblock lattice_cart\n"
    )
    assert band_structure._castep_cell_for(bare) is None


def test_write_castep_band_structure_calculation_single(
        castep_scf_inputs: Path, tmp_path: Path
) -> None:
    """splits=1 writes band.cell plus the adjusted .param and reused .check."""
    structure = vasp_inputs.read_structure_pmg(castep_scf_inputs)
    _, kpath = band_structure.generate_band_structure_path(structure, density=10)
    band_dir = tmp_path / "band"

    band_structure.write_castep_band_structure_calculation(
        castep_scf_inputs, kpath, band_dir
    )

    cell_text = (band_dir / "band.cell").read_text()
    param_text = (band_dir / "Si.param").read_text()

    assert "%block spectral_kpoint_list" in cell_text
    assert "task : Spectral" in param_text
    assert "spectral_task : BandStructure" in param_text
    assert "reuse : Si.check" in param_text
    assert (band_dir / "Si.check").is_file()


def test_write_castep_band_structure_calculation_splits(
        castep_scf_inputs: Path, tmp_path: Path
) -> None:
    """splits=2 delegates to sumo's folder mode: cell, param and check per split."""
    structure = vasp_inputs.read_structure_pmg(castep_scf_inputs)
    _, kpath = band_structure.generate_band_structure_path(structure, density=10)
    band_dir = tmp_path / "band"

    band_structure.write_castep_band_structure_calculation(
        castep_scf_inputs, kpath, band_dir, splits=2
    )

    splits = sorted(p.name for p in band_dir.iterdir())
    assert splits == ["split-01", "split-02"]
    for split in splits:
        names = {p.name for p in (band_dir / split).iterdir()}
        assert {"Si.cell", "Si.param", "Si.check"} <= names
        assert "%block spectral_kpoint_list" in (band_dir / split / "Si.cell").read_text()


def test_write_castep_band_structure_calculation_missing_param_raises(
        castep_scf_inputs: Path, tmp_path: Path
) -> None:
    """A cell without its sibling .param raises instead of printing an error."""
    structure = vasp_inputs.read_structure_pmg(castep_scf_inputs)
    _, kpath = band_structure.generate_band_structure_path(structure, density=10)
    orphan_cell = tmp_path / "orphan.cell"
    shutil.copyfile(castep_scf_inputs, orphan_cell)

    with pytest.raises(FileNotFoundError, match=".param"):
        band_structure.write_castep_band_structure_calculation(
            orphan_cell, kpath, tmp_path / "band"
        )


def test_write_castep_band_structure_calculation_missing_cell_raises(
        castep_scf_inputs: Path, tmp_path: Path
) -> None:
    """A nonexistent cell file raises up front."""
    structure = vasp_inputs.read_structure_pmg(castep_scf_inputs)
    _, kpath = band_structure.generate_band_structure_path(structure, density=10)

    with pytest.raises(FileNotFoundError, match="cell"):
        band_structure.write_castep_band_structure_calculation(
            tmp_path / "ghost.cell", kpath, tmp_path / "band"
        )


def test_castep_plot_band_structure_with_dos_panel(
        castep_band_structure_obj: BandStructureSymmLine, castep_band_dir: Path
) -> None:
    """The DOS side panel renders from a .bands file on the CASTEP path.

    sumo's pretty_subplot does not create axes when handed an existing plt
    (its own TODO says as much, for VASP and CASTEP alike), so the two-panel
    figure has to exist before the call.
    """
    import matplotlib.pyplot as plt

    plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1], "wspace": 0})

    band_structure.plot_band_structure(
        castep_band_structure_obj,
        plt,
        dos_file=str(castep_band_dir / "toy.bands"),
        code="castep",
    )

    assert plt.get_fignums()


def test_castep_plot_band_structure_scissor_rejected(
        castep_band_structure_obj: BandStructureSymmLine, castep_band_dir: Path
) -> None:
    """The CASTEP DOS reader has no scissor support, so asking for one raises."""
    import matplotlib.pyplot as plt

    with pytest.raises(ValueError, match="[Ss]cissor"):
        band_structure.plot_band_structure(
            castep_band_structure_obj,
            plt,
            dos_file=str(castep_band_dir / "toy.bands"),
            scissor=0.5,
            code="castep",
        )


# --- contracts that are not yet honoured -----------------------------------


def test_write_band_structure_missing_scf_raises(
        relaxed_structure: Structure, tmp_path: Path
) -> None:
    """A hybrid setup missing SCF k-points should raise instead of printing an error."""
    canonical, kpath = band_structure.generate_band_structure_path(
        structure=relaxed_structure
    )

    with pytest.raises(ValueError, match="scf_kpoints"):
        band_structure.write_band_structure_calculation(
            structure=canonical,
            kpath=kpath,
            band_directory=str(tmp_path),
            functional="HSE06",
            splits=2,
            scf_charge=None,
            scf_kpoints=None,
        )


def test_write_band_structure_missing_charge_raises(
        relaxed_structure: Structure, tmp_path: Path
) -> None:
    """The GGA branch carried the same print-and-return defect as the hybrid one."""
    canonical, kpath = band_structure.generate_band_structure_path(
        structure=relaxed_structure
    )

    with pytest.raises(ValueError, match="scf_charge"):
        band_structure.write_band_structure_calculation(
            structure=canonical,
            kpath=kpath,
            band_directory=str(tmp_path),
            functional="PBE",
            splits=2,
            scf_charge=None,
            scf_kpoints=None,
        )


def test_write_band_structure_missing_scf_writes_nothing(
        relaxed_structure: Structure, tmp_path: Path
) -> None:
    """The guard must fire before anything lands on disk."""
    canonical, kpath = band_structure.generate_band_structure_path(
        structure=relaxed_structure
    )

    with pytest.raises(ValueError):
        band_structure.write_band_structure_calculation(
            structure=canonical,
            kpath=kpath,
            band_directory=str(tmp_path),
            functional="HSE06",
            splits=2,
            scf_charge=None,
            scf_kpoints=None,
        )

    assert list(tmp_path.iterdir()) == []


def test_splits_argument_is_honoured(
        band_dir: Path, band_structure_obj: BandStructureSymmLine
) -> None:
    """Reading a subset of the splits should not reproduce the full path."""
    subset = band_structure.get_band_structure(str(band_dir), 3)

    assert len(subset.kpoints) < len(band_structure_obj.kpoints)


def test_splits_argument_reads_exactly_that_many(band_dir: Path) -> None:
    """The k-point count must track the number of splits asked for.

    Reading more splits must strictly add k-points - "fewer than all seven"
    alone would still pass if the slice were off by one. Counts start at 2
    because splits=1 takes the single-file branch, not the split branch.
    """
    counts = [
        len(band_structure.get_band_structure(str(band_dir), n).kpoints)
        for n in (2, 3, 4)
    ]

    assert counts[0] < counts[1] < counts[2]


def test_splits_beyond_those_present_raises(band_dir: Path) -> None:
    """Asking for more splits than exist must fail, not silently truncate."""
    with pytest.raises(FileNotFoundError, match="only 7"):
        band_structure.get_band_structure(str(band_dir), 8)
