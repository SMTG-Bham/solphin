"""High-symmetry paths and band-structure assembly.

The k-point count is deliberately never asserted: the notebook records 179 for
its path while a fresh run against the committed CONTCAR gives 239, so the
count depends on inputs that have drifted and is not a stable contract.
"""

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine

import solphin.band_structure as band_structure
import solphin.vasp_inputs as vasp_inputs


@pytest.fixture(scope="module")
def relaxed_structure(relax_dir: Path) -> Structure:
    return vasp_inputs.read_structure_pmg(relax_dir / "CONTCAR")


# --- k-path generation -----------------------------------------------------


def test_generate_band_structure_path_returns_structure_and_path(
        relaxed_structure: Structure
) -> None:
    canonical, (kpoints, labels) = band_structure.generate_band_structure_path(
        structure=relaxed_structure, definition="bradcrack"
    )

    assert canonical.composition.reduced_formula == "Cu2GeS3"
    assert np.asarray(kpoints).shape[1] == 3
    assert len(labels) == len(kpoints)
    # Endpoints of a high-symmetry path are always labelled.
    assert labels[0] and labels[-1]


def test_generate_band_structure_path_is_deterministic(relaxed_structure: Structure) -> None:
    first = band_structure.generate_band_structure_path(structure=relaxed_structure)
    second = band_structure.generate_band_structure_path(structure=relaxed_structure)

    np.testing.assert_allclose(np.asarray(first[1][0]), np.asarray(second[1][0]))
    assert first[1][1] == second[1][1]


# --- assembling the split calculations -------------------------------------


def test_get_band_structure_returns_symmline(band_structure_obj: BandStructureSymmLine) -> None:
    assert isinstance(band_structure_obj, BandStructureSymmLine)
    assert band_structure_obj.efermi is not None
    assert len(band_structure_obj.branches) > 0


def test_get_band_structure_tutorial_gap(band_structure_obj: BandStructureSymmLine) -> None:
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
    import matplotlib.pyplot as plt

    band_structure.plot_band_structure(band_structure_obj, plt, ymin=-6, ymax=6)

    assert plt.get_fignums()


# --- contracts that are not yet honoured -----------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "band_structure.py:212 prints 'ERROR: ...' and returns None when the "
        "hybrid path is missing scf_kpoints, so a caller cannot detect the failure"
    ),
)
def test_write_band_structure_missing_scf_raises(
        relaxed_structure: Structure, tmp_path: Path
) -> None:
    canonical, kpath = band_structure.generate_band_structure_path(
        structure=relaxed_structure
    )

    with pytest.raises((ValueError, TypeError, FileNotFoundError)):
        band_structure.write_band_structure_calculation(
            structure=canonical,
            kpath=kpath,
            band_directory=str(tmp_path),
            functional="HSE06",
            splits=2,
            scf_charge=None,
            scf_kpoints=None,
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "get_band_structure globs every split-*/vasprun.xml whenever splits > 1, "
        "so the argument's value is ignored - asking for 3 still reads all 7"
    ),
)
def test_splits_argument_is_honoured(
        band_dir: Path, band_structure_obj: BandStructureSymmLine
) -> None:
    """Reading a subset of the splits should not reproduce the full path."""
    subset = band_structure.get_band_structure(str(band_dir), 3)

    assert len(subset.kpoints) < len(band_structure_obj.kpoints)
