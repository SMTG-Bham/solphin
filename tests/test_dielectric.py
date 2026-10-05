"""Ionic contribution to the static dielectric constant, for VASP and CASTEP.

Both fixtures encode the same ionic tensor - ((4, 1, 0), (1, 5, 0.5),
(0, 0.5, 6)), diagonal mean 5.0 - by different routes: VASP prints it
directly, CASTEP prints optical and DC permittivities whose difference it is.
``test_both_codes_agree`` holds the two generators to that shared answer, so
editing one without the other fails here rather than quietly drifting.

The error paths get as much attention as the happy one. A run that computed no
ionic response is the common way to get here - a hybrid with LEPSILON, which
VASP does not implement, or a CASTEP run without
``efield_calc_ion_permittivity`` - and in both cases the output file is
otherwise perfectly valid, so the reader has to notice the absence rather than
return a zero tensor.
"""

import gzip
from pathlib import Path

import numpy as np
import pytest

import castep_fixtures
import vasp_fixtures
from solphin import dielectric

EXPECTED_TENSOR = np.array(vasp_fixtures.IONIC_TENSOR)
EXPECTED_MEAN = 5.0


@pytest.fixture
def outcar(vasp_dfpt_dir: Path) -> Path:
    """The synthetic OUTCAR carrying the ionic dielectric block."""
    return vasp_dfpt_dir / "OUTCAR"


@pytest.fixture
def castep_file(castep_efield_dir: Path) -> Path:
    """The synthetic .castep file carrying the Efield permittivity table."""
    return castep_efield_dir / "toy.castep"


# --- the committed fixtures -------------------------------------------------


def test_castep_fixture_regenerates_byte_identically(castep_file: Path) -> None:
    """The committed .castep is exactly what its generator writes today."""
    assert castep_file.read_text(encoding="utf-8") == castep_fixtures.efield_castep_text()


def test_outcar_fixture_regenerates_byte_identically(outcar: Path) -> None:
    """The committed OUTCAR is exactly what its generator writes today."""
    assert outcar.read_text(encoding="utf-8") == vasp_fixtures.dfpt_outcar_text()


# --- the happy path, both codes --------------------------------------------


def test_vasp_reads_the_ionic_tensor(outcar: Path) -> None:
    """The OUTCAR block is returned as a 3x3 tensor, off-diagonals included."""
    eps_ion, eps_ion_tensor = dielectric.parse_ionic_dielectric(outcar)

    assert eps_ion == pytest.approx(EXPECTED_MEAN)
    assert eps_ion_tensor == pytest.approx(EXPECTED_TENSOR)


def test_castep_subtracts_optical_from_dc(castep_file: Path) -> None:
    """The CASTEP ionic tensor is the DC permittivity less the optical one."""
    eps_ion, eps_ion_tensor = dielectric.parse_ionic_dielectric(
        castep_file, code="castep"
    )

    assert eps_ion == pytest.approx(EXPECTED_MEAN)
    assert eps_ion_tensor == pytest.approx(EXPECTED_TENSOR)


def test_both_codes_agree(outcar: Path, castep_file: Path) -> None:
    """The two fixtures encode one tensor by two routes; they must still match."""
    _, vasp_tensor = dielectric.parse_ionic_dielectric(outcar)
    _, castep_tensor = dielectric.parse_ionic_dielectric(castep_file, code="castep")

    assert vasp_tensor == pytest.approx(castep_tensor)


@pytest.mark.parametrize("code", ["vasp", "castep"])
def test_scalar_is_the_diagonal_mean(
        outcar: Path, castep_file: Path, code: str
) -> None:
    """The scalar is the mean of the diagonal, not of the whole tensor.

    The fixture tensors have non-zero off-diagonal entries precisely so these
    two differ: the diagonal mean is 5.0 and the full mean is 2.0.
    """
    path = outcar if code == "vasp" else castep_file
    eps_ion, eps_ion_tensor = dielectric.parse_ionic_dielectric(path, code=code)

    assert eps_ion == pytest.approx(np.mean(eps_ion_tensor.diagonal()))
    assert eps_ion != pytest.approx(np.mean(eps_ion_tensor))


def test_castep_reads_optical_and_dc_separately(castep_file: Path) -> None:
    """The two halves of the table are read into the right tensors."""
    optical, dc = dielectric._read_castep_permittivity(castep_file)

    assert optical == pytest.approx(np.array(castep_fixtures.EFIELD_OPTICAL))
    assert dc == pytest.approx(np.array(castep_fixtures.EFIELD_DC))


# --- resolving the file -----------------------------------------------------


def test_vasp_accepts_a_directory(vasp_dfpt_dir: Path) -> None:
    """A directory is resolved to the OUTCAR inside it."""
    eps_ion, _ = dielectric.parse_ionic_dielectric(vasp_dfpt_dir)

    assert eps_ion == pytest.approx(EXPECTED_MEAN)


def test_castep_accepts_a_directory(castep_efield_dir: Path) -> None:
    """A directory with one .castep file needs no seedname."""
    eps_ion, _ = dielectric.parse_ionic_dielectric(castep_efield_dir, code="castep")

    assert eps_ion == pytest.approx(EXPECTED_MEAN)


def test_castep_seedname_selects_among_several(tmp_path: Path) -> None:
    """With several .castep files the seedname picks one; without it, an error."""
    (tmp_path / "wanted.castep").write_text(
        castep_fixtures.efield_castep_text(), encoding="utf-8"
    )
    (tmp_path / "other.castep").write_text(
        castep_fixtures.efield_castep_text(runs=3), encoding="utf-8"
    )

    eps_ion, _ = dielectric.parse_ionic_dielectric(
        tmp_path, code="castep", seedname="wanted"
    )
    assert eps_ion == pytest.approx(EXPECTED_MEAN)

    with pytest.raises(ValueError, match="pass seedname"):
        dielectric.parse_ionic_dielectric(tmp_path, code="castep")


def test_castep_missing_seed_names_the_path(tmp_path: Path) -> None:
    """A seedname with no matching file says which file it looked for."""
    with pytest.raises(FileNotFoundError, match="absent.castep"):
        dielectric.parse_ionic_dielectric(tmp_path, code="castep", seedname="absent")


def test_castep_empty_directory_raises(tmp_path: Path) -> None:
    """A directory with no .castep file raises rather than returning nothing."""
    with pytest.raises(FileNotFoundError, match=r"No \*.castep file"):
        dielectric.parse_ionic_dielectric(tmp_path, code="castep")


def test_gzipped_outcar_is_read(tmp_path: Path) -> None:
    """An archived calculation keeps its OUTCAR gzipped; it reads the same."""
    path = tmp_path / "OUTCAR.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(vasp_fixtures.dfpt_outcar_text())

    assert dielectric.parse_ionic_dielectric(path)[0] == pytest.approx(EXPECTED_MEAN)
    # ... and the directory resolves to it when no plain OUTCAR sits beside it.
    assert dielectric.parse_ionic_dielectric(tmp_path)[0] == pytest.approx(EXPECTED_MEAN)


def test_plain_outcar_wins_over_gzipped(tmp_path: Path) -> None:
    """With both present the uncompressed one is used, and is the newer copy."""
    with gzip.open(tmp_path / "OUTCAR.gz", "wt", encoding="utf-8") as handle:
        handle.write(vasp_fixtures.dfpt_outcar_text(runs=4))
    (tmp_path / "OUTCAR").write_text(
        vasp_fixtures.dfpt_outcar_text(), encoding="utf-8"
    )

    assert dielectric.parse_ionic_dielectric(tmp_path)[0] == pytest.approx(EXPECTED_MEAN)


def test_missing_outcar_names_the_plain_file(tmp_path: Path) -> None:
    """An empty directory reports OUTCAR, not the .gz fallback never found."""
    # The trailing quote is the point: "OUTCAR.gz'" would also contain "OUTCAR".
    with pytest.raises(FileNotFoundError, match="OUTCAR'"):
        dielectric.parse_ionic_dielectric(tmp_path)


@pytest.mark.parametrize("code", ["quantum_espresso", "VASP", ""])
def test_unknown_code_raises(outcar: Path, code: str) -> None:
    """An unrecognised code raises the package's canonical message."""
    with pytest.raises(ValueError, match="expected 'vasp' or 'castep'"):
        dielectric.parse_ionic_dielectric(outcar, code=code)


# --- output that holds no ionic response ------------------------------------


def test_vasp_without_the_block_raises(tmp_path: Path) -> None:
    """A valid OUTCAR that computed no ionic response raises, not returns zeros.

    This is what a hybrid run with LEPSILON produces: VASP does not implement
    DFPT for hybrids, so the calculation finishes and the block is simply
    never printed.
    """
    path = tmp_path / "OUTCAR"
    path.write_text(vasp_fixtures.dfpt_outcar_text(include_block=False), encoding="utf-8")

    with pytest.raises(ValueError, match="IBRION"):
        dielectric.parse_ionic_dielectric(path)


def test_castep_without_dc_columns_raises(tmp_path: Path) -> None:
    """An optical-only table cannot give the ionic part, and says so."""
    path = tmp_path / "toy.castep"
    path.write_text(
        castep_fixtures.efield_castep_text(include_dc=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="efield_calc_ion_permittivity"):
        dielectric.parse_ionic_dielectric(path, code="castep")


def test_castep_without_any_table_raises(tmp_path: Path) -> None:
    """A .castep file from some other task names the task that would work."""
    path = tmp_path / "toy.castep"
    path.write_text(" Final free energy (E-TS) =  -12.3456 eV\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Efield"):
        dielectric.parse_ionic_dielectric(path, code="castep")


def test_truncated_block_raises(tmp_path: Path) -> None:
    """A block cut short mid-tensor raises rather than returning a partial one."""
    full = vasp_fixtures.dfpt_outcar_text().splitlines()
    header = next(i for i, line in enumerate(full) if "IONIC CONTRIBUTION" in line)
    path = tmp_path / "OUTCAR"
    # Header, the rule of dashes, and one of the three rows.
    path.write_text("\n".join(full[: header + 3]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="found 1"):
        dielectric.parse_ionic_dielectric(path)


# --- restarted and continued runs -------------------------------------------


@pytest.mark.parametrize("code", ["vasp", "castep"])
def test_last_block_wins(tmp_path: Path, code: str) -> None:
    """A restarted run appends; the final block is the converged one.

    The generators offset every earlier block by a whole number, so reading
    the first block, or averaging them, gives a different answer.
    """
    if code == "vasp":
        path = tmp_path / "OUTCAR"
        path.write_text(vasp_fixtures.dfpt_outcar_text(runs=3), encoding="utf-8")
    else:
        path = tmp_path / "toy.castep"
        path.write_text(castep_fixtures.efield_castep_text(runs=3), encoding="utf-8")

    eps_ion, eps_ion_tensor = dielectric.parse_ionic_dielectric(path, code=code)

    assert eps_ion == pytest.approx(EXPECTED_MEAN)
    assert eps_ion_tensor == pytest.approx(EXPECTED_TENSOR)
