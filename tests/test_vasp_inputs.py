"""VASP input generation.

Most of the logic here is pure dictionary assembly and needs no pseudopotentials,
so it is tested unconditionally. Only the final write_input call needs POTCARs,
which are licensed and machine-local, so that one test is gated.
"""

from pathlib import Path

import pytest
from pymatgen.core import Structure

import solphin.vasp_inputs as vasp_inputs
from conftest import requires_potcars
from solphin.castep_inputs import write_castep_calculation
from solphin.vasp_inputs import RecipeConfig

FUNCTIONALS = ["LDA", "PBEsol", "PBE", "HSE06", "PBE0", "R2SCAN"]


@pytest.fixture
def config() -> RecipeConfig:
    """A fresh config per test - _prepare_incar mutates what it is given."""
    return vasp_inputs._load_config("base_recipes.json")


# --- the packaged recipe file ---------------------------------------------


def test_load_config_structure(config: RecipeConfig) -> None:
    """The packaged recipe has the four top-level keys and covers every functional."""
    assert set(config) == {"INCAR", "PATCHES", "POTCAR", "POTCAR_FUNCTIONAL"}
    assert set(FUNCTIONALS) <= set(config["INCAR"])
    assert config["POTCAR_FUNCTIONAL"] == "PBE_64"


def test_load_config_resolves_through_package_resources(config: RecipeConfig) -> None:
    """It must work from an installed wheel, not just a source checkout.

    _load_config goes through importlib.resources, and pyproject declares the
    json under package-data; this fails loudly if either is dropped.
    """
    assert config["PATCHES"]["optics"]["LOPTICS"] is True


@pytest.mark.parametrize("patch", ["relax_cell", "tight_relax", "optics", "eff_mass"])
def test_recipe_patches_exist(config: RecipeConfig, patch: str) -> None:
    """Every patch name the reference workflow uses exists in the packaged recipes."""
    assert patch in config["PATCHES"]


# --- POTCAR functional selection ------------------------------------------


def test_determine_potcar_functional_prefers_explicit_argument(config: RecipeConfig) -> None:
    """An explicitly requested POTCAR functional wins over recipe and config."""
    chosen = vasp_inputs._determine_potcar_functional("HSE06", "PBE_52", config)

    assert chosen == "PBE_52"


def test_determine_potcar_functional_lda_special_case(config: RecipeConfig) -> None:
    """LDA maps to the LDA_64 POTCAR set regardless of the config default."""
    assert vasp_inputs._determine_potcar_functional("LDA", None, config) == "LDA_64"


def test_determine_potcar_functional_falls_back_to_config(config: RecipeConfig) -> None:
    """Without an explicit choice or special case, the config default applies."""
    chosen = vasp_inputs._determine_potcar_functional("HSE06", None, config)

    assert chosen == config["POTCAR_FUNCTIONAL"]


# --- INCAR assembly --------------------------------------------------------


def test_prepare_incar_applies_patches(config: RecipeConfig) -> None:
    """The optics patch lands its LOPTICS, CSHIFT and NEDOS tags in the INCAR."""
    incar = vasp_inputs._prepare_incar("PBE", ["optics"], config)

    assert incar["LOPTICS"] is True
    assert "CSHIFT" in incar
    assert "NEDOS" in incar


@pytest.mark.parametrize("recipe", ["HSE06", "PBE0"])
def test_prepare_incar_hybrid_adds_ncore(config: RecipeConfig, recipe: str) -> None:
    """Hybrid recipes get NCORE = 4 added to their INCAR."""
    assert vasp_inputs._prepare_incar(recipe, [], config)["NCORE"] == 4


def test_prepare_incar_defect_patch_expands(config: RecipeConfig) -> None:
    """'defect' is not a patch in the json - it stands for two that are."""
    incar = vasp_inputs._prepare_incar("PBE", ["defect"], config)

    for key, value in config["PATCHES"]["relax_atoms"].items():
        assert incar[key] == value
    for key, value in config["PATCHES"]["spin_polarised"].items():
        assert incar[key] == value


def test_prepare_incar_gamma_only_is_not_an_incar_patch(config: RecipeConfig) -> None:
    """gamma_only changes k-points, not the INCAR, so it is skipped here."""
    plain = dict(vasp_inputs._prepare_incar("PBE", [], config))
    fresh = vasp_inputs._load_config("base_recipes.json")
    with_gamma = dict(vasp_inputs._prepare_incar("PBE", ["gamma_only"], fresh))

    assert plain == with_gamma


# --- van der Waals tags ----------------------------------------------------


@pytest.mark.parametrize(
    "patch,recipe,expected_ivdw",
    [
        ("vdw_d3_bj", "PBE", 12),
        ("vdw_d3_bj", "HSE06", 12),
        ("vdw_d3", "PBE", 11),
        ("vdw_d3", "HSE06", 11),
        ("vdw_d4", "PBE", 13),
        ("vdw_d4", "HSE06", 13),
    ],
)
def test_prepare_vdw_tags_per_scheme(patch: str, recipe: str, expected_ivdw: int) -> None:
    """Each vdW patch maps to its IVDW code for both GGA and hybrid recipes."""
    tags = vasp_inputs._prepare_vdw_tags(recipe, [patch])

    assert tags["IVDW"] == expected_ivdw


def test_prepare_vdw_tags_hse06_carries_refitted_parameters() -> None:
    """HSE06 needs its own damping parameters; PBE uses the VASP defaults."""
    hse = vasp_inputs._prepare_vdw_tags("HSE06", ["vdw_d3_bj"])
    pbe = vasp_inputs._prepare_vdw_tags("PBE", ["vdw_d3_bj"])

    assert {"VDW_S8", "VDW_A1", "VDW_A2"} <= set(hse)
    assert set(pbe) == {"IVDW"}


def test_prepare_vdw_tags_rvv10_only_for_r2scan() -> None:
    """rVV10 tags apply to R2SCAN only; other recipes get nothing."""
    assert vasp_inputs._prepare_vdw_tags("R2SCAN", ["rvv10"])["LUSE_VDW"] is True
    assert vasp_inputs._prepare_vdw_tags("PBE", ["rvv10"]) == {}


def test_prepare_vdw_tags_empty_without_patch() -> None:
    """Non-vdW patches produce no vdW tags."""
    assert vasp_inputs._prepare_vdw_tags("HSE06", ["optics"]) == {}


# --- structures ------------------------------------------------------------


@pytest.mark.parametrize("filename", ["POSCAR", "CONTCAR"])
def test_read_structure_pmg(relax_dir: Path, filename: str) -> None:
    """POSCAR and CONTCAR both load as the 12-atom Cu2GeS3 structure."""
    structure = vasp_inputs.read_structure_pmg(relax_dir / filename)

    assert isinstance(structure, Structure)
    assert structure.composition.reduced_formula == "Cu2GeS3"
    assert len(structure) == 12


def test_read_structure_pmg_reads_castep_cell(relax_dir: Path, tmp_path: Path) -> None:
    """A .cell written by write_castep_calculation reads back as the same structure.

    pymatgen's Structure.from_file cannot sniff the CASTEP format, so this
    pins the .cell extension routing through sumo.
    """
    structure = vasp_inputs.read_structure_pmg(relax_dir / "POSCAR")
    cell_file = write_castep_calculation(structure, "PBE", tmp_path)

    read_back = vasp_inputs.read_structure_pmg(cell_file)

    assert isinstance(read_back, Structure)
    assert read_back.composition.reduced_formula == "Cu2GeS3"
    assert len(read_back) == len(structure)


# --- the write path --------------------------------------------------------


def test_relax_cell_scales_encut(relax_dir: Path, config: RecipeConfig) -> None:
    """relax_cell raises ENCUT by 30 %, which is how the committed INCAR got 585.

    Checked against tests/data/Cu2GeS3/Relax/INCAR, which the reference
    workflow produced from ENCUT = 450 with this patch applied.
    """
    structure = vasp_inputs.read_structure_pmg(relax_dir / "POSCAR")
    incar = vasp_inputs._prepare_incar("HSE06", ["relax_cell"], config)
    vasp_set = vasp_inputs._create_vasp_set(
        structure,
        incar,
        "PBE_64",
        config,
        user_incar_settings={"KSPACING": 0.2, "ENCUT": 450},
    )

    vasp_inputs._apply_patches(vasp_set, ["relax_cell"], "HSE06", incar)

    assert vasp_set.user_incar_settings["ENCUT"] == pytest.approx(585.0)


@requires_potcars
def test_write_vasp_calculation_writes_inputs(relax_dir: Path, tmp_path: Path) -> None:
    """The write path produces the input files with the patched ENCUT and no KPOINTS."""
    structure = vasp_inputs.read_structure_pmg(relax_dir / "POSCAR")

    vasp_inputs.write_vasp_calculation(
        structure=structure,
        recipe="HSE06",
        out_dir=tmp_path,
        patches=["relax_cell", "tight_relax"],
        user_incar_settings={"KSPACING": 0.2, "ENCUT": 450},
    )

    written = {p.name for p in tmp_path.iterdir()}
    incar_text = (tmp_path / "INCAR").read_text()

    assert {"INCAR", "POSCAR", "POTCAR"} <= written
    # KSPACING replaces the KPOINTS file, so pymatgen writes no KPOINTS here.
    assert "KSPACING" in incar_text
    assert "KPOINTS" not in written
    assert "ENCUT = 585" in incar_text


# --- defects ---------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
            "_prepare_incar mutates config['INCAR'][recipe] in place, so a second "
            "call with the same config inherits the first call's patches; latent "
            "only because write_vasp_calculation re-reads the json every time"
    ),
)
def test_prepare_incar_does_not_mutate_config(config: RecipeConfig) -> None:
    """_prepare_incar should not leak one call's patches into the next."""
    vasp_inputs._prepare_incar("PBE", ["optics"], config)

    unpatched = vasp_inputs._prepare_incar("PBE", [], config)

    assert "LOPTICS" not in unpatched


@pytest.mark.xfail(
    strict=True,
    reason=(
            'vasp_inputs.py:271 reads `... or "vdw_d4"`, a bare truthy string, so '
            "the vdW branch runs for every calculation; latent because "
            "_prepare_vdw_tags returns {} when no vdW patch was asked for"
    ),
)
def test_vdw_branch_skipped_without_vdw_patch(
        relax_dir: Path, config: RecipeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The vdW branch should not run when no vdW patch was requested."""
    structure = vasp_inputs.read_structure_pmg(relax_dir / "POSCAR")
    incar = vasp_inputs._prepare_incar("HSE06", ["optics"], config)
    vasp_set = vasp_inputs._create_vasp_set(
        structure, incar, "PBE_64", config, user_incar_settings={"KSPACING": 0.2}
    )
    called = []
    monkeypatch.setattr(
        vasp_inputs,
        "_prepare_vdw_tags",
        lambda recipe, patches: called.append(patches) or {},
    )

    vasp_inputs._apply_patches(vasp_set, ["optics"], "HSE06", incar)

    assert called == []
