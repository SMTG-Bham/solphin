"""CASTEP input generation.

CASTEP uses on-the-fly pseudopotentials, so unlike the VASP twin of this file
nothing here needs licensed data: the full write path runs unconditionally.
The tests mirror test_vasp_inputs.py section for section, including the
shared contract that _prepare_param must NOT mutate its config.
"""

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from sumo.io.castep import CastepCell

import solphin.castep_inputs as castep_inputs
from solphin.castep_inputs import CastepRecipeConfig

FUNCTIONALS = ["LDA", "PBEsol", "PBE", "HSE06", "PBE0", "R2SCAN"]


@pytest.fixture
def config() -> CastepRecipeConfig:
    """A fresh config per test, so no test depends on another's leftovers."""
    return castep_inputs._load_config("castep_recipes.json")


@pytest.fixture
def fe_structure() -> Structure:
    """Bcc iron: two Fe sites, both listed in the recipes' SPINS map."""
    return Structure(
        Lattice.cubic(2.87), ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )


# --- the packaged recipe file ----------------------------------------------


def test_load_config_structure(config: CastepRecipeConfig) -> None:
    """The packaged recipe has the five top-level keys and covers every functional."""
    assert set(config) == {"PARAM", "CELL", "CELL_DEFAULTS", "SPINS", "PATCHES"}
    assert set(FUNCTIONALS) <= set(config["PARAM"])


def test_load_config_resolves_through_package_resources(config: CastepRecipeConfig) -> None:
    """It must work from an installed wheel, not just a source checkout.

    _load_config goes through importlib.resources, and pyproject declares the
    json under package-data; this fails loudly if either is dropped.
    """
    assert config["PATCHES"]["optics"]["param"]["spectral_task"] == "Optics"


@pytest.mark.parametrize("patch", ["relax_cell", "tight_relax", "optics", "eff_mass"])
def test_recipe_patches_exist(config: CastepRecipeConfig, patch: str) -> None:
    """Every patch name the reference workflow uses exists in the packaged recipes."""
    assert patch in config["PATCHES"]


def test_r2scan_recipe_maps_to_rscan(config: CastepRecipeConfig) -> None:
    """The R2SCAN recipe name selects CASTEP's RSCAN functional."""
    assert config["PARAM"]["R2SCAN"]["xc_functional"] == "RSCAN"


@pytest.mark.parametrize("recipe", ["HSE06", "PBE0"])
def test_hybrid_recipes_select_norm_conserving_potentials(
        config: CastepRecipeConfig, recipe: str
) -> None:
    """Hybrid functionals need norm-conserving pseudopotentials in CASTEP."""
    assert config["CELL"][recipe]["species_pot"] == "NCP19"


# --- patch expansion and validation ----------------------------------------


def test_expand_patches_defect_expands(config: CastepRecipeConfig) -> None:
    """'defect' is not a patch in the json - it stands for two that are."""
    expanded = castep_inputs._expand_patches(["defect"], config)

    assert expanded == ["relax_atoms", "spin_polarised"]


@pytest.mark.parametrize("patch", ["elastic_tensor", "rvv10", "deformation_potential", "lobster"])
def test_expand_patches_rejects_vasp_only_patches(
        config: CastepRecipeConfig, patch: str
) -> None:
    """VASP-only patches are rejected with a message naming the patch."""
    with pytest.raises(ValueError, match=f"{patch}.*VASP-only"):
        castep_inputs._expand_patches([patch], config)


def test_expand_patches_rejects_unknown_patch(config: CastepRecipeConfig) -> None:
    """An unknown patch name raises rather than being silently dropped."""
    with pytest.raises(ValueError, match="banana"):
        castep_inputs._expand_patches(["banana"], config)


# --- .param assembly --------------------------------------------------------


def test_prepare_param_applies_patches(config: CastepRecipeConfig) -> None:
    """The optics patch lands the spectral task tags in the .param settings."""
    param = castep_inputs._prepare_param("PBE", ["optics"], config)

    assert param["task"] == "Spectral"
    assert param["spectral_task"] == "Optics"


def test_prepare_param_defect_patches_apply(config: CastepRecipeConfig) -> None:
    """The expanded defect patches land relaxation and spin settings together."""
    patches = castep_inputs._expand_patches(["defect"], config)
    param = castep_inputs._prepare_param("PBE", patches, config)

    assert param["task"] == "GeometryOptimization"
    assert param["spin_polarized"] is True


def test_prepare_param_unknown_recipe_raises(config: CastepRecipeConfig) -> None:
    """An unknown recipe raises naming the supported ones."""
    with pytest.raises(ValueError, match="B3LYP"):
        castep_inputs._prepare_param("B3LYP", [], config)


def test_prepare_param_does_not_mutate_config(config: CastepRecipeConfig) -> None:
    """One call's patches must not leak into the next.

    The VASP twin is test_prepare_incar_does_not_mutate_config; both sides
    deep-copy the recipe before layering patches on with update().
    """
    castep_inputs._prepare_param("PBE", ["optics"], config)

    unpatched = castep_inputs._prepare_param("PBE", [], config)

    assert "spectral_task" not in unpatched


@pytest.mark.parametrize(
    "patch,expected_scheme",
    [("vdw_d3", "D3"), ("vdw_d3_bj", "D3-BJ"), ("vdw_d4", "D4")],
)
def test_prepare_param_vdw_schemes(
        config: CastepRecipeConfig, patch: str, expected_scheme: str
) -> None:
    """Each vdW patch maps to its SEDC dispersion scheme."""
    param = castep_inputs._prepare_param("PBE", [patch], config)

    assert param["sedc_apply"] is True
    assert param["sedc_scheme"] == expected_scheme


def test_prepare_param_gamma_only_is_not_a_param_patch(config: CastepRecipeConfig) -> None:
    """gamma_only changes the k-point tags in the cell, not the .param."""
    plain = castep_inputs._prepare_param("PBE", [], config)
    with_gamma = castep_inputs._prepare_param("PBE", ["gamma_only"], config)

    assert plain == with_gamma


# --- .cell tag assembly -----------------------------------------------------


def test_prepare_cell_tags_defaults(config: CastepRecipeConfig) -> None:
    """Without patches the defaults carry symmetry and a k-point spacing."""
    tags = castep_inputs._prepare_cell_tags("PBE", [], config)

    assert tags["symmetry_generate"] == "true"
    assert "kpoint_mp_spacing" in tags


def test_prepare_cell_tags_grid_and_spacing_are_exclusive(config: CastepRecipeConfig) -> None:
    """Supplying both k-point tags raises: CASTEP treats them as alternatives."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        castep_inputs._prepare_cell_tags(
            "PBE", [], config, kpoint_mp_grid=(2, 2, 2), kpoint_mp_spacing=0.05
        )


def test_prepare_cell_tags_grid_drops_default_spacing(config: CastepRecipeConfig) -> None:
    """An explicit grid removes the default spacing rather than shipping both."""
    tags = castep_inputs._prepare_cell_tags("PBE", [], config, kpoint_mp_grid=(3, 3, 3))

    assert tags["kpoint_mp_grid"] == "3 3 3"
    assert "kpoint_mp_spacing" not in tags


def test_prepare_cell_tags_explicit_spacing(config: CastepRecipeConfig) -> None:
    """An explicit spacing replaces the recipe default."""
    tags = castep_inputs._prepare_cell_tags("PBE", [], config, kpoint_mp_spacing=0.05)

    assert tags["kpoint_mp_spacing"] == "0.05"
    assert "kpoint_mp_grid" not in tags


def test_prepare_cell_tags_gamma_only_patch_drops_default_spacing(
        config: CastepRecipeConfig,
) -> None:
    """The gamma_only patch sets a 1 1 1 grid and removes the spacing default."""
    tags = castep_inputs._prepare_cell_tags("PBE", ["gamma_only"], config)

    assert tags["kpoint_mp_grid"] == "1 1 1"
    assert "kpoint_mp_spacing" not in tags


def test_prepare_cell_tags_relax_atoms_fixes_cell(config: CastepRecipeConfig) -> None:
    """relax_atoms keeps the cell fixed while the geometry optimisation runs."""
    tags = castep_inputs._prepare_cell_tags("PBE", ["relax_atoms"], config)

    assert tags["fix_all_cell"] == "true"


# --- initial spins ----------------------------------------------------------


def test_apply_spins_annotates_positions(
        config: CastepRecipeConfig, fe_structure: Structure
) -> None:
    """Fe sites gain a spin=... annotation and the total moment is returned."""
    cell = CastepCell.from_structure(fe_structure)

    total = castep_inputs._apply_spins(fe_structure, cell, config)

    rows = cell.blocks["positions_frac"].values
    assert all(row[-1] == "spin=5" for row in rows)
    assert total == pytest.approx(10.0)


def test_apply_spins_skips_unlisted_elements(config: CastepRecipeConfig) -> None:
    """Elements without a SPINS entry are left unannotated."""
    structure = Structure(Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
    cell = CastepCell.from_structure(structure)

    total = castep_inputs._apply_spins(structure, cell, config)

    assert total == 0.0
    assert len(cell.blocks["positions_frac"].values[0]) == 4  # element + 3 coords


# --- the write path ---------------------------------------------------------


def test_write_castep_calculation_writes_inputs(relax_dir: Path, tmp_path: Path) -> None:
    """The write path produces a .cell/.param pair named after the formula."""
    structure = Structure.from_file(relax_dir / "POSCAR")

    cell_file = castep_inputs.write_castep_calculation(
        structure=structure,
        recipe="PBE",
        out_dir=tmp_path,
        patches=["relax_cell"],
    )

    assert cell_file == tmp_path / "Cu2GeS3.cell"
    param_text = (tmp_path / "Cu2GeS3.param").read_text()
    cell_text = cell_file.read_text()

    assert "task" in param_text and "GeometryOptimization" in param_text
    assert "xc_functional" in param_text and "PBE" in param_text
    assert "cut_off_energy" in param_text
    assert "%block lattice_cart" in cell_text
    assert "%block positions_frac" in cell_text
    assert "kpoint_mp_spacing" in cell_text


def test_write_castep_calculation_seedname_override(
        fe_structure: Structure, tmp_path: Path
) -> None:
    """An explicit seedname names both files."""
    cell_file = castep_inputs.write_castep_calculation(
        fe_structure, "PBE", tmp_path, seedname="iron"
    )

    assert cell_file.name == "iron.cell"
    assert (tmp_path / "iron.param").is_file()


def test_write_castep_calculation_hybrid_writes_species_pot_block(
        fe_structure: Structure, tmp_path: Path
) -> None:
    """Hybrid recipes carry their pseudopotential library as a species_pot block."""
    cell_file = castep_inputs.write_castep_calculation(fe_structure, "HSE06", tmp_path)

    cell_text = cell_file.read_text()
    assert "%block species_pot" in cell_text
    assert "NCP19" in cell_text


def test_write_castep_calculation_spin_patch(
        fe_structure: Structure, tmp_path: Path
) -> None:
    """The spin_polarised patch annotates sites and sets the total spin."""
    cell_file = castep_inputs.write_castep_calculation(
        fe_structure, "PBE", tmp_path, patches=["spin_polarised"]
    )

    cell_text = cell_file.read_text()
    param_text = (tmp_path / "Fe.param").read_text()
    assert cell_file.name == "Fe.cell"
    assert "spin=5" in cell_text
    assert "spin_polarized" in param_text
    assert "spin                    : 10.0" in param_text


def test_write_castep_calculation_user_settings_take_precedence(
        fe_structure: Structure, tmp_path: Path
) -> None:
    """User .param and .cell settings override the recipe values."""
    castep_inputs.write_castep_calculation(
        fe_structure,
        "PBE",
        tmp_path,
        user_param_settings={"cut_off_energy": 700},
        user_cell_settings={"kpoint_mp_grid": "4 4 4"},
    )

    param_text = (tmp_path / "Fe.param").read_text()
    cell_text = (tmp_path / "Fe.cell").read_text()
    assert "700" in param_text
    assert "kpoint_mp_grid" in cell_text and "4 4 4" in cell_text
    assert "kpoint_mp_spacing" not in cell_text


def test_write_castep_calculation_user_blocks(
        fe_structure: Structure, tmp_path: Path
) -> None:
    """Extra cell blocks, e.g. a spectral k-point list, are written verbatim."""
    rows = [["0.0", "0.0", "0.0"], ["0.5", "0.0", "0.0"]]

    cell_file = castep_inputs.write_castep_calculation(
        fe_structure, "PBE", tmp_path, user_cell_blocks={"spectral_kpoint_list": rows}
    )

    cell_text = cell_file.read_text()
    assert "%block spectral_kpoint_list" in cell_text
    assert "0.5 0.0 0.0" in cell_text


def test_written_cell_round_trips_the_structure(
        relax_dir: Path, tmp_path: Path
) -> None:
    """The written .cell reads back as the same structure."""
    structure = Structure.from_file(relax_dir / "POSCAR")

    cell_file = castep_inputs.write_castep_calculation(structure, "PBE", tmp_path)

    read_back = CastepCell.from_file(str(cell_file)).structure
    assert read_back.composition.reduced_formula == "Cu2GeS3"
    assert len(read_back) == len(structure)
    assert read_back.lattice.matrix == pytest.approx(structure.lattice.matrix, abs=1e-8)
