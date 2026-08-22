"""Shared fixtures for the solphin test suite.

The tutorial notebook and the ``Cu2GeS3`` VASP output committed beside it are the
only executable definition of what the package is meant to do, so most fixtures
here simply hand tests a path into ``tutorial/Cu2GeS3`` or a cached parse of one
of its files.

Two rules the fixtures exist to enforce:

* Nothing writes into ``tutorial/``. The tutorial's own cells overwrite files
  that are tracked in git; tests that exercise a write path get a ``tmp_path``
  copy instead.
* Parsing is cached at session scope. Each parse is fast on its own, but
  ``compute_dos`` re-reads its vasprun up to ten times internally whenever the
  fit-quality check trips, which it does on the tutorial data.
"""

import shutil
from pathlib import Path

import matplotlib

# Must precede the solphin import: solphin/__init__.py eagerly imports all nine
# modules, several of which pull in pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from pymatgen.core import SETTINGS  # noqa: E402

import solphin.band_structure as band_structure  # noqa: E402
import solphin.db_fom as db_fom  # noqa: E402
import solphin.dos as dos  # noqa: E402
import solphin.optics as optics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def _potcars_available() -> bool:
    """True when pymatgen can find a VASP pseudopotential directory.

    POTCARs are licensed and cannot be redistributed, so they are present on a
    developer machine and absent everywhere else.
    """
    psp_dir = SETTINGS.get("PMG_VASP_PSP_DIR")
    return bool(psp_dir) and Path(psp_dir).is_dir()


requires_potcars = pytest.mark.skipif(
    not _potcars_available(),
    reason="needs VASP pseudopotentials (PMG_VASP_PSP_DIR)",
)


# --- locations -------------------------------------------------------------
# Resolved from __file__ rather than the cwd so the suite runs from anywhere.


@pytest.fixture(scope="session")
def tutorial_data() -> Path:
    """Root of the committed Cu2GeS3 calculation set."""
    path = REPO_ROOT / "tutorial" / "Cu2GeS3"
    if not path.is_dir():
        pytest.skip(f"tutorial data not found at {path}")
    return path


@pytest.fixture(scope="session")
def opt_dir(tutorial_data) -> Path:
    """Optics/dielectric calculation - the source of absorption.dat and n_real.dat."""
    return tutorial_data / "OPT_hybrid"


@pytest.fixture(scope="session")
def dos_vasprun(tutorial_data) -> Path:
    """Static HSE06 run used for the DOS effective mass."""
    return tutorial_data / "DOS_HDFT" / "vasprun.xml"


@pytest.fixture(scope="session")
def band_dir(tutorial_data) -> Path:
    """Directory holding the seven split-NN band-structure calculations."""
    return tutorial_data / "BAND_SP_HDFT"


@pytest.fixture(scope="session")
def relax_dir(tutorial_data) -> Path:
    """Geometry optimisation - holds the POSCAR and the relaxed CONTCAR."""
    return tutorial_data / "Relax"


# --- cached parses ---------------------------------------------------------


@pytest.fixture(scope="session")
def am15():
    """AM1.5G as shipped: column 0 wavelength (nm), column 1 irradiance (W m^-2 nm^-1)."""
    return db_fom.load_spectrum("AM1.5")


@pytest.fixture(scope="session")
def photon_spectrum(am15):
    """AM1.5G converted: column 0 energy (eV), column 1 photon flux (m^-2 s^-1 eV^-1)."""
    return db_fom.convert_spectrum(am15)


@pytest.fixture(scope="session")
def dielectric(opt_dir):
    """The 5-tuple from calc_dielectric: (eps_inf, tensor, eps_full, eps_imag, energies)."""
    return optics.calc_dielectric(filename=str(opt_dir / "vasprun.xml"))


@pytest.fixture(scope="session")
def dos_result(dos_vasprun):
    """DOSResult for electrons at the tutorial's 0.1 eV window."""
    return dos.compute_dos(
        dos_vasprun=str(dos_vasprun), carrier="electrons", energy_window=0.1
    )


@pytest.fixture(scope="session")
def band_structure_obj(band_dir):
    """Recombined BandStructureSymmLine across the committed splits."""
    return band_structure.get_band_structure(str(band_dir), 5)


# --- writable copies -------------------------------------------------------


@pytest.fixture
def tmp_opt_dir(opt_dir, tmp_path) -> Path:
    """Copy of the optics calculation, without the generated .dat files.

    Write tests run here so that regenerating absorption.dat / n_real.dat can be
    compared against the committed originals without touching them.
    """
    dest = tmp_path / "OPT_hybrid"
    dest.mkdir()
    for name in ("vasprun.xml", "POSCAR", "INCAR"):
        shutil.copy(opt_dir / name, dest / name)
    return dest


@pytest.fixture(autouse=True)
def close_figures():
    """Keep the Agg canvas from accumulating across the plotting tests."""
    yield
    plt.close("all")
