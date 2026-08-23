"""Shared fixtures for the solphin test suite.

The suite's fixture data is the ``Cu2GeS3`` VASP reference calculation set
committed at ``tests/data/Cu2GeS3`` - produced by the tutorial workflow once,
but owned by the tests: the suite runs the same with ``tutorial/`` deleted.
Most fixtures here simply hand tests a path into a copy of that tree or a
cached parse of one of its files.

Two rules the fixtures exist to enforce:

* Tests never touch the tracked tree. They work from a read-only copy of the
  files named in ``_DATA_MANIFEST``, so the tracked tree is opened once a
  session, for reading, by the fixture that makes that copy. The copy is
  ~132 MB under pytest's temp root, of which the last three sessions' worth
  are kept.
* Parsing is cached at session scope. Each parse is fast on its own, but
  ``compute_dos`` re-reads its vasprun up to ten times internally whenever the
  fit-quality check trips, which it does on the reference data.
"""

import shutil
from pathlib import Path

import matplotlib

# Must precede the solphin import: solphin/__init__.py eagerly imports all nine
# modules, several of which pull in pyplot.
matplotlib.use("Agg")

from collections.abc import Iterator

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from numpy.typing import NDArray
from pymatgen.core import SETTINGS  # noqa: E402
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine

import solphin.band_structure as band_structure  # noqa: E402
import solphin.db_fom as db_fom  # noqa: E402
import solphin.dos as dos  # noqa: E402
import solphin.optics as optics  # noqa: E402
from solphin.dos import DOSResult

REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_SOURCE = Path(__file__).resolve().parent / "data" / "Cu2GeS3"

# Everything the suite reads out of the committed calculation set, and nothing
# else. The copy below is built from this list rather than by copying the tree,
# so the suite's data dependency is written down rather than implied: a test
# that reaches for a file not named here fails with FileNotFoundError against
# the copy, which is the signal to add it.
#
# What it leaves out is ~150 KB of a 132 MB tree - the eight vasprun.xml files
# are the rest of it - so this is documentation, not an optimisation.
_DATA_MANIFEST = (
    "Relax/POSCAR",
    "Relax/CONTCAR",
    "OPT_hybrid/vasprun.xml",
    "OPT_hybrid/POSCAR",
    "OPT_hybrid/INCAR",
    "OPT_hybrid/absorption.dat",
    "OPT_hybrid/n_real.dat",
    "DOS_HDFT/vasprun.xml",
    # KPOINTS is not decoration: BSVasprun.get_band_structure(line_mode=True)
    # defaults kpoints_filename to the KPOINTS beside the vasprun and raises
    # VaspParseError without it. Names are matched literally throughout - the
    # vasprun glob in get_band_structure would not see a gzipped fixture - so
    # nothing here may be compressed.
    *(
        f"BAND_SP_HDFT/split-{n:02d}/{name}"
        for n in range(1, 8)
        for name in ("vasprun.xml", "KPOINTS")
    ),
)


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
def reference_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Read-only copy of the committed Cu2GeS3 reference calculation set.

    The suite works from a copy so that the tracked tree is read once, here, and
    never handed to anything that could write to it. The copied files have their
    write bits stripped, so an in-place write fails where the bug is rather than
    silently corrupting the input of a later test; the directories stay writable
    so that pytest can still clean the temp tree up.

    The data lives inside tests/, so there is no optional-data skip: a source
    tree that is absent, or missing one of the manifest's files, is a broken
    checkout, and raises FileNotFoundError naming what could not be found.
    """
    if not _DATA_SOURCE.is_dir():
        raise FileNotFoundError(f"test data missing at {_DATA_SOURCE} - broken checkout?")

    dest = tmp_path_factory.mktemp("Cu2GeS3")
    for relative in _DATA_MANIFEST:
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # copyfile rather than copy/copy2: the destination is created fresh
        # under the umask, so the source's mode is not carried over.
        shutil.copyfile(_DATA_SOURCE / relative, target)
        target.chmod(0o444)
    return dest


@pytest.fixture(scope="session")
def opt_dir(reference_data: Path) -> Path:
    """Optics/dielectric calculation - the source of absorption.dat and n_real.dat."""
    return reference_data / "OPT_hybrid"


@pytest.fixture(scope="session")
def dos_vasprun(reference_data: Path) -> Path:
    """Static HSE06 run used for the DOS effective mass."""
    return reference_data / "DOS_HDFT" / "vasprun.xml"


@pytest.fixture(scope="session")
def band_dir(reference_data: Path) -> Path:
    """Directory holding the seven split-NN band-structure calculations."""
    return reference_data / "BAND_SP_HDFT"


@pytest.fixture(scope="session")
def relax_dir(reference_data: Path) -> Path:
    """Geometry optimisation - holds the POSCAR and the relaxed CONTCAR."""
    return reference_data / "Relax"


# --- cached parses ---------------------------------------------------------


@pytest.fixture(scope="session")
def am15() -> NDArray:
    """AM1.5G as shipped: column 0 wavelength (nm), column 1 irradiance (W m^-2 nm^-1)."""
    return db_fom.load_spectrum("AM1.5")


@pytest.fixture(scope="session")
def photon_spectrum(am15: NDArray) -> NDArray:
    """AM1.5G converted: column 0 energy (eV), column 1 photon flux (m^-2 s^-1 eV^-1)."""
    return db_fom.convert_spectrum(am15)


@pytest.fixture(scope="session")
def dielectric(opt_dir: Path) -> tuple[float, NDArray, NDArray, NDArray, NDArray]:
    """The 5-tuple from calc_dielectric: (eps_inf, tensor, eps_full, eps_imag, energies)."""
    return optics.calc_dielectric(filename=str(opt_dir / "vasprun.xml"))


@pytest.fixture(scope="session")
def dos_result(dos_vasprun: Path) -> DOSResult:
    """DOSResult for electrons at the 0.1 eV window the reference mass is quoted at."""
    return dos.compute_dos(
        dos_vasprun=str(dos_vasprun), carrier="electrons", energy_window=0.1
    )


@pytest.fixture(scope="session")
def band_structure_obj(band_dir: Path) -> BandStructureSymmLine:
    """Recombined BandStructureSymmLine across the committed splits.

    Seven is the number of ``split-NN`` folders actually committed. The value
    is ignored today - see the ``test_splits_argument_is_honoured`` xfail - but
    if that defect is ever fixed, 7 is the number that keeps this fixture
    reading all of them.
    """
    return band_structure.get_band_structure(str(band_dir), 7)


# --- writable copies -------------------------------------------------------


@pytest.fixture
def tmp_opt_dir(opt_dir: Path, tmp_path: Path) -> Path:
    """Writable copy of the optics calculation, without the generated .dat files.

    Write tests run here so that regenerating absorption.dat / n_real.dat can be
    compared against the originals it was copied from without touching them.
    """
    dest = tmp_path / "OPT_hybrid"
    dest.mkdir()
    for name in ("vasprun.xml", "POSCAR", "INCAR"):
        # copyfile, not copy: opt_dir is the read-only session copy, and copy
        # would carry mode 0o444 into what has to be a working directory.
        shutil.copyfile(opt_dir / name, dest / name)
    return dest


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Keep the Agg canvas from accumulating across the plotting tests."""
    yield
    plt.close("all")
