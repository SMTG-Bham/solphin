"""Generators for the synthetic VASP fixtures under tests/data/vasp_toy.

The ``Cu2GeS3`` tree beside this one is real output from a real calculation.
This tree is the other kind: small hand-built files with a known answer
written into them, for readers whose input would otherwise have to be a
multi-hundred-megabyte OUTCAR to exercise nine lines of parsing.

* ``DFPT/OUTCAR`` holds the ``MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC
  CONTRIBUTION`` block a run with ionic displacements prints, and nothing else
  of substance.

As with ``castep_fixtures.py``, the committed files are regenerated
byte-identically by these functions - tests assert that - so all floats are
written with fixed-width formats and the ``__main__`` guard rewrites the tree
in place.

Note that this OUTCAR is deliberately *not* parseable by pymatgen's ``Outcar``,
whose constructor reads the whole file and requires NBANDS, NPLWV and other
fields unrelated to the dielectric response. ``solphin.dielectric`` reads the
block directly for that reason, so the fixture only has to contain the block.
"""

from pathlib import Path

# The ionic contribution the fixture encodes. Deliberately the same tensor as
# ``EFIELD_DC - EFIELD_OPTICAL`` in castep_fixtures.py, so both readers are
# anchored to one number; test_dielectric.py asserts they still agree. The
# diagonal mean is exactly 5.0, and the off-diagonal entries are non-zero so a
# reader that returned only the diagonal would be caught.
IONIC_TENSOR = (
    (4.0, 1.0, 0.0),
    (1.0, 5.0, 0.5),
    (0.0, 0.5, 6.0),
)


def dfpt_outcar_text(include_block: bool = True, runs: int = 1) -> str:
    """Build the OUTCAR fixture carrying the ionic dielectric tensor block.

    Reproduces the block VASP prints for ``IBRION=6|8`` with ``LEPSILON`` or
    ``LCALCEPS``: the header, a rule of dashes, three rows of three numbers,
    and a closing rule. The piezoelectric block that follows it in a real
    OUTCAR is included too, so the reader has to stop at the end of the
    tensor it was asked for rather than running on into the next one.

    Parameters
    ----------
    include_block : bool, optional
        Whether to print the ionic dielectric block at all. False reproduces
        a run that computed no ionic response - a plain static calculation,
        or a hybrid run where ``LEPSILON`` was silently inapplicable.
        Default is True.
    runs : int, optional
        How many times to repeat the block, reproducing a restarted run that
        appended to an existing OUTCAR. Earlier blocks are offset so only the
        last one gives the expected answer. Default is 1.

    Returns
    -------
    str
        Complete OUTCAR file contents.
    """
    rule = " " + "-" * 76
    lines = [
        " vasp.6.4.2 24Jul24 (build Jan 01 2025 00:00:00) complex",
        "",
        " POSCAR: Si  Si",
        "   number of dos      NEDOS =    301   number of ions     NIONS =      2",
        "",
        "  energy  without entropy=      -10.123456  energy(sigma->0) =      -10.123456",
        "",
    ]

    for run in range(runs):
        if not include_block:
            break
        # Earlier blocks are shifted by a whole number, so taking the first
        # block, or averaging the blocks, gives a different answer.
        offset = float(runs - 1 - run)

        lines.append(" MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION")
        lines.append(rule)
        for row in IONIC_TENSOR:
            lines.append("".join(f"{value + offset:13.6f}" for value in row))
        lines.append(rule)
        lines.append("")

        # The piezoelectric tensor VASP prints straight after, in different
        # units and with six columns: the reader must not continue into it.
        lines.append(" PIEZOELECTRIC TENSOR IONIC CONTRIBUTION  for field in x, y, z (C/m^2)")
        lines.append(rule)
        for axis in "xyz":
            values = [0.0] * 6
            lines.append(f"  {axis}" + "".join(f"{value:13.6f}" for value in values))
        lines.append(rule)
        lines.append("")

    lines.extend(
        [
            " General timing and accounting informations for this job:",
            " ========================================================",
            "",
            "                  Total CPU time used (sec):       45.678",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_all(root: Path) -> None:
    """Write every fixture file into the vasp_toy tree.

    Parameters
    ----------
    root : Path
        Directory that becomes ``vasp_toy``; subdirectories are created.
    """
    targets = {
        root / "DFPT" / "OUTCAR": dfpt_outcar_text,
    }
    for path, build in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build(), encoding="utf-8")


if __name__ == "__main__":
    generate_all(Path(__file__).resolve().parent / "data" / "vasp_toy")
