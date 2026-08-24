# solphin tests

Run them with a bare `pytest` from the repository root, as `CONTRIBUTING.md`
describes:

```bash
pytest
```

The suite needs no network access, no VASP, no CASTEP, no OptaDOS, and nothing
outside `tests/` and the installed package. Its data has two families under
`tests/data`:

* `Cu2GeS3` — the VASP reference calculation set (once produced by the
  tutorial workflow, but owned by the tests — deleting `tutorial/` does not
  change a single result).
* `castep_toy` — a synthetic CASTEP set: small hand-built `.bands`, `.cell`
  and OptaDOS `_epsilon.dat` files whose physics is closed-form, so every
  CASTEP reader anchors to an analytic expectation. `tests/castep_fixtures.py`
  documents each file's construction and regenerates it byte-identically;
  per-file regeneration tests pin that, the same discipline as the `.dat`
  regeneration below.

The tracked data is never worked on in place: `conftest.py` copies the files
the suite reads — the `_DATA_MANIFEST` list — into pytest's temp area once per
session and strips their write bits, so every fixture hands out a path into a
read-only copy and a stray write fails with `PermissionError` at the point of
the bug (`test_fixture_copy_is_write_protected` in `test_optics.py` pins this).
A test that exercises a write path gets a writable `tmp_path` copy on top of
that, from the `tmp_opt_dir` fixture.

Two tests need VASP pseudopotentials (`PMG_VASP_PSP_DIR`) and skip themselves
when those are unavailable, which is the normal case away from a machine that
has a licensed POTCAR directory.

## What is being checked

Each module is tested on its own, against analytic limits and published values
where one exists — as `CONTRIBUTING.md` asks — and against the committed
reference data where the property is "this input produces this number". The
main anchors:

| Anchor | Where |
|---|---|
| AM1.5G integrates to 1000.4 W m⁻² | `test_db_fom.py` |
| Shockley–Queisser limit: 33.7 % at 1.34 eV | `test_db_fom.py` |
| `q·Voc < E_gap`; `J(Voc) = 0` | `test_db_fom.py` |
| Power is conserved across the nm → eV change of variable | `test_db_fom.py` |
| A constant α gives back exactly α, with exactly zero dispersion | `test_spectral.py` |
| A lossless dielectric has `n = √ε` and zero absorption | `test_optics.py` |
| The parabolic-band DOS relation, run backwards to recover a chosen mass | `test_dos.py` |
| `absorption.dat` / `n_real.dat` regenerate from `vasprun.xml` | `test_optics.py` |
| A Lorentz-oscillator `_epsilon.dat` parses back to the analytic ε(E), with ε(0) = 6 | `test_optics.py` |
| The OptaDOS tensor geometry gives `n = √ε` per axis and zero absorption end to end | `test_optics.py` |
| A parabolic-DOS `.bands` file returns the electron and hole masses it encodes | `test_dos.py` |
| The cosine-band `.bands` fixture has a 1.5 eV direct gap at Γ | `test_band_structure.py` |
| Every `castep_toy` file regenerates byte-identically from `castep_fixtures.py` | per consumer file |

The two `.dat` regeneration tests are worth calling out: the committed files are
reproducible from the committed `vasprun.xml`, which makes them verified
reference data rather than a snapshot of whatever the code emits today.

`test_import_side_effects.py` checks a different kind of property: that
`import solphin` leaves the root logger, matplotlib's `font_manager` logger and
the warnings filters as it found them. Seven modules used to reconfigure those
at module scope, which `solphin/__init__.py`'s eager imports made unavoidable
for anyone importing the package. Each check runs in a subprocess, because
`conftest.py` imports solphin at collection time and pytest's own logging plugin
attaches a handler to the root logger for the duration of every test — in-process
the property is unobservable either way.

## Known defects (`xfail`)

Six tests state behaviour the code does not yet have. They are all
`xfail(strict=True)`, so **fixing the source turns the suite red** and tells you
to drop the marker — the register cannot silently rot.

`pytest -rx` prints the whole list on any run.

| Test | File | Defect |
|---|---|---|
| `test_write_local_kpoints_writes_file` | `test_dos.py` | `dos.py:1453` calls `len()`/`.tolist()` on a pymatgen `Kpoints` |
| `test_write_band_structure_missing_scf_raises` | `test_band_structure.py` | `band_structure.py:276` prints an error and returns `None` instead of raising |
| `test_splits_argument_is_honoured` | `test_band_structure.py` | `get_band_structure` globs every split, ignoring the `splits` value (the CASTEP branch deliberately mirrors this, so one fix covers both) |
| `test_prepare_incar_does_not_mutate_config` | `test_vasp_inputs.py` | `_prepare_incar` mutates the config dict it is handed (the CASTEP `_prepare_param` deep-copies; its non-mutation test is green) |
| `test_vdw_branch_skipped_without_vdw_patch` | `test_vasp_inputs.py` | `vasp_inputs.py:271` — `or "vdw_d4"` is always truthy |
| `test_mobility_plot_draws_one_line_per_lifetime` | `test_plots.py` | `final_results.py:439` appends a 3-tuple instead of one element |

## Not covered, and why

* **The tutorial notebook.** It is documentation, deliberately outside the
  suite: nothing here executes it, reads it, or pins its recorded numbers, so
  the full-pipeline composition of the library calls is exercised only by the
  notebook itself. The reverse dependency does exist — the notebook's analysis
  half reads `tests/data/Cu2GeS3` — so the data stays put when the notebook
  moves or goes.
* **The `*_interactive` functions.** `plot_db_combined_interactive` and
  `plot_final_result_interactive` set `fig.canvas.header_visible`, an
  ipympl-only attribute. ipympl is now declared — `pyproject.toml`'s
  `interactive` extra and `environment.yml` both carry it — but the functions
  need a live widget backend, so they stay outside the suite.

## Observations that are not tests

Real defects that no assertion can currently demonstrate, recorded here rather
than as `xfail`:

* `db_fom.py:195` — the ideal-diode term reads `Jph - J0·exp(qV/kT) - 1` where
  it should be `Jph - J0·(exp(qV/kT) - 1)`. Photon fluxes are ~1e21 m⁻² s⁻¹, so
  the stray unit term is lost below float64 precision.
* `optics._bb_per_eV` uses a `2/(h³c²)` prefactor while `db_fom._rr0` uses
  `2π/(c²h_eV³)` — the same physical quantity under two conventions, differing
  by exactly π.
* `db_fom.max_power` computes an unused `index`; `max_eff` documents "%" but
  returns a fraction; `band_structure._is_soc_vasprun` is never called.
