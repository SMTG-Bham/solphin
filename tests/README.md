# solphin tests

Run them with a bare `pytest` from the repository root, as `CONTRIBUTING.md`
describes:

```bash
pytest
```

The suite needs no network access and no VASP. It reads the committed
`tutorial/Cu2GeS3` calculation set and never writes to it — every test that
exercises a write path gets a `tmp_path` copy from the `tmp_opt_dir` fixture.

Two tests need VASP pseudopotentials (`PMG_VASP_PSP_DIR`) and skip themselves
when those are unavailable, which is the normal case away from a machine that
has a licensed POTCAR directory.

## What is being checked

The tutorial notebook is the spine. Its recorded output —

```
SQ limit 33.51 %   FOM relative to SQ 80.28 %   FOM total efficiency 26.90 %
```

— is reproduced end to end in `test_tutorial_workflow.py`, so the notebook and
the code stay pinned to each other in both directions.

Everywhere else the preference is for analytic limits and published values over
pinned output, as `CONTRIBUTING.md` asks. The main anchors:

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

Ten tests state behaviour the code does not yet have. They are all
`xfail(strict=True)`, so **fixing the source turns the suite red** and tells you
to drop the marker — the register cannot silently rot.

`pytest -rx` prints the whole list on any run.

| Test | File | Defect |
|---|---|---|
| `test_recomb_rate_returns_finite_float` | `test_db_fom.py` | `db_fom.py:177` calls `_rr0` without `Tcell` |
| `test_v_at_mpp_between_zero_and_voc` | `test_db_fom.py` | `db_fom.py:248` calls `voc` without `Tcell` |
| `test_j_at_mpp_below_jsc` | `test_db_fom.py` | `db_fom.py:269` calls `max_power` without `Tcell` |
| `test_fill_factor_between_zero_and_one` | `test_db_fom.py` | `db_fom.py:326` missing `Tcell`, and `:331` divides by the tuple `(j_sc, v_oc)` |
| `test_write_local_kpoints_writes_file` | `test_dos.py` | `dos.py:1283` calls `len()`/`.tolist()` on a pymatgen `Kpoints` |
| `test_write_band_structure_missing_scf_raises` | `test_band_structure.py` | `band_structure.py:212` prints an error and returns `None` instead of raising |
| `test_splits_argument_is_honoured` | `test_band_structure.py` | `get_band_structure` globs every split, ignoring the `splits` value |
| `test_prepare_incar_does_not_mutate_config` | `test_vasp_inputs.py` | `_prepare_incar` mutates the config dict it is handed |
| `test_vdw_branch_skipped_without_vdw_patch` | `test_vasp_inputs.py` | `vasp_inputs.py:246` — `or "vdw_d4"` is always truthy |
| `test_mobility_plot_draws_one_line_per_lifetime` | `test_plots.py` | `final_results.py:439` appends a 3-tuple instead of one element |

The first four are public API that appears in the Sphinx docs but is unreachable
from the tutorial, which is why nothing has ever exercised it.

## Not covered, and why

* **Executing the notebook.** No longer blocked on syntax — cell 27's PEP 701
  nested-quote f-string was rewritten to use inner single quotes, so the
  notebook parses on the pinned 3.11. Executing it end to end is still not
  wired into the suite; `test_tutorial_workflow.py` reproduces its recorded
  numbers through the library instead.
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
