# Solphin

***

<img width="1600" height="480" alt="solphin" src="https://github.com/user-attachments/assets/04549cb0-a768-4322-9a17-173756d84d75" />

***

[![test](https://github.com/SMTG-Bham/solphin/actions/workflows/test.yml/badge.svg)](https://github.com/SMTG-Bham/solphin/actions/workflows/test.yml)
[![docs](https://github.com/SMTG-Bham/solphin/actions/workflows/docs.yml/badge.svg)](https://github.com/SMTG-Bham/solphin/actions/workflows/docs.yml)
[![package](https://github.com/SMTG-Bham/solphin/actions/workflows/package.yml/badge.svg)](https://github.com/SMTG-Bham/solphin/actions/workflows/package.yml)
[![codecov](https://codecov.io/gh/SMTG-Bham/solphin/branch/main/graph/badge.svg)](https://codecov.io/gh/SMTG-Bham/solphin)

`solphin` is a code developed to assist with the characterisation of novel photovoltaic materials with `VASP`. It
combines detailed balance analysis with other tools such as the photovoltaic figure of merit proposed by A. Crovetto,
Spectroscopic Limited Maximum Efficiency (SLME), Blank et. al. Maximum Efficiency and optical absorption plots to
provide a full picture of a material's theoretical photovoltaic efficiency. The code supports the automatic generation
of `VASP` input files for the required calculations based on an initial crystal structure.

**Please note that Solphin is still in early stage testing and development**

<img width="1076" height="596" alt="solphin_1 drawio" src="https://github.com/user-attachments/assets/0f981e7a-fcf3-4ec0-a2ef-01980a3f56ee" />

## Installation

`solphin` is not yet published to PyPI, so install it from a checkout:

```bash
git clone https://github.com/SMTG-Bham/solphin
cd solphin
pip install -e .
```

`plot_db_combined_interactive` and `plot_final_result_interactive` drive
`%matplotlib widget`, which needs the `ipympl` backend:

```bash
pip install -e ".[interactive]"
```

For a full conda development environment — tutorial, docs and dev tooling
included — see [environment.yml](environment.yml). Note that `sumo` has to be
installed from PyPI rather than conda-forge there; the reason is documented at
the top of that file.

If using the `VASP` input file generation functionality, please ensure that your `VASP` pseudopotentials are added to
your path through the use of a `pymatgen` configuration file `$HOME/.pmgrc.yaml`. The file should contain:

```
PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>
```

## Citation

If you use `solphin` in your work, please cite the following:

* Cox, P. U., Russell, P. P., Crovetto, A., Squires, A. G., & Scanlon, D. O.
  Solphin [Computer software]. https://github.com/SMTG-Bham/solphin
* Crovetto, A., 2024. A phenomenological figure of merit for photovoltaic materials. Journal of Physics: Energy, 6 (2),
  p.025009.
* Alex M. Ganose, Adam J. Jackson, David O. Scanlon. sumo: Command-line tools for plotting and analysis of periodic ab
  initio calculations. Journal of Open Source Software, 2018 3 (28), 717, doi:10.21105/joss.00717.

## Acknowledgements

The developers Philippa U. Cox and Peter P. Russell would like to thank Alexander G. Squires, Andrea Crovetto and David
O. Scanlon for their guidance on this project, Brooke Busbee for her work on the branding and Jacob Baggott for his
assistance.
