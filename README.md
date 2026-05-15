# Solphin

***

<img width="1600" height="480" alt="solphin" src="https://github.com/user-attachments/assets/04549cb0-a768-4322-9a17-173756d84d75" />

***

`solphin` is a code developed to assist with the characterisation of novel photovoltaic materials with `VASP`. It combines detailed balance analysis with other tools such as the photovoltaic figure of merit proposed by A. Crovetto, Spectroscopic Limited Maximum Efficiency (SLME), Blank et. al. Maximum Efficiency and optical absorption plots to provide a full picture of a material's theoretical photovoltaic efficiency. The code supports the automatic generation of `VASP` input files for the required calculations based on an initial crystal structure.

<img width="1076" height="596" alt="solphin_1 drawio" src="https://github.com/user-attachments/assets/0f981e7a-fcf3-4ec0-a2ef-01980a3f56ee" />

## Installation 

`solphin` is currently only installable through the following method:

```
git clone https://<PATOKEN>@github.com/SMTG-Bham/PV-FoM
```

Navigate into `solphin` and run:

```
pip install -e.
```

If using the `VASP` input file generation functionality, please ensure that your `VASP` pseudopotentials are added to your path through the use of a `pymatgen` configuration file `$HOME/.pmgrc.yaml`. The file should contain:

```
PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>
```

## Citation
If you use `solphin` in your work, please cite the following:
* Philippa U. Cox, Peter P. Russell, Andrea Crovetto, Alexander G. Squires, David O. Scanlon. Solphin, 2026, https://github.com/SMTG-Bham/solphin/
* Crovetto, A., 2024. A phenomenological figure of merit for photovoltaic materials. Journal of Physics: Energy, 6(2), p.025009.
* Alex M. Ganose, Adam J. Jackson, David O. Scanlon. sumo: Command-line tools for plotting and analysis of periodic ab initio calculations. Journal of Open Source Software, 2018 3 (28), 717, doi:10.21105/joss.00717.

## Acknowledgements

The developers Philippa U. Cox and Peter P. Russell would like to thank Alexander G. Squires, Andrea Crovetto and David O. Scanlon for their guidance on this project, Brooke Busbee for her work on the branding and Jacob Baggott for his assistance. 
