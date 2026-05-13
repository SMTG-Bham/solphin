# Solphin
<img width="1600" height="1600" alt="1000059871 (1)" src="https://github.com/user-attachments/assets/d494c8a9-6e87-43b3-ba6d-26726cc4d6e1" />

`solphin` is a code developed to assist with the characterisation of novel photovoltaic materials with `VASP`. It combines detailed balance analysis with other tools such as the photovoltaic figure of merit proposed by A.Crovetto, Spectroscopic Limited Maximum Efficiency (SLME), Blank et al Maximum Efficiency and Optical absorption plots to provide a full picture of a materials theoretical photovoltaic efficiency. The full workflow allows the generation of VASP calculations from structure input files to generate the required files for all the calculations. 

<img width="1076" height="596" alt="solphin_1 drawio" src="https://github.com/user-attachments/assets/0f981e7a-fcf3-4ec0-a2ef-01980a3f56ee" />

*We request that any use of `solphin` in your work cites the code and theory papers*

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



