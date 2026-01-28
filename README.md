# PV-FoM

**Please note that this code is still under development and not currently suitable for use**

`solphin` is a code developed to assist with the characterisation of novel photovoltaic materials with `VASP`. It combines detailed balance analysis with other tools such as the photovoltaic figure of merit proposed by A.Crovetto, Spectroscopic Limited Maximum Efficiency (SLME), Blank et al Maximum Efficiency and Optical tauc plots to provide a full picture of a materials theoretical photovoltaic efficiency. 

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



