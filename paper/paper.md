---
title: "Solphin: Photovoltaic efficiency analysis for bulk materials using Python"
tags:
  - Python
  - photovoltaics
  - materials screening
  - density functional theory
  - VASP
authors:
  - name: Philippa U Cox
    affiliation: "1"
  - name: Peter P Russell
    affiliation: "1"
  - name: Andrea Crovetto
    orcid: 0000-0003-1499-8740
    affiliation: "2"
  - name: Alexander G Squires
    orcid: 0000-0001-6967-3690
    affiliation: "1"
  - name: David O Scanlon
    orcid: 0000-0001-9174-8601
    affiliation: "1"
affiliations:
  - index: 1
    name: School of Chemistry, University of Birmingham, Edgbaston, Birmingham, B15 2TT, UK
  - index: 2
    name: DTU Nanotech, Technical University of Denmark, DK-2800 Kgs, Lyngby, Denmark
date: 24 August 2026
bibliography: paper.bib
---

# Summary

Photovoltaic materials convert light into electricity and are the active
components of solar cells. Solar electricity is increasingly important as a
cheap and sustainable route to reducing carbon emissions
[@malik2025parameter; @kirchartz2025state]. Although high conversion
efficiencies have been achieved, many leading inorganic absorbers contain
toxic or expensive elements. Finding alternatives therefore requires methods
that evaluate efficiency alongside the environmental and economic constraints
that ultimately govern whether a material can be used.

`Solphin` is an open-source Python package for characterising the photovoltaic
potential of inorganic bulk materials. It connects the preparation and
post-processing of density functional theory calculations with detailed
balance analysis, thin-film efficiency metrics, optical and electronic
properties, and the photovoltaic figure of merit developed by Crovetto
[@crovetto2024fom]. A user can begin with a crystal structure and follow the
complete workflow, or supply properties obtained from independent calculations
or experiments and run only the relevant analysis. This flexibility supports
both computational screening and comparison with experimentally characterised
materials.

# Statement of need

Computational screening can remove unsuitable candidates before expensive
laboratory work, accelerating photovoltaic materials discovery. In practice,
however, assessing a new bulk absorber requires several calculations,
post-processing tools, and efficiency models. The necessary inputs and the
reported theoretical efficiencies vary between studies, making results hard to
reproduce and compare directly.

`Solphin` provides a guided workflow for computational materials researchers
and experimentalists who want a consistent assessment from whatever bulk
properties are available. Its principal capabilities are:

- generation of Vienna *Ab initio* Simulation Package (VASP) inputs for the
  solid-state calculations used to obtain photovoltaic properties
  [@hafner1997vasp];
- calculation of Crovetto's figure of merit, which combines optical, transport,
  and defect-related bulk properties [@crovetto2024fom];
- parameter sweeps showing how doping density, non-radiative recombination
  lifetime, and carrier mobility affect predicted efficiency;
- post-processing and plotting for publication-ready outputs; and
- partial workflows with minimal input requirements, while accepting further
  theoretical or experimental properties when they are available.

# State of the field

Existing photovoltaic Python tools address complementary levels of the
problem. `pvlib` models solar-energy systems [@holmgren2018pvlib], `OTSun`
performs optical ray tracing for devices with arbitrary geometry
[@cardona2020otsun], and `PVAnalytics` processes system-level time-series data
[@perry2022pvanalytics]. `Pypvcell` provides composable solar-cell models
[@lee2017pypvcell], while `Solcore` spans optical and electrical device
simulation [@alonsoalvarez2018solcore]. These packages are valuable once a
device or operating system has been specified, but they do not provide the same
guided route from a bulk crystal structure and VASP outputs to a combined set
of absorber-level efficiency indicators.

`Solphin` therefore targets a different abstraction rather than extending one
of the device packages. It reuses `pymatgen` for materials data and VASP input
infrastructure [@ong2013pymatgen], and adapts `sumo` functionality for
electronic-structure analysis and plotting [@ganose2018sumo]. The package adds
the orchestration, photovoltaic metrics, and shared data model needed to make
those calculations part of one reproducible bulk-material screening workflow.

# Software design

The workflow is organised as independent stages connected by explicit files
and material properties (Figure \ref{fig:workflow}). `Solphin` prepares VASP
inputs but does not run the licensed VASP executable. This separation keeps
resource scheduling and licensed pseudopotentials under the user's control,
while versioned calculation recipes make the intended electronic-structure
workflow reproducible.

![The basic workflow, with VASP calculations in teal, primary outputs in purple,
and secondary outputs in blue. []{label="fig:workflow"}](../docs/source/_static/solphin_workflow.drawio.png){width="100%"}

Input generation supports LDA, the PBE and PBEsol generalised-gradient
approximations, the R2SCAN meta-GGA, and the HSE06 and PBE0 hybrid functionals.
Recipes may include dispersion corrections and user-supplied VASP flags. The
initial VASP-specific design trades calculator breadth for a constrained,
documented path through relaxation, band-structure, density-of-states, and
optical calculations. Manual property inputs provide an escape hatch when a
different code or an experiment supplies part of the workflow.

The analysis modules produce four groups of results:

- Shockley-Queisser detailed-balance curves and limiting efficiencies;
- spectroscopic limited maximum efficiency (SLME) and the Blank thin-film
  selection metric [@yu2012slme; @blank2017selection];
- band structure, density of states, band gap, density-of-states effective
  mass, absorption onset, and static dielectric response; and
- spectral average, spectral dispersion, and the photovoltaic figure of merit
  [@crovetto2024fom].

The modular design lets users stop after any group or combine the outputs in a
final efficiency summary. The detailed-balance implementation was adapted from
the pedagogical implementations by Kaklin and Byrnes
[@kaklin2024sqlimit; @byrnes2024efficiencylimits]. Numerical and plotting work
uses NumPy, SciPy, and Matplotlib
[@harris2020numpy; @virtanen2020scipy; @hunter2007matplotlib]. Reusing these
specialised libraries reduces duplicated algorithms while keeping the
photovoltaic workflow accessible through one Python interface.

# Research impact statement

`Solphin` is an early release, so its impact claim is based on research
readiness and near-term significance rather than broad external adoption. A
complete, executed tutorial demonstrates the workflow using committed
Cu~2~GeS~3~ VASP reference data. The automated tests check analytic and
published limits, including the AM1.5G irradiance integral and the
Shockley-Queisser maximum, and regenerate optical properties from the reference
VASP output. Continuous integration tests the supported Python versions and
builds both the documentation and distributable package.

These materials give researchers a reproducible starting point for comparing
candidate absorbers with several established metrics rather than a single
headline efficiency. They also make the intermediate quantities inspectable,
which helps identify whether optical absorption, transport, recombination, or
band gap is limiting a candidate. Future work will broaden input generation and
output parsing beyond VASP to codes such as CASTEP, CP2K, and Quantum ESPRESSO,
and will connect calculated carrier lifetimes, mobilities, and dopant densities
from specialised packages in place of parameter sweeps.

# Author contributions

P.U.C. developed the concept, researched and implemented the mathematical and
physics functionality, debugged and tested the software, and wrote and
documented the methodologies. P.P.R. contributed software development,
standardisation, visualisation, package management, and documentation. A.C.
developed the photovoltaic figure of merit and the initial implementation of
that metric. A.G.S. supervised the project, advised on software-development
practice, and provided the initial calculation-input code. D.O.S. supervised
the project and managed resources and funding.

# Conflicts of interest

The authors declare no conflicts of interest.

# AI usage disclosure

Claude (Sonnet 4.6) assisted with debugging some functions and, only for the
density-of-states effective-mass module, suggested approaches for implementing
the equations. ChatGPT (5.3 mini) generated docstrings from an author-supplied
template. The authors extensively reviewed, validated, and edited that
AI-assisted work; the tools did not generate the underlying physics or make
software design and architecture decisions. AI was not used to write the
original manuscript.

OpenAI Codex (GPT-5) was subsequently used on 24 August 2026 to reconstruct the
missing Markdown and BibLaTeX sources from an author-provided PDF, verify
bibliographic metadata, and reorganise the manuscript for the current JOSS
section requirements. The reconstructed sources must be reviewed and validated
by the authors before submission.

# Acknowledgements

The authors thank Brooke Busbee for creating the `Solphin` branding and Seán R.
Kavanagh for providing CdTe data. P.U.C., P.P.R., A.G.S., and D.O.S.
acknowledge the University of Birmingham's
[BlueBEAR HPC service](https://www.birmingham.ac.uk/bear). The authors are
grateful to Jacob Baggott for valuable discussions and to all members of the
Scanlon Materials Theory Group for their feedback.

# References
