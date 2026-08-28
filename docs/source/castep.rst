Using solphin with CASTEP
=========================

Every VASP-facing capability in ``solphin`` has a CASTEP counterpart. The
reader functions take a ``code`` keyword — ``code="vasp"`` is the default
everywhere, so existing VASP workflows are unchanged — and input generation
has a dedicated module, :mod:`solphin.castep_inputs`. The physics between
those boundaries (detailed balance, the figure of merit, SLME and the
plotting) is identical for both codes.

This page is the reference for what maps to what. For the workflow walked
end to end — generating the input sets, running OptaDOS, then reading the
results back — see the :doc:`CASTEP tutorial <castep_workflow_tutorial>`.

What each function reads
------------------------

CASTEP splits its outputs over seed-named files where VASP concentrates them
in ``vasprun.xml``:

.. list-table::
   :header-rows: 1

   * - Quantity
     - VASP source
     - CASTEP source
   * - Dielectric function (optics)
     - ``vasprun.xml``
     - OptaDOS ``<seed>_epsilon.dat``
   * - Density of states
     - ``vasprun.xml``
     - ``<seed>.bands`` (histogrammed by sumo)
   * - Band structure
     - ``vasprun.xml`` (+ ``KPOINTS``)
     - ``<seed>.bands`` (+ sibling ``.cell`` for labels)
   * - Restart data for band runs
     - ``CHGCAR`` / ``IBZKPT``
     - ``<seed>.check`` (via the ``reuse`` tag)
   * - Crystal structure
     - ``POSCAR`` / ``CONTCAR``
     - ``<seed>.cell``

Directory-based functions (``generate_absorption``, ``get_band_structure``,
…) discover the seed file by globbing — one match is required. If a
directory holds several calculations, pass ``seedname=`` (optics) or point
the function at a directory containing a single calculation.

Generating inputs
-----------------

:func:`solphin.castep_inputs.write_castep_calculation` mirrors
``write_vasp_calculation``: the same recipe names (``LDA``, ``PBEsol``,
``PBE``, ``HSE06``, ``PBE0``, ``R2SCAN``) and the same patch vocabulary,
emitted as a ``<seed>.cell`` / ``<seed>.param`` pair. CASTEP generates
on-the-fly pseudopotentials, so no POTCAR equivalent is needed — the hybrid
recipes select the norm-conserving ``NCP19`` library, which hybrid
functionals require in CASTEP.

Patch mapping:

.. list-table::
   :header-rows: 1

   * - Patch
     - CASTEP meaning
   * - ``relax_cell`` / ``relax_atoms`` / ``tight_relax``
     - Geometry optimisation (``relax_atoms`` fixes the cell with
       ``fix_all_cell``)
   * - ``dfpt``
     - ``task: Efield`` (DFPT dielectric permittivity)
   * - ``dos`` / ``gaussian_dos``
     - ``task: Spectral`` with ``spectral_task: DOS`` and PDOS weights
   * - ``optics`` / ``smeared_optics``
     - ``task: Spectral`` with ``spectral_task: Optics`` (broadening is an
       OptaDOS setting, not a CASTEP one)
   * - ``eff_mass``
     - ``task: Spectral`` with ``spectral_task: BandStructure``
   * - ``spin_polarised``
     - ``spin_polarized: true`` plus per-site ``spin=`` annotations
   * - ``vdw_d3`` / ``vdw_d3_bj`` / ``vdw_d4``
     - SEDC dispersion (``sedc_scheme: D3`` / ``D3-BJ`` / ``D4``; D4 needs
       CASTEP 24 or newer)
   * - ``ncl``
     - ``spin_treatment: Vector`` with spin-orbit coupling
   * - ``gamma_only``
     - ``kpoint_mp_grid: 1 1 1``
   * - ``elastic_tensor``, ``rvv10``, ``deformation_potential``, ``lobster``
     - VASP-only; requesting them raises ``ValueError``

Recipe caveats worth knowing:

* ``R2SCAN`` maps to CASTEP's ``RSCAN`` functional — the closest supported
  meta-GGA, not a bit-for-bit equivalent.
* ``elec_energy_tol`` is per atom in CASTEP, where VASP's ``EDIFF`` is a
  total energy; the recipes use ``1e-7`` eV/atom to mirror ``EDIFF = 1e-6``.
* ``kpoint_mp_spacing`` is in Å\ :sup:`-1` without VASP's 2π factor: the
  default ``0.03`` corresponds to ``KSPACING ≈ 0.19``.

Band structures
---------------

The k-path generation is shared. For CASTEP the calculation setup builds on
a converged SCF run instead of copying a charge density:

.. code-block:: python

   import matplotlib.pyplot as plt
   import solphin

   structure = solphin.vasp_inputs.read_structure_pmg("scf/Si.cell")
   canonical, kpath = solphin.band_structure.generate_band_structure_path(structure)

   solphin.band_structure.write_castep_band_structure_calculation(
       "scf/Si.cell", kpath, "band", splits=2
   )

   # ... run CASTEP in each split-NN folder ...

   bs = solphin.band_structure.get_band_structure("band", 2, code="castep")
   solphin.band_structure.plot_band_structure(bs, plt)

The setup appends the path to a copy of the SCF ``.cell`` as a
``spectral_kpoint_list``, switches the copied ``.param`` to
``task: Spectral`` / ``spectral_task: BandStructure``, and copies the
``.check`` with ``reuse`` set — CASTEP's analogue of VASP's
``ICHARG = 11`` restart. High-symmetry labels are read back from the
``.cell`` beside each ``.bands`` file.

Density of states and the effective mass
----------------------------------------

``compute_dos``, ``get_dos_effective_mass`` and ``plot_dos`` accept the
``.bands`` file where they take a ``vasprun.xml`` for VASP:

.. code-block:: python

   result = solphin.dos.compute_dos("dos/Si.bands", code="castep")

CASTEP's DOS is histogrammed from the eigenvalues, so the ``bin_width``
keyword (default 0.01 eV) plays the role VASP's ``NEDOS`` does: a denser
spectral k-point mesh and a bin width small against the fitting window give
the √E fit clean data. The valence-band-maximum referencing assumes a gapped
material — for metals the CASTEP effective-mass path is not meaningful.

Optics through OptaDOS
----------------------

CASTEP's spectral task writes optical matrix elements (``.ome_bin``); the
dielectric function itself comes from `OptaDOS
<https://github.com/optados-developers/optados>`_. Run the ``optics`` task
(the ``optics`` patch sets it up), then OptaDOS with ``task : optics``, and
hand the resulting ``<seed>_epsilon.dat`` to solphin:

.. code-block:: python

   solphin.optics.generate_absorption("optics", code="castep")
   solphin.optics.generate_n_real("optics", code="castep")

Both the default polycrystalline geometry and ``optics_geom : tensor`` are
understood; the polycrystalline spectrum is treated as an isotropic tensor.
Broadening choices live in the OptaDOS ``.odi`` file. The static dielectric
constant ``eps_inf`` is taken from the lowest-energy row, matching the VASP
convention. Downstream of the generated ``absorption.dat`` and
``n_real.dat``, everything — spectral averages, SLME, the Blank models, the
figure of merit — is code-agnostic and needs no ``code`` argument.

OptaDOS is a Fortran post-processing tool distributed alongside CASTEP; it
is not a Python dependency of ``solphin``, and nothing in ``solphin`` needs
a CASTEP installation.
