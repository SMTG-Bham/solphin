Python API
==========

Input generation
----------------

* :doc:`solphin.vasp_inputs` — VASP input sets from the packaged calculation
  recipes.
* :doc:`solphin.castep_inputs` — CASTEP ``.cell`` / ``.param`` input sets
  from the same recipes.

Parsing and analysis
--------------------

* :doc:`solphin.band_structure` — k-path generation, band-structure inputs,
  reconstruction and plotting.
* :doc:`solphin.dos` — DOS calculation setup and density-of-states effective
  masses.
* :doc:`solphin.optics` — absorption, refractive index, SLME and the Blank
  thickness sweep from dielectric data.
* :doc:`solphin.spectral` — photon-flux-weighted spectral average and
  dispersion of the absorption coefficient.

Figures of merit
----------------

* :doc:`solphin.db_fom` — detailed-balance (Shockley-Queisser) limit
  efficiency.
* :doc:`solphin.pv_fom` — components of the :math:`\Gamma_{\mathrm{PV}}`
  photovoltaic figure of merit.

Plotting and results
--------------------

* :doc:`solphin.db_plots` — plots of detailed-balance quantities: photon
  flux, J-V curves and efficiency limits.
* :doc:`solphin.final_results` — combines the detailed-balance limit and
  :math:`\Gamma_{\mathrm{PV}}` into final efficiency results.

.. toctree::
   :maxdepth: 1
   :hidden:

   solphin.vasp_inputs
   solphin.castep_inputs
   solphin.band_structure
   solphin.dos
   solphin.optics
   solphin.spectral
   solphin.db_fom
   solphin.pv_fom
   solphin.db_plots
   solphin.final_results
