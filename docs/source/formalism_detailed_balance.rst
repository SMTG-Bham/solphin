Detailed Balance and the Shockley-Queisser Limit
================================================

:mod:`solphin.db_fom` implements the Shockley-Queisser (SQ) construction
[Shockley1961]_: the efficiency ceiling of a single-junction cell whose only
loss channel is the radiative emission it is obliged to produce by detailed
balance. Every quantity on this page is a function of the band gap
:math:`E_{\mathrm{g}}`, the incident spectrum and the cell temperature
:math:`T` — and of nothing else about the material.

Putting the spectrum in photon-flux form
----------------------------------------

Reference spectra are tabulated as spectral irradiance
:math:`I_\lambda(\lambda)` in W m\ :sup:`-2` nm\ :sup:`-1`.
:func:`~solphin.db_fom.convert_spectrum` re-expresses that as a photon flux
per unit photon energy, which is the form every integral below wants:

.. math::

   \phi(E) \;=\; I_\lambda(\lambda)\,
                 \underbrace{\frac{\mathrm{d}\lambda}{\mathrm{d}E}}_{hc/E^{2}}\,
                 \frac{1}{E}
           \;=\; \frac{hc}{E^{3}}\,I_\lambda\!\left(\lambda = \frac{hc}{E}\right),
   \qquad \lambda = \frac{hc}{E}

The :math:`\mathrm{d}\lambda/\mathrm{d}E` factor is the Jacobian of the
change of variable, and the remaining :math:`1/E` turns an energy flux into a
photon count. The result is returned in m\ :sup:`-2` s\ :sup:`-1`
eV\ :sup:`-1`, with photon energy in eV in the first column.

Everything downstream of ``convert_spectrum`` expects this energy-space
array. Passing a raw wavelength-space spectrum, or the reverse, is the single
most common way to get a plausible-looking wrong answer out of these
functions.

Photocurrent
------------

In the SQ limit the absorber is a step function: transparent below the gap,
perfectly absorbing above it, with each absorbed photon delivering exactly
one collected electron-hole pair. The photogenerated current density is then
the photon flux above the gap,

.. math::

   J_{\mathrm{ph}} = q \int_{E_{\mathrm{g}}}^{\infty} \phi(E)\,\mathrm{d}E

which is what :func:`~solphin.db_fom.jsc` returns — under these assumptions
the short-circuit current and the photogenerated current are the same thing.
The integral itself is evaluated by the trapezium rule over the tabulated
spectrum.

Radiative recombination
-----------------------

Detailed balance fixes the dark current: a body that absorbs all photons
above :math:`E_{\mathrm{g}}` must also emit as a blackbody above
:math:`E_{\mathrm{g}}`. Integrating the Planck spectrum over the emitting
hemisphere gives the emitted photon rate per unit area at zero applied bias,

.. math::

   R_0 = \frac{2\pi}{c^{2}h^{3}}
         \int_{E_{\mathrm{g}}}^{\infty}
         \frac{E^{2}}{\exp\!\left(E/k_{\mathrm{B}}T\right) - 1}\,\mathrm{d}E

computed by ``_rr0``. Under an applied bias :math:`V` the quasi-Fermi levels
split by :math:`qV` and the emission is amplified by the Boltzmann factor
:math:`\exp\!\left(qV/k_{\mathrm{B}}T\right)`, giving the recombination
current of :func:`~solphin.db_fom.recomb_rate`.

Current-voltage characteristic and the operating point
------------------------------------------------------

Subtracting emission from absorption gives the ideal diode characteristic
evaluated by :func:`~solphin.db_fom.current_density`:

.. math::

   J(V) = q\left[\,\frac{J_{\mathrm{ph}}}{q}
                 - R_0\left(\mathrm{e}^{qV/k_{\mathrm{B}}T} - 1\right)\right]

Open circuit is where the two balance, which
:func:`~solphin.db_fom.voc` evaluates in closed form:

.. math::

   V_{\mathrm{oc}} = \frac{k_{\mathrm{B}}T}{q}
                     \ln\!\left(\frac{J_{\mathrm{ph}}/q}{R_0} + 1\right)

The maximum power point has no closed form, so
:func:`~solphin.db_fom.max_power` scans :math:`V` on a grid spanning
:math:`0` to :math:`V_{\mathrm{oc}}` and takes the largest :math:`JV`
product. :func:`~solphin.db_fom.v_at_mpp` and
:func:`~solphin.db_fom.j_at_mpp` return the voltage and current there.

.. note::

   The scan uses ``numpy.linspace`` at its default of 50 points. That is
   ample for the efficiency itself, which is stationary at the maximum, but
   :math:`V_{\mathrm{mpp}}` and :math:`J_{\mathrm{mpp}}` are quantised to the
   grid — so a fill factor from :func:`~solphin.db_fom.fill_factor` carries a
   coarser error than the efficiency does.

Efficiency and fill factor
--------------------------

The power conversion efficiency of :func:`~solphin.db_fom.max_eff` is the
maximum power density divided by the total incident power density, the latter
recovered from the same photon-flux array by re-weighting with the photon
energy:

.. math::

   \eta_{\mathrm{SQ}} = \frac{\max_V\left[J(V)\,V\right]}
                             {\displaystyle\int_0^{\infty} E\,\phi(E)\,\mathrm{d}E}

and the fill factor of :func:`~solphin.db_fom.fill_factor` is the usual
squareness measure,

.. math::

   \mathrm{FF} = \frac{J_{\mathrm{mpp}}V_{\mathrm{mpp}}}
                      {J_{\mathrm{sc}}V_{\mathrm{oc}}}

Under AM1.5G at 300 K this reproduces the familiar SQ curve, peaking near
33.7 % at a gap of about 1.34 eV. :mod:`solphin.db_plots` draws that curve
(:func:`~solphin.db_plots.sq_limit_plot`) together with the photon-flux
budget, the *J-V* curve and the power curve; ``plot_db_combined`` assembles
the panel, and ``plot_db_combined_interactive`` puts the band gap on a slider.

What the limit ignores
----------------------

Three assumptions do the work above, and each is relaxed on a later page:

* **Step-function absorption.** A real :math:`\alpha(E)` rises gradually from
  the gap and is finite above it, so a film of finite thickness absorbs less
  than the SQ ideal. Relaxed in :doc:`formalism_optics`.
* **Purely radiative recombination.** Real absorbers recombine through
  defects far faster than they radiate. Relaxed by the SLME emission
  fraction, and much more thoroughly by
  :math:`\Gamma_{\mathrm{PV}}`'s lifetime :math:`\tau`.
* **Perfect carrier collection.** Every photogenerated carrier is assumed to
  reach a contact. Transport is what :math:`\Gamma_{\mathrm{PV}}` prices in
  through :math:`\mu`, :math:`n` and :math:`\epsilon`, and it is the main
  reason the :math:`\eta_\Gamma` limit is stricter than any detailed-balance
  limit. Relaxed in :doc:`formalism_pv_fom`.

.. [Shockley1961] W. Shockley and H. J. Queisser, *Detailed balance limit of
   efficiency of p-n junction solar cells*, J. Appl. Phys. **32**, 510
   (1961). https://doi.org/10.1063/1.1736034
