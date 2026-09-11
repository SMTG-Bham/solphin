Optical Constants and Thickness-Dependent Efficiency
====================================================

:mod:`solphin.optics` takes the calculated dielectric function of the
absorber, turns it into the optical constants a solar-cell model needs, and
then evaluates the efficiency of a film of that material as a function of its
thickness. Two independent models are available — the detailed-balance
treatment of Blank *et al.* [Blank2017]_ and the spectroscopic limited
maximum efficiency (SLME) of Yu and Zunger [Yu2012]_ — and they are plotted
against each other on the same axes.

From the dielectric tensor to the absorption coefficient
--------------------------------------------------------

Both VASP and CASTEP (through OptaDOS) supply the frequency-dependent complex
dielectric tensor :math:`\varepsilon_{ij}(E)`.
:func:`~solphin.optics.calc_dielectric` reads it;
:func:`~solphin.optics.calc_absorption` reduces it to scalar optical
constants.

The tensor is first diagonalised, and the complex refractive index is formed
per eigenvalue before averaging — not the other way round — so that the
average is taken over the physically meaningful quantity. This matches
``sumo``'s convention:

.. math::

   \tilde{n}(E) = n(E) + \mathrm{i}\,\kappa(E)
                = \frac{1}{3}\sum_{j=1}^{3}\sqrt{\varepsilon_j(E)}

The absorption coefficient follows from the extinction coefficient
:math:`\kappa`, and the energy loss function from the averaged scalar
dielectric function :math:`\varepsilon = \frac{1}{3}\sum_j \varepsilon_j`:

.. math::

   \alpha(E) = \frac{4\pi E\,\kappa(E)}{hc},
   \qquad
   L(E) = \mathrm{Im}\!\left[\frac{-1}{\varepsilon(E)}\right]

:func:`~solphin.optics.calc_absorption` returns :math:`\alpha` in
m\ :sup:`-1`; the ``absorption.dat`` file written by
:func:`~solphin.optics.generate_absorption` is in cm\ :sup:`-1`, which is the
unit the figure of merit expects.

Averaging an anisotropic material this way is a deliberate simplification:
for a strongly anisotropic absorber no single :math:`\alpha(E)` represents
the material well — inspect the polarisation-resolved curves with
:func:`~solphin.optics.plot_absorption`.

How much light a film of thickness *d* absorbs
----------------------------------------------

Neither model assumes step-function absorption. Instead the absorptance
:math:`A(E, d)` — the fraction of incident photons absorbed — is built from
:math:`\alpha(E)` under one of two optical models:

.. math::

   A_{\mathrm{flat}}(E, d)      &= 1 - \mathrm{e}^{-2\alpha(E) d} \\
   A_{\mathrm{Lambertian}}(E, d) &= 1 - \frac{1}{1 + 4n^{2}\alpha(E) d}

The first is Beer-Lambert absorption over a double pass, the factor of two
standing for a perfect back reflector. The second is the Lambertian
light-trapping limit for a randomly textured surface, in which the
:math:`4n^{2}` enhancement is the classical Yablonovitch factor; it is the
more generous of the two, and increasingly so as :math:`\alpha d` falls.
Both are clipped to :math:`[0, 1]`. The refractive index used for the
:math:`4n^{2}` factor is the scalar ``n`` argument of
:func:`~solphin.optics.make_blank_plot` (default 3.5), not the
energy-resolved :math:`n(E)`.

The Blank detailed-balance model
--------------------------------

Blank *et al.* generalise the SQ construction to a film that is neither
perfectly absorbing nor perfectly radiative. The chain, evaluated at each
thickness by :func:`~solphin.optics.make_blank_plot`, is as follows.

**Photon escape probability.** Of the photons generated inside the absorber,
only a fraction escape rather than being reabsorbed. Comparing the emission
that leaves the film with the emission generated throughout its volume gives

.. math::

   p_{\mathrm{e}} = \min\left[
       \frac{\displaystyle\int A(E,d)\,\phi_{\mathrm{bb}}(E)\,\mathrm{d}E}
            {\displaystyle 4d \int n^{2}(E)\,\alpha(E)\,\phi_{\mathrm{bb}}(E)\,\mathrm{d}E},
       \;1\right]

where :math:`\phi_{\mathrm{bb}}` is the blackbody photon flux at the cell
temperature (computed by :func:`~solphin.optics.power_efficiency`).

**External luminescence efficiency.** Combining the escape probability with
the internal luminescence efficiency :math:`Q_{\mathrm{i}}` — the fraction of
recombination events that are radiative, and the model's single knob for
non-radiative loss — gives the external efficiency

.. math::

   Q_{\mathrm{e}} = \frac{p_{\mathrm{e}} Q_{\mathrm{i}}}
                         {1 + \left(p_{\mathrm{e}} - 1\right) Q_{\mathrm{i}}}

The default :math:`Q_{\mathrm{i}} = 1` is the radiative limit; lowering it is
how a known non-radiative lifetime is fed into this model.

**Currents.** The radiative saturation current is the emission of the film
into a hemisphere, and the total saturation current is that divided by the
external luminescence efficiency, so that non-radiative loss enters as an
inflated dark current:

.. math::

   J_0^{\mathrm{rad}} = q\pi\int \phi_{\mathrm{bb}}(\lambda)\,A(\lambda, d)\,\mathrm{d}\lambda,
   \qquad
   J_0 = \frac{J_0^{\mathrm{rad}}}{Q_{\mathrm{e}}},
   \qquad
   J_{\mathrm{sc}} = q\int \phi_{\odot}(\lambda)\,A(\lambda, d)\,\mathrm{d}\lambda

**Operating point.** The diode characteristic is then the standard one,
scanned for its maximum power point and divided by the incident power density
:math:`P_{\mathrm{in}}` obtained by integrating the solar irradiance:

.. math::

   J(V) = J_{\mathrm{sc}} - J_0\left(\mathrm{e}^{qV/k_{\mathrm{B}}T} - 1\right),
   \qquad
   \eta(d) = \frac{\max_V\left[J(V)\,V\right]}{P_{\mathrm{in}}}

.. _illuminance-normalisation:

Illuminance normalisation of the indoor spectra
-----------------------------------------------

:math:`\eta` above is not invariant under a rescaling of the spectrum:
:math:`J_{\mathrm{sc}}` grows with the illumination while :math:`J_0` is fixed
by the cell temperature, so :math:`V_{\mathrm{oc}} \sim \ln(J_{\mathrm{sc}}/J_0)`
and hence the efficiency both depend on *how much* light falls on the cell, not
only on its spectral shape. An indoor efficiency is therefore meaningless
unless the light level is quoted with it.

The bundled illuminant files do not share a common scale as supplied — the red,
white and infrared LED spectra are normalised to a peak of unity, the
fluorescent, blue and green LED spectra are absolute measurements taken at
different illuminances, and the nine CIE 15:2018 standard LED illuminants
(``LED-B1`` to ``LED-B5``, ``LED-BH1``, ``LED-RGB1``, ``LED-V1`` and
``LED-V2``) are tabulated on an arbitrary scale of their own. ``solphin``
therefore treats them all as *relative* spectra and rescales each to a target
illuminance

.. math::

   E_{\mathrm{v}} = K_{\mathrm{m}} \int V(\lambda)\,E(\lambda)\,\mathrm{d}\lambda,
   \qquad
   K_{\mathrm{m}} = 683\ \mathrm{lm\,W^{-1}},

with :math:`V(\lambda)` the photopic luminosity function (bundled as
``photopic.csv``) and :math:`K_{\mathrm{m}}` its efficacy at 555 nm, the
definition of the lumen. The target is the ``target_lux`` argument of
:func:`solphin.db_fom.load_spectrum` and
:func:`solphin.optics.make_blank_plot`, 1000 lx by default, and the resulting
spectra carry a few W m⁻² — some three orders of magnitude below one sun.

Two spectra are exempt:

* **AM1.5G** is an absolute standard integrating to 1000 W m⁻², and is used
  exactly as supplied; ``target_lux`` does not apply to it.
* The **infrared LED** peaks at 849 nm, where :math:`V(\lambda) \approx 0`, so
  its illuminance is ~0.009 lx and normalising it by :math:`E_{\mathrm{v}}`
  would inflate it by five orders of magnitude. It is instead matched on
  *radiant power*, to the irradiance the white LED carries at the same target
  illuminance. Comparisons involving it are power-matched rather than
  illuminance-matched.

A common wavelength support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The source files cover very different wavelength ranges: 280-4000 nm for
AM1.5G, 300-2420 nm for the fluorescent and white LED spectra, and only
380-780 nm for the CIE LED illuminants. That range is not a presentational
detail, because the radiative recombination integrals above take their
wavelength grid from the illumination spectrum — :math:`\phi_{\odot}` and
:math:`\phi_{\mathrm{bb}}` are integrated over the same grid, in
:func:`solphin.db_fom._rr0` as well as in the :math:`J_0^{\mathrm{rad}}` of
the Blank model. A spectrum ending at 780 nm therefore truncates the blackbody
integral at 1.59 eV, and for any band gap below that returns a recombination
current orders of magnitude too small, so that

.. math::

   V_{\mathrm{oc}} \sim \frac{k_{\mathrm{B}}T}{q}\ln\frac{J_{\mathrm{sc}}}{J_0}

and the efficiency with it come back far too high — and, worse, stop varying
with the gap at all.

Every loaded spectrum is therefore padded with **zero irradiance** out to a
common 280-4000 nm support. Padding changes no measured quantity — a region of
zero irradiance contributes nothing to :math:`E_{\mathrm{v}}`,
:math:`P_{\mathrm{in}}` or :math:`J_{\mathrm{sc}}` — but it gives the
blackbody integral the grid it needs, and it replaces the flat extrapolation
that :func:`solphin.spectral._resample_common_grid` would otherwise apply
beyond a spectrum's last tabulated point with the zero that belongs there.
With the support shared, :math:`J_0^{\mathrm{rad}}` becomes a property of the
gap and the cell temperature alone, as detailed balance requires, rather than
an artefact of which lamp file was selected.

Spectroscopic limited maximum efficiency
----------------------------------------

SLME shares the absorptance and the diode law but replaces the escape-probability
machinery with a single empirical factor. The fraction of recombination that
is radiative is estimated from how far the lowest allowed *direct* transition
:math:`E_{\mathrm{g}}^{\mathrm{d}}` lies above the fundamental, possibly
indirect, gap :math:`E_{\mathrm{g}}^{\mathrm{i}}`:

.. math::

   f_{\mathrm{r}} = \exp\!\left(-\frac{\Delta}{k_{\mathrm{B}}T}\right),
   \qquad
   \Delta = E_{\mathrm{g}}^{\mathrm{d}} - E_{\mathrm{g}}^{\mathrm{i}},
   \qquad
   J_0 = \frac{J_0^{\mathrm{rad}}}{f_{\mathrm{r}}}

The reasoning is that carriers thermalise to the band edge, so in an indirect
absorber they sit :math:`\Delta` below the energy at which they could
recombine radiatively, and radiative recombination is suppressed by the
corresponding Boltzmann factor. For a direct-gap material
:math:`\Delta = 0`, :math:`f_{\mathrm{r}} = 1` and SLME reduces to the
radiative limit with realistic absorption — which is why ``solphin`` draws
the SLME curve with a dashed line when the two gaps coincide, as a reminder
that the penalty term is inactive.

SLME itself is evaluated by ``pymatgen.analysis.solar.slme.slme``, with
its own bundled AM1.5G spectrum. It is therefore only computed when the
AM1.5G spectrum is selected; for the indoor spectra the plot shows the two
Blank curves alone.

.. note::

   :math:`\Delta` prices only the indirect-gap penalty — a direct-gap
   absorber with a picosecond Shockley-Read-Hall lifetime is still scored at
   its radiative limit. Non-radiative loss is instead quantified by
   :math:`\tau` in :doc:`formalism_pv_fom`.

Efficiency against thickness
----------------------------

:func:`~solphin.optics.make_blank_plot` sweeps :math:`d` over a logarithmic
range (by default :math:`10^{-8}` to :math:`10^{-3}` m, 80 points) and
evaluates all three curves — Blank with flat absorption, Blank with
Lambertian light trapping, and SLME — at each thickness. The absorption
coefficient is interpolated onto the solar wavelength grid with a cubic
spline, and clamped to its end values outside the calculated range.

The efficiency rises with thickness while absorption is the binding
constraint, then flattens once the film is optically thick — the flattening
point is the minimum film thickness the material needs, and the gap between
the flat and Lambertian curves is what light trapping is worth for this
absorber. None of the curves turn over at large thickness, because none of
them models carrier collection.

.. [Blank2017] B. Blank, T. Kirchartz, S. Lany and U. Rau, *Selection metric
   for photovoltaic materials screening based on detailed-balance analysis*,
   Phys. Rev. Appl. **8**, 024032 (2017).
   https://doi.org/10.1103/PhysRevApplied.8.024032

.. [Yu2012] L. Yu and A. Zunger, *Identification of potential photovoltaic
   absorbers based on first-principles spectroscopic screening of materials*,
   Phys. Rev. Lett. **108**, 068701 (2012).
   https://doi.org/10.1103/PhysRevLett.108.068701
