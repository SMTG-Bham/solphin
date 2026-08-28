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

Averaging over an anisotropic material this way is a deliberate
simplification. For a strongly anisotropic absorber the three eigenvalues can
differ by enough that no single :math:`\alpha(E)` represents the material
well, and the polarisation-resolved curves are worth inspecting directly with
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
perfectly absorbing nor perfectly radiative. The chain implemented in
``_eta_d`` is as follows.

**Photon escape probability.** Of the photons generated inside the absorber,
only a fraction escape rather than being reabsorbed. Comparing the emission
that leaves the film with the emission generated throughout its volume gives

.. math::

   p_{\mathrm{e}} = \min\left[
       \frac{\displaystyle\int A(E,d)\,\phi_{\mathrm{bb}}(E)\,\mathrm{d}E}
            {\displaystyle 4d \int n^{2}(E)\,\alpha(E)\,\phi_{\mathrm{bb}}(E)\,\mathrm{d}E},
       \;1\right]

where :math:`\phi_{\mathrm{bb}}` is the blackbody photon flux at the cell
temperature. This is :func:`~solphin.optics.power_efficiency`. Note that the
denominator is thickness-independent, so :math:`p_{\mathrm{e}} \propto 1/d`
in the optically thick limit — thicker films reabsorb more of their own
luminescence.

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

   :math:`\Delta` is a proxy, not a measurement of the non-radiative rate. It
   captures the indirect-gap penalty and nothing else, so it says nothing
   about defect-mediated recombination in a direct-gap absorber — a material
   with :math:`\Delta = 0` and a picosecond Shockley-Read-Hall lifetime is
   scored at its radiative limit. Quantifying that loss is the job of
   :math:`\tau` in :doc:`formalism_pv_fom`.

Efficiency against thickness
----------------------------

:func:`~solphin.optics.make_blank_plot` sweeps :math:`d` over a logarithmic
range (by default :math:`10^{-8}` to :math:`10^{-3}` m, 80 points) and
evaluates all three curves — Blank with flat absorption, Blank with
Lambertian light trapping, and SLME — at each thickness. The absorption
coefficient is interpolated onto the solar wavelength grid with a cubic
spline, and clamped to its end values outside the calculated range.

Reading the resulting plot is the point of the exercise. The efficiency rises
with thickness while absorption is the binding constraint, then flattens once
the film is optically thick; the thickness at which it flattens is the
minimum film thickness the material needs. The gap between the flat and
Lambertian curves is how much light trapping is worth for this absorber —
large for a weak or indirect absorber, negligible for a strong direct one.
Unlike a drift-diffusion simulation, none of these curves turn over at large
thickness, because none of them models carrier collection.

.. [Blank2017] B. Blank, T. Kirchartz, S. Lany and U. Rau, *Selection metric
   for photovoltaic materials screening based on detailed-balance analysis*,
   Phys. Rev. Appl. **8**, 024032 (2017).
   https://doi.org/10.1103/PhysRevApplied.8.024032

.. [Yu2012] L. Yu and A. Zunger, *Identification of potential photovoltaic
   absorbers based on first-principles spectroscopic screening of materials*,
   Phys. Rev. Lett. **108**, 068701 (2012).
   https://doi.org/10.1103/PhysRevLett.108.068701
