The Photovoltaic Figure of Merit
================================

:mod:`solphin.pv_fom` implements :math:`\Gamma_{\mathrm{PV}}`, the
phenomenological figure of merit of Crovetto [Crovetto2024]_, and
:func:`solphin.final_results.SQ_relative_FOM_PV_efficiency` turns it into an
efficiency limit :math:`\eta_\Gamma`. Unlike the limits on the previous two
pages, this one prices in imperfect carrier collection, and so applies to a
real absorber with finite mobility, finite lifetime and a particular doping
density rather than to an idealised one.

How it was built
----------------

:math:`\Gamma_{\mathrm{PV}}` is not derived from solar-cell physics; it is
fitted to it. The construction had three stages:

1. **Reference dataset.** 2573 drift-diffusion simulations (SCAPS) of an
   idealised cell — the absorber between two perfectly carrier-selective
   contacts, with no interface, transport-layer or contact losses — sampling
   eight bulk properties independently. Each simulation was repeated over a
   range of absorber thicknesses and the efficiency taken at the optimum
   thickness :math:`d_{\mathrm{opt}}`, giving a benchmark maximum efficiency
   :math:`\eta_{\mathrm{sim}} \approx \eta_{\mathrm{max}}`.

2. **The figure of merit.** An expression was built up property by property,
   starting from the :math:`\alpha\tau` figure of merit of Kaienburg *et al.*
   and generalising it until all 2573 points collapsed onto a single curve
   when plotted against it.

3. **The efficiency limit.** A closed-form function was fitted to that
   curve, so that :math:`\eta_\Gamma` can be read off from
   :math:`\Gamma_{\mathrm{PV}}` directly, without the iterative thickness
   optimisation a drift-diffusion calculation needs.

Because every loss mechanism present in the simulations is folded into the
fit, :math:`\eta_\Gamma` is a *stricter* limit than any detailed-balance
result at the same band gap — and much stricter for low-mobility absorbers,
whose collection losses deductive methods omit entirely.

The eight properties
--------------------

:math:`\Gamma_{\mathrm{PV}}` is a function of the property set
:math:`\mathbb{P} = \{\bar{\alpha}, \sigma, \tau, \mu, n, \epsilon, m,
E_{\mathrm{g}}\}`. The sampled range is the range over which the fit was
trained; a prediction for an absorber falling well outside it may be badly
wrong.

.. list-table::
   :header-rows: 1
   :widths: 26 8 14 20 32

   * - Property
     - Symbol
     - Unit
     - Sampled range
     - Where ``solphin`` gets it
   * - Band gap
     - :math:`E_{\mathrm{g}}`
     - eV
     - 0.7 - 2.0
     - :func:`solphin.band_structure.get_band_structure`
   * - Average absorption coefficient
     - :math:`\bar{\alpha}`
     - cm\ :sup:`-1`
     - :math:`5\times10^{3}` - :math:`5\times10^{5}`
     - :func:`solphin.spectral.generate_spectral_parameters`
   * - Dispersion of absorption coefficient
     - :math:`\sigma`
     - --
     - 0.2 - 1.8
     - :func:`solphin.spectral.generate_spectral_parameters`
   * - Non-radiative (SRH) lifetime
     - :math:`\tau`
     - s
     - :math:`10^{-15}` - :math:`10^{3}`
     - supplied by the user
   * - Carrier mobility
     - :math:`\mu`
     - cm\ :sup:`2` V\ :sup:`-1` s\ :sup:`-1`
     - :math:`10^{-2}` - :math:`10^{9}`
     - supplied by the user
   * - Doping density
     - :math:`n`
     - cm\ :sup:`-3`
     - :math:`10^{10}` - :math:`10^{18}`
     - supplied by the user
   * - Static dielectric constant
     - :math:`\epsilon`
     - --
     - 1 - 100
     - :func:`solphin.optics.calc_dielectric` (see caveat below)
   * - DOS effective mass
     - :math:`m`
     - :math:`m_0`
     - 0.12 - 2.5
     - :func:`solphin.dos.get_dos_effective_mass`

Three of the eight — :math:`\tau`, :math:`\mu` and :math:`n` — are not
calculated by ``solphin``. They are measured properties of a particular
sample, or the output of separate first-principles machinery (carrier
capture, electron-phonon scattering, defect thermodynamics), and they must be
passed in. This is deliberate: the same electronic structure gives very
different efficiency limits at different material quality, and sweeping
:math:`\tau` and :math:`\mu` over plausible ranges is the intended way to use
the figure of merit. :func:`solphin.final_results.mobility_plot` and
:func:`solphin.final_results.plot_final_result_interactive` exist for exactly
that sweep.

.. warning::

   :math:`\epsilon` is the **static** dielectric constant — the zero-frequency
   limit, to which both electronic *and* ionic displacements contribute.
   :func:`~solphin.optics.calc_dielectric` returns the real part of
   :math:`\varepsilon(E)` at the lowest tabulated energy of the optical
   calculation, which is the **electronic** (high-frequency) constant
   :math:`\varepsilon_\infty` and omits the ionic response entirely. For an
   ionic absorber the difference is large — the reference value for
   CH\ :sub:`3`\ NH\ :sub:`3`\ PbI\ :sub:`3` is
   :math:`\epsilon = 33.5` against :math:`\varepsilon_\infty \approx 5` — and
   it moves :math:`\Gamma_{\mathrm{PV}}` through the :math:`\mathcal{D}_1`,
   :math:`\mathcal{D}_3` and :math:`\mathcal{T}_1` factors. Where the ionic
   contribution matters, obtain it from a DFPT or finite-difference
   calculation (``LEPSILON = .TRUE.`` or ``IBRION = 8`` in VASP;
   ``efield`` / ``phonon`` in CASTEP) and pass that value instead.

Units
-----

:math:`\Gamma_{\mathrm{PV}}` contains logarithms, exponentials and fractional
powers of its arguments, so each property must be made unitless before it is
used. The convention throughout is division by the unit in the table above:
:math:`\alpha/(\mathrm{cm}^{-1})`, :math:`\tau/(\mathrm{s})`,
:math:`\mu/(\mathrm{cm}^{2}\,\mathrm{V}^{-1}\mathrm{s}^{-1})`,
:math:`n/(\mathrm{cm}^{-3})`, :math:`m/m_0` and
:math:`E_{\mathrm{g}}/(\mathrm{eV})`. :math:`\sigma` and :math:`\epsilon` are
already unitless. The functions in :mod:`solphin.pv_fom` take bare floats and
assume they are in these units — there is no unit checking, and passing
:math:`\tau` in nanoseconds will silently return a number.

Spectral descriptors of the absorption coefficient
--------------------------------------------------

Two numbers stand in for the whole :math:`\alpha(\lambda)` spectrum: its
average and its spread. Both are weighted by the incident spectrum over the
range that matters for photocurrent — from :math:`\lambda_1 = 300` nm, below
which the photon flux is negligible and calculated data unreliable, up to the
band-gap wavelength :math:`\lambda_{\mathrm{g}} = hc/E_{\mathrm{g}}`:

.. math::

   \bar{\alpha} = \frac{\displaystyle\int_{\lambda_1}^{\lambda_{\mathrm{g}}}
                        \alpha(\lambda)\,\phi(\lambda)\,\mathrm{d}\lambda}
                       {\displaystyle\int_{\lambda_1}^{\lambda_{\mathrm{g}}}
                        \phi(\lambda)\,\mathrm{d}\lambda}

.. math::

   \sigma = \sqrt{\frac{\displaystyle\int_{\lambda_1}^{\lambda_{\mathrm{g}}}
                        \phi(\lambda)\left[\log\alpha(\lambda) - \log\bar{\alpha}\right]^{2}
                        \mathrm{d}\lambda}
                       {\displaystyle\int_{\lambda_1}^{\lambda_{\mathrm{g}}}
                        \phi(\lambda)\,\mathrm{d}\lambda}}

The dispersion :math:`\sigma` matters because a spectrum spread over orders
of magnitude generates carriers at very different depths: the generation
profile is a sum of exponentials with very different penetration depths, so
at fixed :math:`\bar{\alpha}` a high-:math:`\sigma` absorber needs a thicker
film to capture the same fraction of the spectrum, and a thicker film means
more volume in which to recombine. This is why crystalline silicon, with its
gradual indirect absorption onset, scores worse than a chalcogenide or
halide perovskite at the same :math:`\bar{\alpha}\tau`.

:func:`solphin.spectral.generate_spectral_parameters` computes both from the
``absorption.dat`` written by :func:`solphin.optics.generate_absorption` and
a wavelength-space reference spectrum. It truncates the absorption and light
spectra to the window above, maps the spectrum onto the absorption
wavelength grid by nearest neighbour, and integrates with Simpson's rule.
Points with :math:`\alpha \le 0` are dropped before the logarithm.

.. note::

   ``solphin``'s evaluation differs from the definitions above in three
   respects, all in :func:`~solphin.spectral.calculate_spectral_dispersion`
   and :func:`~solphin.spectral.calculate_spectral_average`:

   * the weights are the spectral **irradiance**
     :math:`I_\lambda(\lambda)` rather than the photon flux
     :math:`\phi(\lambda)`, which overweights short wavelengths by
     :math:`1/\lambda`;
   * :math:`\sigma` uses the natural logarithm, where the reference values it
     is compared against (a sampled range of 0.2 - 1.8, and
     :math:`\sigma = 0.63` for CH\ :sub:`3`\ NH\ :sub:`3`\ PbI\ :sub:`3`)
     are base-10, so the computed value is larger by a factor of
     :math:`\ln 10 \approx 2.303`;
   * :math:`\sigma` is centred on the weighted mean of
     :math:`\log\alpha` — the logarithm of the geometric mean — rather than
     on :math:`\log\bar{\alpha}`, the logarithm of the arithmetic mean
     defined above.

   :math:`\sigma` enters :math:`\Gamma_{\mathrm{PV}}` as
   :math:`\sigma^{10}` and as an exponent of :math:`\bar{\alpha}`, so these
   are not cosmetic differences. Treat :math:`\sigma` from
   ``generate_spectral_parameters`` as a relative descriptor for comparing
   materials computed the same way, and supply :math:`\sigma` explicitly when
   an absolute value from the literature is what you want.

Density-of-states effective mass
--------------------------------

:math:`m` is the DOS effective mass, obtained by fitting the calculated
density of states near a band edge to the form a three-dimensional parabolic
band would give. For the conduction band,

.. math::

   g(E) = \frac{1}{2\pi^{2}}
          \left(\frac{2m_{\mathrm{e}}}{\hbar^{2}}\right)^{3/2}
          \left(E - E_{\mathrm{c}}\right)^{1/2}

so that a least-squares fit of :math:`g` against :math:`\sqrt{E -
E_{\mathrm{c}}}` through the origin gives a coefficient :math:`A` from which
the mass follows in closed form:

.. math::

   m_{\mathrm{e}} = \frac{\hbar^{2}}{2}
                    \left(2\pi^{2}A\right)^{2/3}

:func:`solphin.dos.get_dos_effective_mass` does this over a window above
:math:`E_{\mathrm{c}}` (or below :math:`E_{\mathrm{v}}` for holes) that
defaults to 0.15 eV, and reports the :math:`R^{2}` of the fit alongside the
mass. The window is the main judgement call: too narrow and the fit is
dominated by the DOS sampling grid, too wide and it leaves the region where a
parabolic band is a fair description.
:func:`solphin.dos.test_dos_mass_windows` sweeps it so that the sensitivity
is visible, and :func:`solphin.dos.print_dos_summary` flags fits whose
quality does not support the number.

Electrons and holes generally have different masses. The figure of merit
takes one, and the paper's prescription for reducing two to one is the
geometric mean :math:`m = \sqrt{m_{\mathrm{e}}m_{\mathrm{h}}}`, which has to
be formed by the caller — ``get_dos_effective_mass`` returns one carrier at a
time. Note also that this DOS mass is not the conductivity effective mass
:math:`M` that enters :math:`\mu = q\tau_{\mathrm{s}}/M`; the two are derived
differently and are not interchangeable.

The figure of merit
-------------------

The master expression, equation 18 of [Crovetto2024]_, is

.. math::

   \Gamma_{\mathrm{PV}}\left(\bar{\alpha}, \sigma, \tau, \mu, n, \epsilon, m, E_{\mathrm{g}}\right)
     = E_{\mathrm{g}}^{2.5}
       \left(\frac{\mathcal{A}_1\mathcal{A}_2\mathcal{D}_1}
                  {\mathcal{D}_2\mathcal{D}_3\mathcal{D}_4
                   \left(1 + \mathcal{T}_1\mathcal{T}_2\mathcal{T}_3\right)
                   \left(1 + \mathcal{S}_1\mathcal{S}_2\right)}
       \right)^{E_{\mathrm{g}}^{-0.8}}

evaluated by :func:`solphin.pv_fom.Final_equation`. The factors fall into
four groups, each of which was introduced to absorb the effect of one more
property:

**Absorption** (:math:`\mathcal{A}`), the generalisation of the
:math:`\alpha\tau` figure of merit to a spectrum of finite dispersion:

.. math::

   \mathcal{A}_1 &= \frac{\bar{a}_1\,\tau\,
                          \bar{\alpha}^{\,\sigma^{-\bar{a}_2 E_{\mathrm{g}}^{0.5}}}}
                         {m^{2}} \\
   \mathcal{A}_2 &= 1 + \left(\frac{\bar{a}_3\,\sigma^{10}}
                                   {\bar{\alpha}\tau}\right)^{0.4}

**Doping** (:math:`\mathcal{D}`), which enters mainly through the
quasi-Fermi-level splitting and the width of the depletion regions.
:math:`\mathcal{D}_1` raises the efficiency when the absorber is in low
injection (:math:`n \gtrsim \bar{\alpha}^{2}/\bar{d}_1`);
:math:`\mathcal{D}_2` and :math:`\mathcal{D}_4` are corrections at high and
low :math:`\bar{\alpha}\tau` respectively:

.. math::

   \mathcal{D}_1 &= \left(1 + \frac{\bar{d}_1 n}
                                   {\epsilon^{0.8}\bar{\alpha}^{2}}\right)
                    ^{0.22\log\left(\bar{\alpha}/\bar{d}_2\right)} \\
   \mathcal{D}_2 &= \left(1 + \frac{\bar{d}_3 n}
                                   {\bar{\alpha}^{2}\tau}\right)^{0.05E_{\mathrm{g}}^{4}} \\
   \mathcal{D}_3 &= 1 + \frac{\left(\bar{d}_4 E_{\mathrm{g}}^{8.5}\tau\,
                                    \bar{\alpha}^{\,0.68E_{\mathrm{g}}^{-1.5}}\right)
                                    ^{\log\left(10n/\epsilon\right)/\bar{d}_5}}
                              {1 + 10^{\left(E_{\mathrm{g}} - 1.5\right)/0.1}} \\
   \mathcal{D}_4 &= 1 + \left(\frac{\bar{d}_6}
                                   {E_{\mathrm{g}}^{17}\bar{\alpha}\tau}\right)^{0.6}

**Transport** (:math:`\mathcal{T}`), the imperfect-collection penalty. This
group is what deductive detailed-balance methods have no counterpart for:

.. math::

   \mathcal{T}_1 &= \frac{\bar{t}_1\left(E_{\mathrm{g}} + 0.5\right)^{11}}
                         {m\,\epsilon^{0.5}\,
                          \mu^{\left(\bar{t}_2 E_{\mathrm{g}}^{4.3} + 0.9\right)}} \\
   \mathcal{T}_2 &= \left(\frac{1 + \bar{t}_8 n\left(1 + E_{\mathrm{g}}^{5.1}\right)}
                               {\bar{t}_9\,\tau\,10^{0.5/\sigma}}\right)
                    ^{0.47E_{\mathrm{g}}^{1.25}} \\
   \mathcal{T}_3 &= \left(1 + \mathcal{T}_3'\mathcal{T}_3''\right)
                    ^{\bar{t}_7 E_{\mathrm{g}}^{8} + 0.6} \\
   \mathcal{T}_3' &= \bar{t}_3\left(1 + \bar{t}_4 E_{\mathrm{g}}^{10}\right)
                     \exp\!\left(-0.1\,
                     \bar{\alpha}^{\,0.5/\left(1 + \bar{t}_5 E_{\mathrm{g}}^{10}\right)}\right) \\
   \mathcal{T}_3'' &= 1 + \frac{\dfrac{0.16}{m^{3}}
                                \left(\dfrac{n m^{3}}{0.16}\right)
                                ^{1 - 0.74\exp\left(-\bar{\alpha}/\bar{t}_6\right)}}
                               {1 + 10^{\left(E_{\mathrm{g}} - 1.5\right)/0.01}}

**Saturation** (:math:`\mathcal{S}`), which caps the efficiency when the
diffusion length is limited by the radiative rather than the SRH lifetime, so
that increasing :math:`\tau` further buys nothing:

.. math::

   \mathcal{S}_1 &= \frac{10^{2.4E_{\mathrm{g}}}\,\bar{\alpha}^{\,0.75}\,\tau}
                         {\bar{s}_1 m^{2}
                          \mu^{1/\left(1 + \bar{s}_2 E_{\mathrm{g}}^{10}\right)}} \\
   \mathcal{S}_2 &= 1 + \left(\frac{\bar{s}_3}{\bar{\alpha}}\right)^{20}
                        \mu^{0.5}\log n

All logarithms are base 10. Each factor has its own private function in
:mod:`solphin.pv_fom` (``_A_1_equation``, ``_D_3_equation`` and so on), which
is the practical way to see which property drives a given result: evaluate
the factors separately and look for the one that is orders of magnitude away
from unity.

Fitted parameters
-----------------

The parameters below are pure numbers obtained from the fit, listed in the
supplementary material of [Crovetto2024]_ and hard-coded in
:mod:`solphin.pv_fom`:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Absorption and saturation
     - Doping
     - Transport
   * - :math:`\bar{a}_1 = 0.295`
     - :math:`\bar{d}_1 = 4.4\times10^{-5}`
     - :math:`\bar{t}_1 = 5.1\times10^{-2}`
   * - :math:`\bar{a}_2 = 0.185`
     - :math:`\bar{d}_2 = 39`
     - :math:`\bar{t}_2 = 4.6\times10^{-2}`
   * - :math:`\bar{a}_3 = 1\times10^{-7}`
     - :math:`\bar{d}_3 = 1\times10^{-21}`
     - :math:`\bar{t}_3 = 1.9\times10^{4}`
   * - :math:`\bar{s}_1 = 2.4\times10^{4}`
     - :math:`\bar{d}_4 = 2.1\times10^{4}`
     - :math:`\bar{t}_4 = 9.5\times10^{-3}`
   * - :math:`\bar{s}_2 = 4\times10^{-4}`
     - :math:`\bar{d}_5 = 50`
     - :math:`\bar{t}_5 = 2.4\times10^{-4}`
   * - :math:`\bar{s}_3 = 4.8\times10^{3}`
     - :math:`\bar{d}_6 = 7.7\times10^{-7}`
     - :math:`\bar{t}_6 = 1.5\times10^{5}`
   * -
     -
     - :math:`\bar{t}_7 = 1.6\times10^{-3}`
   * -
     -
     - :math:`\bar{t}_8 = 9.5\times10^{-18}`
   * -
     -
     - :math:`\bar{t}_9 = 7.80\times10^{7}`

From the figure of merit to an efficiency
-----------------------------------------

:math:`\Gamma_{\mathrm{PV}}` is a ranking number, not an efficiency. The map
onto one, equation 33 of [Crovetto2024]_, is the fit to the collapsed
:math:`\eta_{\mathrm{sim}}` dataset:

.. math::

   \eta_\Gamma = \frac{\eta_{\mathrm{SQ}}}
                      {\left(1 + \dfrac{k_1\Gamma_{\mathrm{PV}}^{-0.235}}
                                       {1 + k_2\Gamma_{\mathrm{PV}}^{0.869}}\right)
                       \left(1 + k_3\Gamma_{\mathrm{PV}}^{-0.362}\right)}

with :math:`k_1 = 3.3\times10^{-1}`, :math:`k_2 = 9.06\times10^{-2}` and
:math:`k_3 = 2.48\times10^{-3}`, and :math:`\eta_{\mathrm{SQ}}` the
Shockley-Queisser limit at the same band gap, from
:func:`solphin.db_fom.max_eff`. This is what
:func:`solphin.final_results.SQ_relative_FOM_PV_efficiency` returns, together
with :math:`\eta_{\mathrm{SQ}}` itself and the ratio
:math:`\eta_{\Gamma}/\eta_{\mathrm{SQ}}`.

The shape of this function carries the physics. A simpler piecewise form
(equation 32 of the paper) captures it:

.. math::

   \eta_\Gamma' = \begin{cases}
       0 & \Gamma_{\mathrm{PV}} < 10^{-6} \\
       \left[\frac{1}{8}\log\Gamma_{\mathrm{PV}} + \frac{3}{4}\right]\eta_{\mathrm{SQ}}
         & 10^{-6} \le \Gamma_{\mathrm{PV}} \le 10^{2} \\
       \eta_{\mathrm{SQ}} & \Gamma_{\mathrm{PV}} > 10^{2}
   \end{cases}

Over eight orders of magnitude in the middle, efficiency goes as the
*logarithm* of the figure of merit. An order-of-magnitude improvement in
:math:`\tau` or :math:`\mu` buys a fixed number of efficiency points, not a
proportional gain — and once :math:`\Gamma_{\mathrm{PV}} \gtrsim 10^{2}` the
absorber is already at its Shockley-Queisser limit and further bulk
improvement is wasted. Equation 33 is used in preference because the
piecewise form fits poorly around
:math:`10^{-8} < \Gamma_{\mathrm{PV}} < 10^{-5}`.

Accuracy and validity
---------------------

Against the 2573-point simulation dataset the fit gives :math:`R^{2} = 0.995`
with a mean squared error of 11.7 % absolute on
:math:`\eta_{\mathrm{sim}}/\eta_{\mathrm{SQ}}`. That is:

* :math:`\pm 3.4` % absolute on the efficiency expressed as a fraction of the
  Shockley-Queisser limit;
* :math:`\pm 1.1` % absolute on the power conversion efficiency itself, at a
  band gap of 1.5 eV.

Both are error bars on how well :math:`\eta_\Gamma` reproduces the
drift-diffusion benchmark. They say nothing about the uncertainty in the
eight properties fed in, which in a first-principles workflow is usually the
larger term by some margin — a factor of two in :math:`\tau` moves
:math:`\eta_\Gamma` further than the fit error does.

The assumptions behind the number are worth keeping in view:

* **Planar single junction**, obeying the Shockley-Queisser construction, at
  its optimal thickness. No tandem, no textured or nanostructured
  architecture.
* **Bulk properties only.** Contacts, interfaces and transport layers are
  ideal by construction. A real device will do worse; the gap between
  :math:`\eta_\Gamma` and a measured efficiency is the device engineering
  that remains.
* **One value per property.** Electrons and holes are assumed to share a
  lifetime (:math:`\tau_{\mathrm{n}} = \tau_{\mathrm{p}}`), a mobility
  (:math:`\mu_{\mathrm{n}} = \mu_{\mathrm{p}}`) and a DOS effective mass
  (:math:`m_{\mathrm{e}} = m_{\mathrm{h}}`). Strongly asymmetric absorbers
  need an effective single value, which introduces ambiguity the figure of
  merit cannot resolve.
* **Isotropy.** All eight properties are treated as scalars. For a layered or
  strongly anisotropic material there is no unambiguous scalar
  :math:`\bar{\alpha}`, :math:`\mu` or :math:`m`.
* **Shockley-Read-Hall and radiative recombination only.** Auger
  recombination is neglected, which is part of why doping densities above
  :math:`10^{18}` cm\ :sup:`-3` are excluded — as is the Boltzmann
  approximation breaking down there.
* **No sub-gap absorption.** :math:`\alpha(E) = 0` below
  :math:`E_{\mathrm{g}}` is enforced, and the electronic gap is assumed equal
  to the optical gap. Excitonic or defect absorption, or band-tail states,
  break this, and in their presence a PV-relevant gap definition would serve
  better than :math:`E_{\mathrm{c}} - E_{\mathrm{v}}`.
* **The eight properties are independent.** They are not, in reality —
  a heavier :math:`m` generally means lower :math:`\mu`, a larger
  :math:`\epsilon` tends to come with longer :math:`\tau`. The fit was built
  by varying each independently, so trends read off by varying one property
  alone (for instance "lower :math:`\epsilon` is better at high
  :math:`\mu`") describe the direct effect only, and the indirect effects
  often run the other way.

.. [Crovetto2024] A. Crovetto, *A phenomenological figure of merit for
   photovoltaic materials*, J. Phys. Energy **6**, 025009 (2024).
   https://doi.org/10.1088/2515-7655/ad2499
