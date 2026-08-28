Photovoltaic Figure of Merit and Realistic Efficiency Limits
============================================================

:mod:`solphin.pv_fom` implements :math:`\Gamma_{\mathrm{PV}}`, the
phenomenological photovoltaic figure of merit of Crovetto [Crovetto2024]_, and
:mod:`solphin.final_results` converts it into an efficiency limit. Where the
detailed-balance models of :doc:`formalism_optics` ask what a film of a given
thickness would achieve if optics and radiative recombination were the only
things that mattered, :math:`\Gamma_{\mathrm{PV}}` asks a blunter question:
given eight bulk properties of the material, how much of the Shockley-Queisser
limit does a planar single-junction device made from it actually keep?

It is worth being clear about where the expression comes from, because it is
unlike the other two models in the package. Crovetto ran 2573 one-dimensional
drift-diffusion simulations of an idealised cell — fully carrier-selective
contacts, no interfaces, each absorber evaluated at its own optimal thickness
:math:`d_{\mathrm{opt}}` — and then fitted a closed-form expression to the
resulting efficiencies. The result is neither a derivation nor a black box: the
factors were chosen by hand and rationalised physically where possible, but the
coefficients are fits. Its accuracy against that training set is
:math:`R^2 = 0.995`, or :math:`\pm 3.4` percentage points on the fraction of
the SQ limit retained, and :math:`\pm 1.1` percentage points on the absolute
efficiency for a 1.5 eV gap.

The eight inputs
----------------

.. list-table::
   :header-rows: 1
   :widths: 10 22 18 22 28

   * - Symbol
     - Argument
     - Unit
     - Sampled range
     - Source
   * - :math:`E_{\mathrm{g}}`
     - ``E_gap``
     - eV
     - 0.7 – 2.0
     - :mod:`solphin.band_structure`, :mod:`solphin.dos`
   * - :math:`\alpha`
     - ``alpha``
     - cm\ :sup:`-1`
     - :math:`5\times10^{3}` – :math:`5\times10^{5}`
     - :func:`~solphin.spectral.calculate_spectral_average`
   * - :math:`\sigma`
     - ``sigma``
     - —
     - 0.2 – 1.8
     - :func:`~solphin.spectral.calculate_spectral_dispersion`
   * - :math:`m`
     - ``dos_mass``
     - :math:`m_0`
     - 0.12 – 2.5
     - :mod:`solphin.dos`
   * - :math:`\epsilon`
     - ``epsilon``
     - —
     - 1 – 100
     - Static dielectric constant, from the dielectric function or DFPT
   * - :math:`\tau`
     - ``tau``
     - s
     - :math:`10^{-15}` – :math:`10^{3}`
     - SRH lifetime — **user-supplied**
   * - :math:`n`
     - ``dop_density``
     - cm\ :sup:`-3`
     - :math:`10^{10}` – :math:`10^{18}`
     - Doping density — **user-supplied**
   * - :math:`\mu`
     - ``mu``
     - cm\ :sup:`2` V\ :sup:`-1` s\ :sup:`-1`
     - :math:`10^{-2}` – :math:`10^{9}`
     - Carrier mobility — **user-supplied**

The first five are calculable from the same electronic-structure run that feeds
:doc:`formalism_optics`. :math:`\alpha` and :math:`\sigma` are not the raw
:math:`\alpha(E)` curve but its photon-flux-weighted average and the weighted
standard deviation of :math:`\log\alpha`, taken from 300 nm to the band-gap
wavelength — equations (1) and (2) of the paper, implemented in
:mod:`solphin.spectral`.

The last three are the ones ``solphin`` cannot calculate. :math:`\tau`,
:math:`n` and :math:`\mu` are properties of a processed film rather than of the
crystal, and must come from measurement, from literature values for a
comparable material, or from a separate defect calculation. This is the central
trade-off of the whole approach: the figure of merit can price non-radiative
loss precisely because it declines to predict it.

Three definitions are easy to get wrong. :math:`\tau` is the bulk
Shockley-Read-Hall lifetime specifically, with :math:`\tau_{\mathrm{n}} =
\tau_{\mathrm{p}}` assumed and Auger recombination neglected. :math:`n` is the
majority-carrier concentration :math:`|N_{\mathrm{D}} - N_{\mathrm{A}}|`, not a
dopant count. :math:`m` is the *DOS* effective mass, in units of
:math:`m_0`, which is not the conductivity effective mass :math:`M` that
appears in :math:`\mu = q\tau_{\mathrm{s}}/M` — the two are derived
differently and are not interchangeable.

.. warning::

   Every property enters the expression in unitless form — divided by the unit
   given above — because the fitted factors take logarithms and fractional
   powers of them. This is why :math:`\alpha` must be in cm\ :sup:`-1` and not
   m\ :sup:`-1`: :func:`~solphin.optics.calc_absorption` returns
   m\ :sup:`-1`, but the ``absorption.dat`` file written by
   :func:`~solphin.optics.generate_absorption` is already in cm\ :sup:`-1`, and
   is the intended route into :mod:`solphin.spectral`.

.. warning::

   The sampled ranges above are the ranges over which the fit was trained.
   Crovetto notes that the efficiency prediction for an absorber falling
   outside them "may be grossly incorrect" — the expression will still return
   a number, and ``solphin`` will not warn you. Check your inputs against the
   table before trusting the output.

How the expression is put together
----------------------------------

The paper builds :math:`\Gamma_{\mathrm{PV}}` in stages, letting one more
property vary at each step and adding whatever factors are needed to keep the
simulated efficiencies collapsed onto a single curve. The group letters record
that history:

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Group
     - Adds
     - Stands for
   * - :math:`\mathcal{A}`
     - :math:`\alpha, \tau, \sigma`
     - **Absorption**
   * - :math:`\mathcal{D}`
     - :math:`n`
     - **Doping**
   * - :math:`\mathcal{T}`
     - :math:`\mu`
     - **Transport**
   * - :math:`\mathcal{S}`
     - —
     - **Saturation**

The final form, equation (18) of the paper, adds :math:`\epsilon`, :math:`m`
and a variable :math:`E_{\mathrm{g}}` and is what
:func:`~solphin.pv_fom.Final_equation` evaluates:

.. math::

   \Gamma_{\mathrm{PV}} = E_{\mathrm{g}}^{2.5}
       \left(\frac{\mathcal{A}_1 \mathcal{A}_2 \mathcal{D}_1}
                  {\mathcal{D}_2 \mathcal{D}_3 \mathcal{D}_4
                   \left(1 + \mathcal{T}_1 \mathcal{T}_2 \mathcal{T}_3\right)
                   \left(1 + \mathcal{S}_1 \mathcal{S}_2\right)}
       \right)^{E_{\mathrm{g}}^{-0.8}}

Larger is better. Each group is built so that it is inert — equal to one — for
a material in which that loss is not binding, and the two groups written as
:math:`1 + (\cdots)` can therefore only ever penalise, never reward. Evaluating
a group on its own tells you *which* property is holding a material back, and
every factor is exposed individually: ``_A_1_equation``, ``_D_3_equation``,
``_T_2_equation`` and the rest, matching equations (19)–(31) one for one.

.. note::

   The calligraphic factors above are not quite the :math:`A_i`, :math:`D_j`
   of the intermediate FOMs in the paper's equations (9), (12), (14) and (17).
   Those are the fixed-\ :math:`E_{\mathrm{g}}` versions; the ones implemented
   here are their generalisations, with re-fitted constants. The paper writes
   the generalised constants with bars for the same reason.

Absorption
----------

The numerator collects the terms that make a material better, and
:math:`\mathcal{A}_1` carries almost all of it:

.. math::

   \mathcal{A}_1 = \frac{0.295\,\tau\,
                \alpha^{\,\sigma^{-0.185\,E_{\mathrm{g}}^{0.5}}}}
              {m^{2}},
   \qquad
   \mathcal{A}_2 = 1 + \left(\frac{10^{-7}\,\sigma^{10}}
                        {\alpha\tau}\right)^{0.4}

:math:`\mathcal{A}_1` is essentially the :math:`\alpha\tau` figure of merit of
Kaienburg *et al.*, with :math:`\sigma` entering as an exponent-of-an-exponent
and :math:`m` dividing it out. The :math:`\sigma` dependence is the interesting
part: a broad spread of :math:`\log\alpha` means the generation profile is a
sum of exponentials with very different penetration depths, so a thicker film
is needed to absorb the same fraction of the spectrum, and the extra volume
recombines. This is why crystalline silicon scores below a chalcogenide
perovskite at equal :math:`\alpha\tau`. :math:`\mathcal{A}_2` is a correction
that only wakes up at low :math:`\alpha\tau`, and :math:`\mathcal{D}_1`,

.. math::

   \mathcal{D}_1 = \left(1 + \frac{4.4\times10^{-5}\,n}
                        {\epsilon^{0.8}\alpha^{2}}\right)
         ^{0.22\log(\alpha/39)}

sits in the numerator because doping raises the quasi-Fermi level splitting,
and with it :math:`V_{\mathrm{oc}}`, under low-injection conditions. Its form
also sets the threshold at which doping starts to matter at all:
:math:`n \gtrsim \alpha^{2}/\bar{d}_1`.

Doping
------

The same doping density is penalised three times over, once for each way it can
go wrong:

.. math::

   \mathcal{D}_2 &= \left(1 + \frac{10^{-21}n}{\alpha^{2}\tau}\right)
          ^{0.05E_{\mathrm{g}}^{4}} \\[6pt]
   \mathcal{D}_3 &= 1 + \frac{\left(2.1\times10^{4}\,E_{\mathrm{g}}^{8.5}\,\tau\,
                          \alpha^{\,0.68E_{\mathrm{g}}^{-1.5}}\right)
                    ^{\log(10n/\epsilon)/50}}
                   {1 + 10^{(E_{\mathrm{g}}-1.5)/0.1}} \\[6pt]
   \mathcal{D}_4 &= 1 + \left(\frac{7.7\times10^{-7}}
                         {E_{\mathrm{g}}^{17}\,\alpha\,\tau}\right)^{0.6}

The competition :math:`\mathcal{D}_1` is up against is a
:math:`J_{\mathrm{sc}}` effect. At low enough :math:`n` the absorber is fully
depleted at short circuit, so an electric field extends through it and carriers
are collected by drift rather than diffusion alone. For a poor absorber — low
:math:`\mu`, low :math:`\tau`, low :math:`\alpha` — that collection gain
outweighs the :math:`V_{\mathrm{oc}}` a higher :math:`n` would have bought. The
optimum is therefore genuinely interior, which is why the figure of merit is
non-monotonic in :math:`n` alone and why
:func:`~solphin.final_results.plot_FOM` sweeps it rather than reporting a
single value.

:math:`\mathcal{D}_3` is worth reading closely for its denominator, a sigmoid in
:math:`E_{\mathrm{g}}` centred on 1.5 eV with a width of 0.1 eV: below that gap
the term is live, above it the switch closes and :math:`\mathcal{D}_3 \to 1`.
:math:`\mathcal{T}_3''` uses the same construction with a width of 0.01 eV.
:math:`\mathcal{D}_4`, by contrast, diverges as :math:`\alpha\tau \to 0` and is
the correction for absorbers that are poor on both counts at once.

Transport
---------

The transport group is where a slow, heavy carrier is charged for the current
it fails to collect:

.. math::

   \mathcal{T}_1 &= \frac{0.051\,(E_{\mathrm{g}}+0.5)^{11}}
               {m\,\epsilon^{0.5}\,
                \mu^{\,0.046E_{\mathrm{g}}^{4.3}+0.9}} \\[6pt]
   \mathcal{T}_2 &= \left(\frac{1 + 9.5\times10^{-18}n
                      \left(1+E_{\mathrm{g}}^{5.1}\right)}
                     {7.8\times10^{7}\,\tau\,10^{\,0.5/\sigma}}
         \right)^{0.47E_{\mathrm{g}}^{1.25}} \\[6pt]
   \mathcal{T}_3 &= \left(1 + \mathcal{T}_3' \mathcal{T}_3''\right)
          ^{1.6\times10^{-3}E_{\mathrm{g}}^{8}+0.6}

with :math:`\mathcal{T}_3` split across ``_T_3_prime_equation`` and
``_T_3_double_prime_equation``:

.. math::

   \mathcal{T}_3' &= 1.9\times10^{4}
           \left(1+9.5\times10^{-3}E_{\mathrm{g}}^{10}\right)
           \exp\!\left(-0.1\,
           \alpha^{\,0.5/(1+2.4\times10^{-4}E_{\mathrm{g}}^{10})}\right) \\[6pt]
   \mathcal{T}_3'' &= 1 + \frac{\dfrac{0.16}{m^{3}}
                      \left(\dfrac{n\,m^{3}}{0.16}\right)
                      ^{1-0.74\exp(-\alpha/1.5\times10^{5})}}
                     {1 + 10^{(E_{\mathrm{g}}-1.5)/0.01}}

Because the group enters as :math:`1 + \mathcal{T}_1\mathcal{T}_2\mathcal{T}_3`
and :math:`\mathcal{T}_1 \propto \mu^{-(0.9 + \cdots)}` while
:math:`\mathcal{T}_2 \propto \tau^{-0.47E_{\mathrm{g}}^{1.25}}`, mobility and
lifetime trade against one another and the penalty bites only when a material
is poor on both. The exponential in :math:`\mathcal{T}_3'` is a strong-absorber
cut-off: for :math:`\alpha` of order 10\ :sup:`5` cm\ :sup:`-1` it has already
collapsed, because a film that absorbs in a short distance never asks much of
transport in the first place.

The practical consequence, and the paper's main argument against simpler
figures of merit, is that there is no universal mobility below which transport
starts to matter. The threshold is a function of every other property — the
paper writes it out as its equation (15) — and it varies by orders of magnitude
between materials.

Saturation
----------

The last group is the one that is easiest to misread from the code alone.
:math:`\mathcal{S}` stands for *saturation*, and it exists because a real
absorber's efficiency stops improving before it reaches the SQ limit:

.. math::

   \mathcal{S}_1 = \frac{10^{2.4E_{\mathrm{g}}}\,\alpha^{0.75}\,\tau}
              {2.4\times10^{4}\,m^{2}\,
               \mu^{\,1/(1+4\times10^{-4}E_{\mathrm{g}}^{10})}},
   \qquad
   \mathcal{S}_2 = 1 + \left(\frac{4.8\times10^{3}}{\alpha}\right)^{20}
             \mu^{0.5}\log n

The diffusion length is ultimately capped by the *radiative* lifetime. Once the
SRH lifetime :math:`\tau` has grown past it, lengthening :math:`\tau` further
buys nothing, and if the diffusion length is still shorter than the absorption
depth — the combination of low :math:`\mu` and low :math:`\alpha` — the SQ
limit is never reached at any thickness. :math:`\mathcal{S}_1` grows with
:math:`\tau` for exactly this reason: it is the brake that stops the numerator's
:math:`\mathcal{A}_1 \propto \tau` from rewarding a lifetime the device cannot
use. :math:`\mathcal{S}_2` is the switch that decides when the brake engages,
and its twentieth power makes it effectively a step at
:math:`\alpha \approx 4.8\times10^{3}` cm\ :sup:`-1` — the bottom of the
sampled absorption range. For any decent absorber :math:`\mathcal{S}_2 = 1`.

Worked example
--------------

The paper's own worked example is methylammonium lead iodide, with the
properties of state-of-the-art films (its table 2), and ``solphin`` reproduces
it. The reference block is quoted in cell 40 of the full workflow tutorial:

.. code-block:: python

   E_gap = 1.55; alpha = 9.9e4; sigma = 0.63; dos_mass = 0.15
   tau = 6.9e-7; mu = 10; dop_density = 1e12; epsilon = 33.5

These give :math:`\Gamma_{\mathrm{PV}} = 4.77`, from a numerator of 3.26
against :math:`\mathcal{D} = 1.54`,
:math:`1 + \mathcal{T}_1\mathcal{T}_2\mathcal{T}_3 = 1.09` and
:math:`1 + \mathcal{S}_1\mathcal{S}_2 = 1.004`. Almost every penalty is
dormant, which is the expression agreeing that MAPI is a good absorber.

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - MAPI case
     - :math:`\Gamma_{\mathrm{PV}}`
     - ``solphin`` :math:`\eta_\Gamma`
     - Crovetto table 2
   * - State of the art
     - 4.77
     - 26.81%
     - :math:`(26.8 \pm 1.1)\%`
   * - :math:`\tau` lowered to 10 ns
     - 0.121
     - 20.35%
     - :math:`(20.4 \pm 1.1)\%`
   * - and :math:`\mu` lowered to 0.1
     - 0.0030
     - 13.44%
     - :math:`(13.5 \pm 1.1)\%`

The two degraded cases show the groups doing their job. Cutting :math:`\tau`
lands mostly in the numerator, since :math:`\mathcal{A}_1 \propto \tau`, which
falls from 3.26 to 0.047; the transport group only rises from 1.09 to 3.74.
Lowering :math:`\mu` on top of that leaves the numerator untouched and drives
:math:`\mathcal{T}` to 699. Same scalar output, different diagnosis.

From the figure of merit to an efficiency
-----------------------------------------

:func:`~solphin.final_results.SQ_relative_FOM_PV_efficiency` implements
equation (33), which maps :math:`\Gamma_{\mathrm{PV}}` onto the fraction of the
detailed-balance limit that survives:

.. math::

   \eta_\Gamma = \frac{\eta_{\mathrm{sq}}}
               {\left(1 + \dfrac{0.330\,\Gamma_{\mathrm{PV}}^{-0.235}}
                                {1 + 0.0906\,\Gamma_{\mathrm{PV}}^{0.869}}
                \right)
                \left(1 + 2.48\times10^{-3}\,
                      \Gamma_{\mathrm{PV}}^{-0.362}\right)}

:math:`\eta_{\mathrm{sq}}` is the Shockley-Queisser limit for the same gap and
spectrum, from :func:`~solphin.db_fom.max_eff`, so the function returns the SQ
limit, the realistic efficiency and their ratio together. The map is monotonic
and saturating: :math:`\Gamma_{\mathrm{PV}} = 1` retains 76.6% of the SQ limit,
MAPI's 4.77 retains 85.4%, :math:`\Gamma_{\mathrm{PV}} = 100` retains 98.1%,
and an ideal material recovers :math:`\eta_{\mathrm{sq}}` exactly. Over the
intermediate decades the efficiency is roughly logarithmic in
:math:`\Gamma_{\mathrm{PV}}`, which is the pedagogical piecewise form given as
the paper's equation (32).

What it does not model
----------------------

:math:`\Gamma_{\mathrm{PV}}` is the only one of the three models in ``solphin``
that prices :math:`\tau`, :math:`\mu` and :math:`n`, and the price it charges
is only as good as the three numbers you feed it. Beyond that, the paper is
explicit about its own limits, and they are worth stating:

* It predicts efficiency only. :math:`J_{\mathrm{sc}}`,
  :math:`V_{\mathrm{oc}}`, fill factor and the optimal thickness
  :math:`d_{\mathrm{opt}}` are not recoverable from
  :math:`\Gamma_{\mathrm{PV}}` — for a thickness answer, use the sweep in
  :doc:`formalism_optics`.
* The eight properties are treated as independent, and they are not. A heavier
  :math:`m` generally means lower :math:`\sigma` and :math:`\mu`; a larger
  :math:`\epsilon` tends to come with longer :math:`\tau` and higher
  :math:`\mu`. Varying one input while holding the rest fixed will therefore
  overstate its effect, sometimes in the wrong direction.
* It is a planar single-junction model obeying the SQ limit, so it says nothing
  about tandems, textured or nanostructured devices, contact and interface
  recombination, or series and shunt resistance.
* Properties are assumed isotropic and equal for electrons and holes
  (:math:`\tau_{\mathrm{n}} = \tau_{\mathrm{p}}`,
  :math:`\mu_{\mathrm{n}} = \mu_{\mathrm{p}}`,
  :math:`m_{\mathrm{e}} = m_{\mathrm{h}}`). A strongly anisotropic or
  strongly carrier-asymmetric absorber has no unambiguous input values — the
  same caveat that attaches to the averaged :math:`\alpha(E)` in
  :doc:`formalism_optics`.
* Sub-band-gap absorption — excitonic, defect-mediated, or from potential
  fluctuations — is not represented, and :math:`E_{\mathrm{g}}` is assumed
  equal to the optical absorption onset.

.. [Crovetto2024] A. Crovetto, *A phenomenological figure of merit for
   photovoltaic materials*, J. Phys. Energy **6**, 025009 (2024).
   https://doi.org/10.1088/2515-7655/ad2499
