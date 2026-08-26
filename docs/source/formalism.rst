Mathematical Formalism
======================

``solphin`` evaluates three efficiency limits for the same absorber. All
three are detailed-balance treatments of a planar single-junction cell under
a reference spectrum; they differ in which idealisations they keep. Each is
stricter than the one before it, and each asks more of the material data.

.. list-table::
   :header-rows: 1
   :widths: 20 36 22 22

   * - Limit
     - What the absorber is allowed to do
     - Material input
     - Module
   * - Shockley-Queisser, :math:`\eta_{\mathrm{SQ}}`
     - Absorb every photon above the gap and none below it; lose carriers
       only radiatively; collect every carrier
     - :math:`E_{\mathrm{g}}`
     - :mod:`solphin.db_fom`
   * - Blank and SLME, :math:`\eta(d)`
     - Absorb according to its own :math:`\alpha(E)` in a film of finite
       thickness :math:`d`; recombine non-radiatively at a fixed rate
     - :math:`\alpha(E)`, :math:`n(E)`,
       :math:`E_{\mathrm{g}}^{\mathrm{d}}`, :math:`E_{\mathrm{g}}^{\mathrm{i}}`
     - :mod:`solphin.optics`
   * - :math:`\Gamma_{\mathrm{PV}}` limit, :math:`\eta_\Gamma`
     - Also collect carriers imperfectly, at finite lifetime, mobility,
       doping density and dielectric screening
     - eight bulk properties
     - :mod:`solphin.pv_fom`

The Shockley-Queisser limit is a function of the band gap alone, so it is the
same number for every material sharing that gap. It sets the ceiling the
other two are measured against: both the Blank/SLME efficiency and
:math:`\eta_\Gamma` are reported by ``solphin`` in absolute terms and as a
fraction of :math:`\eta_{\mathrm{SQ}}` at the same gap, which is what
separates "this material has an awkward band gap" from "this material has a
defect problem".

The two lower limits are complementary rather than competing. The
Blank/SLME calculation is *deductive*: it follows from the calculated
optical constants and the physics of radiative balance, and its only
non-radiative input is a single empirical factor. The
:math:`\Gamma_{\mathrm{PV}}` figure of merit is *inductive*: it is a
closed-form fit to a large set of drift-diffusion simulations, so it can
price in effects that resist deductive treatment — imperfect carrier
collection above all — at the cost of being valid only over the property
ranges it was fitted on.

.. toctree::
   :maxdepth: 1

   formalism_detailed_balance.rst
   formalism_optics.rst
   formalism_pv_fom.rst

Conventions
-----------

Throughout these pages, :math:`h` is Planck's constant, :math:`\hbar =
h/2\pi`, :math:`c` the speed of light, :math:`k_{\mathrm{B}}` Boltzmann's
constant, :math:`q` the elementary charge, :math:`m_0` the free electron
mass and :math:`T` the cell temperature. Photon energies are written
:math:`E` and wavelengths :math:`\lambda`, with :math:`E = hc/\lambda`.
Physical constants are taken from ``scipy.constants`` rather than
hard-coded.

The default reference spectrum is ASTM G173-03 AM1.5G, bundled with the
package. :func:`solphin.db_fom.load_spectrum` also ships fluorescent, white,
blue, green, red and infrared LED spectra and the photopic response curve,
for indoor-photovoltaic work; every formula below applies unchanged to those.
