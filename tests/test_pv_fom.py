"""The Crovetto 2024 photovoltaic figure of merit.

pv_fom is pure numpy with no I/O, so everything here is an algebraic identity or
a physical monotonicity - no fixtures and no pinned numbers.

The reference parameter set is the MAPI block the authors themselves wrote into
tutorial cell 40, rather than values invented for the tests.
"""

import inspect
import math

import numpy as np
import pytest

import solphin.pv_fom as pv_fom
from solphin.pv_fom import (
    Final_equation,
    _Final_D_denominator,
    _Final_numerator,
    _Final_S_denominator,
    _Final_T_denominator,
)

# Methylammonium lead iodide, from the commented reference block in the tutorial.
MAPI = {
    "E_gap": 1.55,  # eV
    "alpha": 9.9e4,  # cm^-1
    "tau": 6.9e-7,  # s
    "sigma": 0.63,  # unitless
    "dos_mass": 0.15,  # m_0
    "dop_density": 1e12,  # cm^-3
    "epsilon": 33.5,  # unitless
    "mu": 10.0,  # cm^2 V^-1 s^-1
}

# Every closed-form component, discovered rather than listed, so a new helper is
# covered as soon as it is added.
COMPONENT_NAMES = sorted(
    name
    for name in dir(pv_fom)
    if name.endswith("_equation") and name != "Final_equation"
)


def _call_with(func, **overrides):
    """Invoke func with the MAPI values its signature happens to ask for."""
    params = {**MAPI, **overrides}
    wanted = inspect.signature(func).parameters
    return func(**{name: params[name] for name in wanted})


def _sweep(parameter, values):
    return [_call_with(Final_equation, **{parameter: v}) for v in values]


def test_final_equation_composition():
    """Gamma = E_gap^2.5 * (numerator / (D * T * S)) ^ (E_gap^-0.8)."""
    numerator = _call_with(_Final_numerator)
    denominator = (
        _call_with(_Final_D_denominator)
        * _call_with(_Final_T_denominator)
        * _call_with(_Final_S_denominator)
    )
    expected = MAPI["E_gap"] ** 2.5 * (numerator / denominator) ** (
        MAPI["E_gap"] ** -0.8
    )

    assert _call_with(Final_equation) == pytest.approx(expected, rel=1e-12)


def test_denominator_components_exceed_one():
    """The T and S groups are built as 1 + (...), so both exceed unity."""
    assert _call_with(_Final_T_denominator) > 1.0
    assert _call_with(_Final_S_denominator) > 1.0


@pytest.mark.parametrize("name", COMPONENT_NAMES)
def test_all_components_positive_and_finite(name):
    value = _call_with(getattr(pv_fom, name))

    assert math.isfinite(value)
    assert value > 0


def test_scalar_return_type():
    """Callers unpack this straight into an f-string, so it must be a real scalar."""
    gamma = _call_with(Final_equation)

    assert np.isscalar(gamma) or np.ndim(gamma) == 0
    assert math.isfinite(float(gamma))


# --- physical monotonicities ----------------------------------------------
#
# Longer-lived, more mobile carriers and stronger absorption all make a better
# absorber; a broader spread of log(alpha) makes a worse one. Doping density is
# deliberately absent - the figure of merit is genuinely non-monotone in it,
# since an optimal doping level exists.


def test_increases_with_lifetime():
    assert np.all(np.diff(_sweep("tau", [1e-9, 1e-8, 1e-7, 1e-6])) > 0)


def test_increases_with_mobility():
    assert np.all(np.diff(_sweep("mu", [1e0, 1e2, 1e4, 1e6])) > 0)


def test_increases_with_absorption():
    assert np.all(np.diff(_sweep("alpha", [1e3, 1e4, 1e5])) > 0)


def test_decreases_with_dispersion():
    assert np.all(np.diff(_sweep("sigma", [0.1, 0.5, 1.0, 2.0])) < 0)
