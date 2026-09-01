"""The Crovetto 2024 photovoltaic figure of merit.

pv_fom is pure numpy with no I/O, so everything here is an algebraic identity or
a physical monotonicity - no fixtures and no pinned numbers.

The reference parameter set is the MAPI block the authors themselves wrote into
tutorial cell 40, rather than values invented for the tests.
"""

import inspect
import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest

import solphin.pv_fom as pv_fom
from solphin.pv_fom import (
    SAMPLED_RANGES,
    Final_equation,
    _Final_D_denominator,
    _Final_numerator,
    _Final_S_denominator,
    _Final_T_denominator,
    check_sampled_ranges,
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


def _call_with(func: Callable[..., float], **overrides: float) -> float:
    """Invoke func with the MAPI values its signature happens to ask for.

    Arguments MAPI has no value for - allow_out_of_range, say - are left to
    their defaults rather than being passed, so a new keyword on a component
    does not have to be added here to keep the sweeps working.
    """
    params = {**MAPI, **overrides}
    wanted = inspect.signature(func).parameters
    return func(**{name: params[name] for name in wanted if name in params})


def _sweep(parameter: str, values: Sequence[float]) -> list[float]:
    return [_call_with(Final_equation, **{parameter: v}) for v in values]


def test_final_equation_composition() -> None:
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


def test_denominator_components_exceed_one() -> None:
    """The T and S groups are built as 1 + (...), so both exceed unity."""
    assert _call_with(_Final_T_denominator) > 1.0
    assert _call_with(_Final_S_denominator) > 1.0


@pytest.mark.parametrize("name", COMPONENT_NAMES)
def test_all_components_positive_and_finite(name: str) -> None:
    """Every Γₚᵥ component evaluates to a positive finite number."""
    value = _call_with(getattr(pv_fom, name))

    assert math.isfinite(value)
    assert value > 0


def test_scalar_return_type() -> None:
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


def test_increases_with_lifetime() -> None:
    """A longer carrier lifetime raises Γₚᵥ."""
    assert np.all(np.diff(_sweep("tau", [1e-9, 1e-8, 1e-7, 1e-6])) > 0)


def test_increases_with_mobility() -> None:
    """Higher carrier mobility raises Γₚᵥ."""
    assert np.all(np.diff(_sweep("mu", [1e0, 1e2, 1e4, 1e6])) > 0)


def test_increases_with_absorption() -> None:
    """Stronger absorption raises Γₚᵥ.

    The sweep stays inside the sampled range of table 1: outside it the paper
    claims no monotonicity, and the guard would refuse the values anyway.
    """
    assert np.all(np.diff(_sweep("alpha", [5e3, 5e4, 5e5])) > 0)


def test_decreases_with_dispersion() -> None:
    """A broader spread of log(α) lowers Γₚᵥ."""
    assert np.all(np.diff(_sweep("sigma", [0.2, 0.6, 1.2, 1.8])) < 0)


# --- the sampled ranges of Crovetto 2024 table 1 --------------------------
#
# Γₚᵥ is a fit, and table 1 is the box it was fitted inside. The guard is what
# stops a property from outside that box being priced as though the expression
# knew anything about it.


def test_sampled_ranges_matches_table_1() -> None:
    """The constant is table 1 of Crovetto 2024, transcribed."""
    assert SAMPLED_RANGES == {
        "E_gap": (0.7, 2.0, "eV"),
        "alpha": (5e3, 5e5, "cm⁻¹"),
        "sigma": (0.2, 1.8, ""),
        "tau": (1e-15, 1e3, "s"),
        "mu": (1e-2, 1e9, "cm² V⁻¹ s⁻¹"),
        "dop_density": (1e10, 1e18, "cm⁻³"),
        "epsilon": (1.0, 100.0, ""),
        "dos_mass": (0.12, 2.5, "m₀"),
    }


def test_reference_set_is_inside_the_sampled_ranges() -> None:
    """MAPI is the paper's own worked example, so it must clear the guard."""
    for name, value in MAPI.items():
        minimum, maximum, _ = SAMPLED_RANGES[name]

        assert minimum <= value <= maximum, name


@pytest.mark.parametrize("name", sorted(SAMPLED_RANGES))
def test_below_the_range_raises(name: str) -> None:
    """Each property is refused below its lower bound, and is named for it."""
    minimum, _, _ = SAMPLED_RANGES[name]

    with pytest.raises(ValueError, match=name):
        _call_with(Final_equation, **{name: minimum * 0.5})


@pytest.mark.parametrize("name", sorted(SAMPLED_RANGES))
def test_above_the_range_raises(name: str) -> None:
    """Each property is refused above its upper bound, and is named for it."""
    _, maximum, _ = SAMPLED_RANGES[name]

    with pytest.raises(ValueError, match=name):
        _call_with(Final_equation, **{name: maximum * 2.0})


@pytest.mark.parametrize("name", sorted(SAMPLED_RANGES))
def test_both_endpoints_are_accepted(name: str) -> None:
    """The intervals are closed: table 1 gives values the dataset attained."""
    for endpoint in SAMPLED_RANGES[name][:2]:
        gamma = _call_with(Final_equation, **{name: endpoint})

        assert math.isfinite(gamma)
        assert gamma > 0


def test_every_corner_of_the_box_evaluates() -> None:
    """No corner of the sampled box is a hole in the expression.

    2^8 corners, so a bound that happens to divide by zero or take the log of
    something non-positive shows up here rather than in a user's notebook.
    """
    names = sorted(SAMPLED_RANGES)
    bounds = [SAMPLED_RANGES[name][:2] for name in names]

    for corner in itertools.product(*bounds):
        gamma = Final_equation(**dict(zip(names, corner)))

        assert math.isfinite(gamma)
        assert gamma > 0


def test_out_of_range_flag_warns_and_still_evaluates() -> None:
    """The paper allows that Γₚᵥ may survive outside table 1, so opting in works."""
    with pytest.warns(UserWarning, match="alpha"):
        gamma = _call_with(Final_equation, alpha=1e3, allow_out_of_range=True)

    assert math.isfinite(gamma)
    assert gamma > 0


def test_out_of_range_flag_does_not_change_the_value() -> None:
    """The flag gates the check, not the arithmetic."""
    guarded = _call_with(Final_equation)
    permitted = _call_with(Final_equation, allow_out_of_range=True)

    assert guarded == pytest.approx(permitted, rel=1e-12)


def test_every_violation_is_reported_at_once() -> None:
    """A property set in the wrong units breaks several bounds together."""
    with pytest.raises(ValueError) as excinfo:
        _call_with(Final_equation, alpha=1e3, sigma=5.0, dos_mass=0.01)

    message = str(excinfo.value)

    assert "alpha" in message
    assert "sigma" in message
    assert "dos_mass" in message


def test_message_carries_the_value_the_bound_and_the_way_out() -> None:
    """The error has to be actionable without opening the paper."""
    with pytest.raises(ValueError) as excinfo:
        _call_with(Final_equation, dos_mass=0.05)

    message = str(excinfo.value)

    assert "0.05" in message
    assert "0.12" in message
    assert "m₀" in message
    assert "allow_out_of_range=True" in message


def test_nan_is_refused() -> None:
    """NaN fails both comparisons, so it is caught rather than propagated."""
    with pytest.raises(ValueError, match="tau"):
        _call_with(Final_equation, tau=float("nan"))


def test_check_sampled_ranges_ignores_unknown_keys() -> None:
    """Callers pass whole argument sets through, Tcell and all."""
    check_sampled_ranges(**MAPI, Tcell=300.0, photon_spectrum=None)
