"""End-to-end checks of the efficiency estimate against Crovetto 2024, Table 2.

The paper works its MAPI example all the way from the eight bulk properties
to the efficiency limit, publishing every intermediate number. Reproducing
them pins the whole chain - Shockley-Queisser limit, Gamma_PV figure of
merit, and the equation-(33) fit with its supplementary-material constants -
against the article rather than against this implementation.
"""

from typing import Any

import pytest
from numpy.typing import NDArray

import solphin.final_results as final_results

# CH3NH3PbI3 (MAPI) property set from Table 2 of the paper; the same block is
# quoted in tutorial cell 40 and reused by test_pv_fom.
#
# dict[str, Any] rather than the inferred dict[str, float]: the set is splatted
# into SQ_relative_FOM_PV_efficiency, whose keyword-only allow_out_of_range is
# a bool, and mypy checks a **dict's value type against every parameter the
# dict could reach.
MAPI: dict[str, Any] = {
    "E_gap": 1.55,  # eV
    "alpha": 9.9e4,  # cm^-1
    "tau": 6.9e-7,  # s
    "sigma": 0.63,  # unitless
    "dos_mass": 0.15,  # m_0
    "dop_density": 1e12,  # cm^-3
    "epsilon": 33.5,  # unitless
    "mu": 10.0,  # cm^2 V^-1 s^-1
}


def test_mapi_reproduces_paper_table_2(photon_spectrum: NDArray) -> None:
    """MAPI's published property set gives the published 31.4 %, 85.4 % and 26.8 %."""
    SQ, SQ_relative, efficiency = final_results.SQ_relative_FOM_PV_efficiency(
        photon_spectrum=photon_spectrum, Tcell=300.0, **MAPI
    )

    assert SQ == pytest.approx(31.4, abs=0.05)
    assert SQ_relative == pytest.approx(85.4, abs=0.1)
    assert efficiency == pytest.approx(26.8, abs=0.1)


def test_degraded_mapi_reproduces_paper(photon_spectrum: NDArray) -> None:
    """The paper's 'bad MAPI' variants: tau = 10 ns gives 20.4 %; mu = 0.1 too, 13.5 %."""
    _, _, short_lifetime = final_results.SQ_relative_FOM_PV_efficiency(
        photon_spectrum=photon_spectrum, Tcell=300.0, **{**MAPI, "tau": 10e-9}
    )
    _, _, low_mobility = final_results.SQ_relative_FOM_PV_efficiency(
        photon_spectrum=photon_spectrum, Tcell=300.0, **{**MAPI, "tau": 10e-9, "mu": 0.1}
    )

    assert short_lifetime == pytest.approx(20.4, abs=0.1)
    assert low_mobility == pytest.approx(13.5, abs=0.1)


# --- the table 1 guard on the efficiency entry point ----------------------


def test_efficiency_refuses_a_property_outside_table_1(
        photon_spectrum: NDArray
) -> None:
    """The guard reaches through to the function the tutorials actually call."""
    with pytest.raises(ValueError, match="dos_mass"):
        final_results.SQ_relative_FOM_PV_efficiency(
            photon_spectrum=photon_spectrum, Tcell=300.0, **{**MAPI, "dos_mass": 0.05}
        )


def test_efficiency_opt_out_warns_and_returns(photon_spectrum: NDArray) -> None:
    """allow_out_of_range is keyword-only, and downgrades the refusal."""
    with pytest.warns(UserWarning, match="dos_mass"):
        _, _, efficiency = final_results.SQ_relative_FOM_PV_efficiency(
            photon_spectrum=photon_spectrum,
            Tcell=300.0,
            allow_out_of_range=True,
            **{**MAPI, "dos_mass": 0.05},
        )

    assert 0 < efficiency < 100


def test_efficiency_still_takes_ten_positional_arguments(
        photon_spectrum: NDArray
) -> None:
    """The tutorials call this positionally, so the new keyword must not shift them."""
    positional = final_results.SQ_relative_FOM_PV_efficiency(
        MAPI["E_gap"], photon_spectrum, MAPI["alpha"], MAPI["tau"], MAPI["sigma"],
        MAPI["dos_mass"], MAPI["dop_density"], MAPI["epsilon"], MAPI["mu"], 300.0,
    )
    keyword = final_results.SQ_relative_FOM_PV_efficiency(
        photon_spectrum=photon_spectrum, Tcell=300.0, **MAPI
    )

    assert positional == keyword
