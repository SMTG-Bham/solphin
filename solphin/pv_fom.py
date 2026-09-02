"""Components of the Γₚᵥ photovoltaic figure of merit.

References
----------
Andrea Crovetto, 2024, J. Phys. Energy 6 025009.
"""

import warnings

import numpy as np

__all__ = ["SAMPLED_RANGES", "Final_equation", "check_sampled_ranges"]

# Sampled ranges of the eight bulk properties in the Γₚᵥ training set, table 1 of
# Crovetto 2024. Keyed by the argument name each property takes, valued as
# (minimum, maximum, unit). The intervals are closed: the table caption defines
# these as the minimum and the maximum value each property attains in the ~2573
# point ηsim-versus-P dataset, so the endpoints are themselves trained input.
#
# The units are the ones the paper's Methods section divides each property by to
# reach the unitless form the fitted factors take logarithms and fractional
# powers of, so they are the units the arguments must already be in.
SAMPLED_RANGES: dict[str, tuple[float, float, str]] = {
    "E_gap": (0.7, 2.0, "eV"),
    "alpha": (5e3, 5e5, "cm⁻¹"),
    "sigma": (0.2, 1.8, ""),
    "tau": (1e-15, 1e3, "s"),
    "mu": (1e-2, 1e9, "cm² V⁻¹ s⁻¹"),
    "dop_density": (1e10, 1e18, "cm⁻³"),
    "epsilon": (1.0, 100.0, ""),
    "dos_mass": (0.12, 2.5, "m₀"),
}


def check_sampled_ranges(
        allow_out_of_range: bool = False, **properties: float
) -> None:
    """Check Γₚᵥ inputs against the sampled ranges of Crovetto 2024 table 1.

    Γₚᵥ is a fit, not a derivation, and table 1 records the range of each
    property across the ~2573 drift-diffusion simulations it was fitted to.
    The table caption warns that the efficiency prediction for an absorber
    falling outside those ranges "may be grossly incorrect", so out-of-range
    input raises by default. The paper's discussion also allows that Γₚᵥ "may
    remain sufficiently accurate even when some of the properties fall outside
    the ranges in table 1", which is what ``allow_out_of_range`` is for.

    Every violation is reported together rather than one at a time: a property
    set supplied in the wrong units usually breaks several bounds at once, and
    one message naming all of them saves the caller as many round trips.

    Parameters
    ----------
    allow_out_of_range : bool, optional
        If True, an out-of-range value raises a ``UserWarning`` instead of a
        ``ValueError`` and evaluation continues. Default is False.
    **properties : float
        Property values keyed by the argument names of
        :data:`SAMPLED_RANGES`. Keys outside that mapping are ignored, so a
        caller may pass a whole argument set through.

    Raises
    ------
    ValueError
        If any property lies outside its table 1 range, or is not a number,
        and ``allow_out_of_range`` is False.

    Warns
    -----
    UserWarning
        The same conditions, when ``allow_out_of_range`` is True.
    """
    violations = []

    for name, value in properties.items():

        if name not in SAMPLED_RANGES:
            continue

        minimum, maximum, unit = SAMPLED_RANGES[name]

        # NaN fails both comparisons, so it is reported as out of range rather
        # than slipping through to produce a silent NaN figure of merit.
        if minimum <= value <= maximum:
            continue

        unit_text = f" {unit}" if unit else ""

        violations.append(
            f"{name} = {value:.3g}{unit_text} is outside the"
            f" {minimum:.3g} - {maximum:.3g}{unit_text} range sampled by"
            " Crovetto 2024 table 1"
        )

    if not violations:
        return

    detail = "; ".join(violations)

    message = (
        f"{detail}; the Γₚᵥ fit was not trained there, so the efficiency it"
        " predicts may be grossly incorrect."
    )

    if allow_out_of_range:
        # Three frames up: this function, the solphin function that called it,
        # and the caller's own code, which is where the offending value came
        # from and so where the warning should point.
        warnings.warn(message, UserWarning, stacklevel=3)

        return

    raise ValueError(
        f"{message} Pass allow_out_of_range=True to evaluate anyway."
    )


def Final_equation(
        E_gap: float, alpha: float, tau: float, sigma: float, dos_mass: float, dop_density: float,
        epsilon: float, mu: float, *, allow_out_of_range: bool = False
) -> float:
    """Calculate the total Γₚᵥ photovoltaic figure of merit from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.
    sigma : float
        Spectral dispersion of the absorption coefficient, dimensionless.
    dos_mass : float
        Density-of-states effective mass in units of m₀.
    dop_density : float
        Doping density in cm⁻³.
    epsilon : float
        Static dielectric constant, dimensionless.
    mu : float
        Charge carrier mobility in cm² V⁻¹ s⁻¹.
    allow_out_of_range : bool, optional
        If True, a property outside its Crovetto 2024 table 1 sampled range
        warns instead of raising, and the figure of merit is evaluated anyway.
        Default is False. Keyword-only.

    Returns
    -------
    float
        Γₚᵥ photovoltaic figure of merit, dimensionless.

    Raises
    ------
    ValueError
        If any property lies outside its table 1 range in
        :data:`SAMPLED_RANGES` and ``allow_out_of_range`` is False.

    Warns
    -----
    UserWarning
        The same conditions, when ``allow_out_of_range`` is True.
    """
    # Every private factor below is reached only through this function, so this
    # is the one place the eight properties have to be checked.
    check_sampled_ranges(
        allow_out_of_range,
        E_gap=E_gap,
        alpha=alpha,
        tau=tau,
        sigma=sigma,
        dos_mass=dos_mass,
        dop_density=dop_density,
        epsilon=epsilon,
        mu=mu,
    )

    E_gap_2_5 = E_gap ** 2.5
    E_gap_0_8 = E_gap ** -0.8

    D_denominator = _Final_D_denominator(E_gap, alpha, tau, dop_density, epsilon)
    T_denominator = _Final_T_denominator(E_gap, alpha, tau, sigma, dos_mass, dop_density, epsilon, mu)
    S_denominator = _Final_S_denominator(E_gap, alpha, tau, dos_mass, dop_density, mu)

    numerator = _Final_numerator(E_gap, alpha, tau, sigma, dos_mass, dop_density, epsilon)

    denominator = D_denominator * T_denominator * S_denominator

    PV_FOM = E_gap_2_5 * ((numerator / denominator) ** E_gap_0_8)

    return PV_FOM


def _Final_numerator(
        E_gap: float, alpha: float, tau: float, sigma: float, dos_mass: float, dop_density: float,
        epsilon: float
) -> float:
    """Calculate the numerator of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.
    sigma : float
        Spectral dispersion of the absorption coefficient, dimensionless.
    dos_mass : float
        Density-of-states effective mass in units of m₀.
    dop_density : float
        Doping density in cm⁻³.
    epsilon : float
        Static dielectric constant, dimensionless.

    Returns
    -------
    float
        The A₁A₂D₁ numerator of Γₚᵥ, dimensionless.
    """
    A_1 = _A_1_equation(E_gap, alpha, tau, sigma, dos_mass)
    A_2 = _A_2_equation(alpha, tau, sigma)
    D_1 = _D_1_equation(alpha, dop_density, epsilon)

    numerator = A_1 * A_2 * D_1

    return numerator


def _Final_D_denominator(
        E_gap: float, alpha: float, tau: float, dop_density: float, epsilon: float
) -> float:
    """Calculate the D₂D₃D₄ component of the denominator of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.
    dop_density : float
        Doping density in cm⁻³.
    epsilon : float
        Static dielectric constant, dimensionless.

    Returns
    -------
    float
        The D₂D₃D₄ component of Γₚᵥ, dimensionless.
    """
    D_2 = _D_2_equation(E_gap, alpha, tau, dop_density)
    D_3 = _D_3_equation(E_gap, alpha, tau, dop_density, epsilon)
    D_4 = _D_4_equation(E_gap, alpha, tau)

    D_denominator = D_2 * D_3 * D_4

    return D_denominator


def _Final_T_denominator(
        E_gap: float, alpha: float, tau: float, sigma: float, dos_mass: float, dop_density: float,
        epsilon: float, mu: float
) -> float:
    """Calculate the 1 + (T₁T₂T₃) component of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.
    sigma : float
        Spectral dispersion of the absorption coefficient, dimensionless.
    dos_mass : float
        Density-of-states effective mass in units of m₀.
    dop_density : float
        Doping density in cm⁻³.
    epsilon : float
        Static dielectric constant, dimensionless.
    mu : float
        Charge carrier mobility in cm² V⁻¹ s⁻¹.

    Returns
    -------
    float
        The 1 + (T₁T₂T₃) component of Γₚᵥ, dimensionless.
    """
    T_1 = _T_1_equation(E_gap, dos_mass, epsilon, mu)
    T_2 = _T_2_equation(E_gap, tau, sigma, dop_density)
    T_3 = _T_3_equation(E_gap, alpha, dos_mass, dop_density)

    T_denominator = 1 + (T_1 * T_2 * T_3)

    return T_denominator


def _Final_S_denominator(
        E_gap: float, alpha: float, tau: float, dos_mass: float, dop_density: float, mu: float
) -> float:
    """Calculate the 1 + (S₁S₂) component of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.
    dos_mass : float
        Density-of-states effective mass in units of m₀.
    dop_density : float
        Doping density in cm⁻³.
    mu : float
        Charge carrier mobility in cm² V⁻¹ s⁻¹.

    Returns
    -------
    float
        The 1 + (S₁S₂) component of Γₚᵥ, dimensionless.
    """
    S_1 = _S_1_equation(E_gap, alpha, tau, dos_mass, mu)
    S_2 = _S_2_equation(alpha, dop_density, mu)

    S_denominator = 1 + (S_1 * S_2)

    return S_denominator


def _A_1_equation(E_gap: float, alpha: float, tau: float, sigma: float, dos_mass: float) -> float:
    """Calculate the A₁ component of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.
    sigma : float
        Spectral dispersion of the absorption coefficient, dimensionless.
    dos_mass : float
        Density-of-states effective mass in units of m₀.

    Returns
    -------
    float
        The A₁ component of Γₚᵥ, dimensionless.
    """
    a_1 = 0.295
    a_2 = 0.185

    E_gap_0_5 = E_gap ** 0.5

    power_1 = - a_2 * E_gap_0_5
    power_2 = sigma ** power_1

    numerator = a_1 * tau * (alpha ** power_2)
    denominator = dos_mass ** 2

    A_1 = numerator / denominator

    return A_1


def _A_2_equation(alpha: float, tau: float, sigma: float) -> float:
    """Calculate the A₂ component of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.
    sigma : float
        Spectral dispersion of the absorption coefficient, dimensionless.

    Returns
    -------
    float
        The A₂ component of Γₚᵥ, dimensionless.
    """
    a_3 = 1.0e-7

    fraction = (sigma ** 10) / (alpha * tau)

    A_2 = 1 + ((a_3 * fraction) ** 0.4)

    return A_2


# D_1 Equation

def _D_1_equation(alpha: float, dop_density: float, epsilon: float) -> float:
    """Calculate the D₁ component of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    dop_density : float
        Doping density in cm⁻³.
    epsilon : float
        Static dielectric constant, dimensionless.

    Returns
    -------
    float
        The D₁ component of Γₚᵥ, dimensionless.
    """
    d_1 = 4.4e-5
    d_2 = 39

    log_bracket = alpha / d_2

    power = 0.22 * np.log10(log_bracket)

    denominator = (epsilon ** 0.8) * (alpha ** 2)

    D_1 = (1 + d_1 * (dop_density / denominator)) ** power

    return D_1


# D_2 Equation

def _D_2_equation(E_gap: float, alpha: float, tau: float, dop_density: float) -> float:
    """Calculate the D₂ component of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.
    dop_density : float
        Doping density in cm⁻³.

    Returns
    -------
    float
        The D₂ component of Γₚᵥ, dimensionless.
    """
    d_3 = 1e-21

    E_gap_4 = E_gap ** 4
    power = 0.05 * E_gap_4

    fraction = dop_density / ((alpha ** 2) * tau)

    D_2 = (1 + d_3 * fraction) ** power

    return D_2


# D_3 Equation

def _D_3_equation(
        E_gap: float, alpha: float, tau: float, dop_density: float, epsilon: float
) -> float:
    """Calculate the D₃ component of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.
    dop_density : float
        Doping density in cm⁻³.
    epsilon : float
        Static dielectric constant, dimensionless.

    Returns
    -------
    float
        The D₃ component of Γₚᵥ, dimensionless.
    """
    d_4 = 2.1e4
    d_5 = 50

    E_gap_8_5 = E_gap ** 8.5
    E_gap_1_5 = E_gap ** -1.5

    power_num_1 = 0.68 * E_gap_1_5
    power_num_2 = np.log10((10 * dop_density) / epsilon) / d_5
    power_denum = (E_gap - 1.5) / 0.1

    numerator = (d_4 * E_gap_8_5 * tau * (alpha ** power_num_1)) ** power_num_2
    denominator = 1 + (10 ** power_denum)

    D_3 = 1 + (numerator / denominator)

    return D_3


# D_4 Equation

def _D_4_equation(E_gap: float, alpha: float, tau: float) -> float:
    """Calculate the D₄ component of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.

    Returns
    -------
    float
        The D₄ component of Γₚᵥ, dimensionless.
    """
    d_6 = 7.7e-7

    fraction = d_6 / ((E_gap ** 17) * alpha * tau)

    D_4 = 1 + (fraction ** 0.6)

    return D_4


# T_1 Equation

def _T_1_equation(E_gap: float, dos_mass: float, epsilon: float, mu: float) -> float:
    """Calculate the T₁ component of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    dos_mass : float
        Density-of-states effective mass in units of m₀.
    epsilon : float
        Static dielectric constant, dimensionless.
    mu : float
        Charge carrier mobility in cm² V⁻¹ s⁻¹.

    Returns
    -------
    float
        The T₁ component of Γₚᵥ, dimensionless.
    """
    t_1 = 5.1e-2
    t_2 = 4.6e-2
    E_gap_4_3 = E_gap ** 4.3

    numerator = t_1 * ((E_gap + 0.5) ** 11)
    power = (t_2 * E_gap_4_3) + 0.9
    denominator = dos_mass * (epsilon ** 0.5) * (mu ** power)

    T_1 = numerator / denominator

    return T_1


# T_2 Equation

def _T_2_equation(E_gap: float, tau: float, sigma: float, dop_density: float) -> float:
    """Calculate the T₂ component of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    tau : float
        Non-radiative recombination lifetime in s.
    sigma : float
        Spectral dispersion of the absorption coefficient, dimensionless.
    dop_density : float
        Doping density in cm⁻³.

    Returns
    -------
    float
        The T₂ component of Γₚᵥ, dimensionless.
    """
    t_8 = 9.5e-18
    t_9 = 7.80e7

    E_gap_1_25 = E_gap ** 1.25
    E_gap_5_1 = E_gap ** 5.1

    power_main = 0.47 * E_gap_1_25
    power_denom = 0.5 / sigma

    numerator = 1 + t_8 * dop_density * (1 + E_gap_5_1)
    denominator = t_9 * tau * np.power(10, power_denom)

    T_2 = (numerator / denominator) ** power_main

    return T_2


# T_3 Equation

def _T_3_prime_equation(E_gap: float, alpha: float) -> float:
    """Calculate the T₃' component of T₃ used in Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.

    Returns
    -------
    float
        The T₃' component of T₃, dimensionless.
    """
    t_3 = 1.9e4
    t_4 = 9.5e-3
    t_5 = 2.4e-4

    E_gap_10 = E_gap ** 10

    pre_exp = t_3 * (1 + t_4 * E_gap_10)
    power = 0.5 / (1 + t_5 * E_gap_10)
    exponent = np.exp(-0.1 * (alpha ** power))

    T_3_prime = pre_exp * exponent

    return T_3_prime


def _T_3_double_prime_equation(
        E_gap: float, alpha: float, dos_mass: float, dop_density: float
) -> float:
    """Calculate the T₃'' component of T₃ used in Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    dos_mass : float
        Density-of-states effective mass in units of m₀.
    dop_density : float
        Doping density in cm⁻³.

    Returns
    -------
    float
        The T₃'' component of T₃, dimensionless.
    """
    t_6 = 1.5e5

    power_num = 1 - 0.74 * np.exp(- alpha / t_6)
    power_dom = (E_gap - 1.5) / 0.01

    numerator = (0.16 / dos_mass ** 3) * (((dop_density * (dos_mass ** 3)) / 0.16) ** power_num)
    denominator = 1 + 10 ** power_dom

    T_3_double_prime = 1 + numerator / denominator

    return T_3_double_prime


def _T_3_equation(E_gap: float, alpha: float, dos_mass: float, dop_density: float) -> float:
    """Calculate the T₃ component of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    dos_mass : float
        Density-of-states effective mass in units of m₀.
    dop_density : float
        Doping density in cm⁻³.

    Returns
    -------
    float
        The T₃ component of Γₚᵥ, dimensionless.
    """
    t_7 = 1.6e-3
    E_gap_8 = E_gap ** 8

    power = (t_7 * E_gap_8) + 0.6

    T_3_prime = _T_3_prime_equation(E_gap, alpha)
    T_3_double_prime = _T_3_double_prime_equation(E_gap, alpha, dos_mass, dop_density)

    T_3 = (1 + T_3_prime * T_3_double_prime) ** power

    return T_3


def _S_1_equation(E_gap: float, alpha: float, tau: float, dos_mass: float, mu: float) -> float:
    """Calculate the S₁ component of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    E_gap : float
        Optical band gap in eV.
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    tau : float
        Non-radiative recombination lifetime in s.
    dos_mass : float
        Density-of-states effective mass in units of m₀.
    mu : float
        Charge carrier mobility in cm² V⁻¹ s⁻¹.

    Returns
    -------
    float
        The S₁ component of Γₚᵥ, dimensionless.
    """
    s_1 = 2.4e4
    s_2 = 4e-4

    E_gap_10 = E_gap ** 10

    power_num = 2.4 * E_gap
    power_den = 1 / (1 + s_2 * E_gap_10)

    numerator = ((10 ** power_num) * (alpha ** 0.75) * tau)
    denominator = s_1 * (dos_mass ** 2) * (mu ** power_den)

    S_1 = numerator / denominator

    return S_1


def _S_2_equation(alpha: float, dop_density: float, mu: float) -> float:
    """Calculate the S₂ component of Γₚᵥ from Crovetto 2024.

    Parameters
    ----------
    alpha : float
        Spectrally averaged absorption coefficient in cm⁻¹.
    dop_density : float
        Doping density in cm⁻³.
    mu : float
        Charge carrier mobility in cm² V⁻¹ s⁻¹.

    Returns
    -------
    float
        The S₂ component of Γₚᵥ, dimensionless.
    """
    s_3 = 4.8e3

    bracket = (s_3 / alpha) ** 20

    S_2 = 1 + (bracket * (mu ** 0.5) * np.log10(dop_density))

    return S_2
