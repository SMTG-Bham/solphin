import logging

import numpy as np

__all__ = []

logging.basicConfig(level=logging.INFO)

'''
This module calculates all the components required for the Γₚᵥ Figure of Merit from Andrea Crovetto 2024 J. Phys. Energy 6 025009
'''


def Final_equation(
        E_gap: float, alpha: float, tau: float, sigma: float, dos_mass: float, dop_density: float,
        epsilon: float, mu: float
) -> float:
    ''' Calculates the total Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        sigma(float): Spectral dispersion of the absorption coefficient spectrum, unitless
        dos_mass(float): Density of States effective mass in m₀
        dop_density(float): Doping density in cm⁻³
        epsilon(float): Static dielectric constant, unitless
        mu(float): Charge carrier mobility in cm²V⁻¹s⁻¹

    Returns:
        PV_FOM(float): Γₚᵥ Photovoltaic Figure of Merit from Crovetto 2024, unitless
    '''

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
    ''' Calculates the numerator for the Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        sigma(float): Spectral dispersion of the absorption coefficient spectrum, unitless
        dos_mass(float): Density of States effective mass in m₀
        dop_density(float): Doping density in cm⁻³
        epsilon(float): Static dielectric constant, unitless

    Returns:
        numerator(float): Γₚᵥ numerator only from Crovetto 2024, unitless
    '''

    A_1 = _A_1_equation(E_gap, alpha, tau, sigma, dos_mass)
    A_2 = _A_2_equation(alpha, tau, sigma)
    D_1 = _D_1_equation(alpha, dop_density, epsilon)

    numerator = A_1 * A_2 * D_1

    return numerator


def _Final_D_denominator(
        E_gap: float, alpha: float, tau: float, dop_density: float, epsilon: float
) -> float:
    ''' Calculates the D₂D₃D₄ component of the denominator of Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        dop_density(float): Doping density in cm⁻³
        epsilon(float): Static dielectric constant, unitless

    Returns:
        D_denominator(float): The D₂D₃D₄ component of Γₚᵥ from Crovetto 2024, unitless
    '''

    D_2 = _D_2_equation(E_gap, alpha, tau, dop_density)
    D_3 = _D_3_equation(E_gap, alpha, tau, dop_density, epsilon)
    D_4 = _D_4_equation(E_gap, alpha, tau)

    D_denominator = D_2 * D_3 * D_4

    return D_denominator


def _Final_T_denominator(
        E_gap: float, alpha: float, tau: float, sigma: float, dos_mass: float, dop_density: float,
        epsilon: float, mu: float
) -> float:
    ''' Calculates the 1 + (T₁T₂T₃) component of the Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        sigma(float): Spectral dispersion of the absorption coefficient spectrum, unitless
        dos_mass(float): Density of States effective mass in m₀
        dop_density(float): Doping density in cm⁻³
        epsilon(float): Static dielectric constant, unitless
        mu(float): Charge carrier mobility in cm²V⁻¹s⁻¹

    Returns:
        T_denominator(float): The 1 + (T₁T₂T₃) component of Γₚᵥ from Crovetto 2024, unitless
    '''

    T_1 = _T_1_equation(E_gap, dos_mass, epsilon, mu)
    T_2 = _T_2_equation(E_gap, tau, sigma, dop_density)
    T_3 = _T_3_equation(E_gap, alpha, dos_mass, dop_density)

    T_denominator = 1 + (T_1 * T_2 * T_3)

    return T_denominator


def _Final_S_denominator(
        E_gap: float, alpha: float, tau: float, dos_mass: float, dop_density: float, mu: float
) -> float:
    ''' Calculates the 1 + (S₁S₂) component of the Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        dos_mass(float): Density of States effective mass in m₀
        dop_density(float): Doping density in cm⁻³
        mu(float): Charge carrier mobility in cm²V⁻¹s⁻¹

    Returns:
        S_denominator(float): The 1 + (S₁S₂) component of Γₚᵥ from Crovetto 2024, unitless
    '''

    S_1 = _S_1_equation(E_gap, alpha, tau, dos_mass, mu)
    S_2 = _S_2_equation(alpha, dop_density, mu)

    S_denominator = 1 + (S_1 * S_2)

    return S_denominator


def _A_1_equation(E_gap: float, alpha: float, tau: float, sigma: float, dos_mass: float) -> float:
    ''' Calculates the A₁ component of Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        sigma(float): Spectral dispersion of the absorption coefficient spectrum, unitless
        dos_mass(float): Density of States effective mass in m₀

    Returns:
        A_1(float): The A₁ component of Γₚᵥ from Crovetto 2024, unitless
    '''

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
    ''' Calculates the A₂ component of Γₚᵥ from Crovetto 2024

    Parameters:
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        sigma(float): Spectral dispersion of the absorption coefficient spectrum, unitless
       
    Returns:
        A_2(float): The A₂ component of Γₚᵥ from Crovetto 2024, unitless
    '''

    a_3 = 1.0e-7

    fraction = (sigma ** 10) / (alpha * tau)

    A_2 = 1 + ((a_3 * fraction) ** 0.4)

    return A_2


# D_1 Equation

def _D_1_equation(alpha: float, dop_density: float, epsilon: float) -> float:
    ''' Calculates the D₁ component of Γₚᵥ from Crovetto 2024

    Parameters:
        alpha(float): Spectral average of incident light in cm⁻¹
        dop_density(float): Doping density in cm⁻³
        epsilon(float): Static dielectric constant, unitless

    Returns:
        D_1(float): The D₁ component of Γₚᵥ from Crovetto 2024, unitless
    '''

    d_1 = 4.4e-5
    d_2 = 39

    log_bracket = alpha / d_2

    power = 0.22 * np.log10(log_bracket)

    denominator = (epsilon ** 0.8) * (alpha ** 2)

    D_1 = (1 + d_1 * (dop_density / denominator)) ** power

    return D_1


# D_2 Equation

def _D_2_equation(E_gap: float, alpha: float, tau: float, dop_density: float) -> float:
    ''' Calculates the D₂ component of Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        dop_density(float): Doping density in cm⁻³

    Returns:
        D_2(float): The D₂ component of Γₚᵥ from Crovetto 2024, unitless
    '''

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
    ''' Calculates the D₃ component of Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        dop_density(float): Doping density in cm⁻³
        epsilon(float): Static dielectric constant, unitless

    Returns:
        D_3(float): The D₃ component of Γₚᵥ from Crovetto 2024, unitless
    '''

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
    ''' Calculates the D₄ component of Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
       
    Returns:
        D_4(float): The D₄ component of Γₚᵥ from Crovetto 2024, unitless
    '''

    d_6 = 7.7e-7

    fraction = d_6 / ((E_gap ** 17) * alpha * tau)

    D_4 = 1 + (fraction ** 0.6)

    return D_4


# T_1 Equation

def _T_1_equation(E_gap: float, dos_mass: float, epsilon: float, mu: float) -> float:
    ''' Calculates the total Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        dos_mass(float): Density of States effective mass in m₀
        epsilon(float): Static dielectric constant, unitless
        mu(float): Charge carrier mobility in cm²V⁻¹s⁻¹

    Returns:
        PV_FOM(float): Γₚᵥ Photovoltaic Figure of Merit from Crovetto 2024, unitless
    '''

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
    ''' Calculates the T₂ component of Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        tau(float): Non-radiative recombination lifetime in s
        sigma(float): Spectral dispersion of the absorption coefficient spectrum, unitless
        dop_density(float): Doping density in cm⁻³

    Returns:
        T_2(float): The T₂ component of Γₚᵥ from Crovetto 2024, unitless
    '''

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
    ''' Calculates the T₃' component of T₃ used in Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        alpha(float): Spectral average of incident light in cm⁻¹

    Returns:
        T_3_prime(float): The T₃' component of T₃ from Crovetto 2024, unitless
    '''

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
    ''' Calculates the T₃'' component of T₃ used in the Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        alpha(float): Spectral average of incident light in cm⁻¹
        dos_mass(float): Density of States effective mass in m₀
        dop_density(float): Doping density in cm⁻³

    Returns:
        T_3_double_prime(float): The T₃'' component of T₃ from Crovetto 2024, unitless
    '''

    t_6 = 1.5e5

    power_num = 1 - 0.74 * np.exp(- alpha / t_6)
    power_dom = (E_gap - 1.5) / 0.01

    numerator = (0.16 / dos_mass ** 3) * (((dop_density * (dos_mass ** 3)) / 0.16) ** power_num)
    denominator = 1 + 10 ** power_dom

    T_3_double_prime = 1 + numerator / denominator

    return T_3_double_prime


def _T_3_equation(E_gap: float, alpha: float, dos_mass: float, dop_density: float) -> float:
    ''' Calculates the T₃ component of Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        alpha(float): Spectral average of incident light in cm⁻¹
        dos_mass(float): Density of States effective mass in m₀
        dop_density(float): Doping density in cm⁻³

    Returns:
        T_3(float): The T₃ component of Γₚᵥ from Crovetto 2024, unitless
    '''

    t_7 = 1.6e-3
    E_gap_8 = E_gap ** 8

    power = (t_7 * E_gap_8) + 0.6

    T_3_prime = _T_3_prime_equation(E_gap, alpha)
    T_3_double_prime = _T_3_double_prime_equation(E_gap, alpha, dos_mass, dop_density)

    T_3 = (1 + T_3_prime * T_3_double_prime) ** power

    return T_3


def _S_1_equation(E_gap: float, alpha: float, tau: float, dos_mass: float, mu: float) -> float:
    ''' Calculates the S₁ component of Γₚᵥ from Crovetto 2024

    Parameters:
        E_gap(float): Optical Band Gap in eV  
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        dos_mass(float): Density of States effective mass in m₀
        mu(float): Charge carrier mobility in cm²V⁻¹s⁻¹

    Returns:
        S_1(float): The S₁ component of Γₚᵥ from Crovetto 2024, unitless
    '''

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
    ''' Calculates the S₂ component of Γₚᵥ from Crovetto 2024

    Parameters:
        alpha(float): Spectral average of incident light in cm⁻¹
        dop_density(float): Doping density in cm⁻³
        mu(float): Charge carrier mobility in cm²V⁻¹s⁻¹

    Returns:
        S_2(float): The S₂ component of Γₚᵥ from Crovetto 2024, unitless
    '''

    s_3 = 4.8e3

    bracket = (s_3 / alpha) ** 20

    S_2 = 1 + (bracket * (mu ** 0.5) * np.log10(dop_density))

    return S_2
