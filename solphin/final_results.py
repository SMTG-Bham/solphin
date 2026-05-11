from solphin.pv_fom import Final_equation
from solphin.db_fom import max_eff

# Calculating equation 33 from the FOM paper

def SQ_relative_FOM_PV_efficiency(E_gap, photon_spectrum, alpha, tau, sigma, dos_mass, dop_density, epsilon, mu, Tcell):

    ''' Calculates the final value for the photovoltaic figure of merit from Crovetto 2024 as a percentage of the Shockley Queisser efficiency limit.
    
    Parameters:
        E_gap(float): Optical Band Gap in eV  
        photon_spectrum(numpy.ndarrray): Converted input spectrum from DB_FOM.convert_spectrum y: Number of photons (Np/m2/s/dE) x: Energy (eV)
        alpha(float): Spectral average of incident light in cm⁻¹
        tau(float): Non-radiative recombination lifetime in s
        sigma(float): Spectral dispersion of the absorption coefficient spectrum, unitless
        dos_mass(float): Density of States effective mass in m₀
        dop_density(float): Doping density in cm⁻³
        epsilon(float): Static dielectric constant, unitless
        mu(float): Charge carrier mobility in cm²V⁻¹s⁻¹
        Tcell(float): Operating temperature of the cell in K

    Returns:
        efficiency(float): Percentage photovoltaic figure of merit efficiency.
    '''

    PV_FOM = Final_equation(E_gap, alpha, tau, sigma, dos_mass, dop_density, epsilon, mu)

    k_1 = 3.3e-1
    k_2 = 9.06e-2
    k_3 = 2.48e-3

    FOM_Pv_235 = PV_FOM ** (-0.235)
    FOM_Pv_869 = PV_FOM ** 0.869
    FOM_Pv_362 = PV_FOM ** (-0.362)

    fraction = (k_1 * FOM_Pv_235) / (1 + k_2 * FOM_Pv_869)
    denom_bracket = 1 + (k_3 * FOM_Pv_362)

    SQ_eff = max_eff(E_gap, photon_spectrum, Tcell)
    SQ = SQ_eff * 100

    SQ_efficiency = SQ / ((1 + fraction) * denom_bracket)

    efficiency = (SQ / 100) * SQ_efficiency

    return SQ, SQ_efficiency, efficiency