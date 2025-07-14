
import numpy as np
from scipy.optimize import root
from scipy.integrate import quad

from dataclasses import dataclass

# conversion factor from density to energy
mev3_fm3 = 1.0 / (197.327 ** 3)

# integration technique
def integrate(func, a, b, **kwargs):
    return quad(func, a, b, full_output=1, **kwargs)

def nucleon_m0(sigma, amplitude, *args):
    """Calculate m0 relevant for M_N = 1/2 (sqrt(4m0^2+a^2sigma^2) -+ bsigma)
    Args:
        [0] (float): quark bare mass
        """
    return amplitude / (1 - sigma / args[0])

def nucleon_m0_derivative(sigma, amplitude, *args):
    """Derivative of m0 with respect to sigma
    Args:
        [0] (float): quark bare mass
    """
    return amplitude / (args[0] * (1 - sigma / args[0]) ** 2)

def nucleon_mass_parity_model(sigma, parity, *args):
    """Calculate nucleon mass based on the parity model
    parity: '+' for N+, '-' for N-
    Args:
        [0] (float): a coupling
        [1] (float): b coupling
        [2] (float): mass amplitude
        [3+] (float): arguments for nucleon_m0 function 
    """
    mult = None
    if parity == '+':
        mult = 1.0
    elif parity == '-':
        mult = -1.0
    else:
        raise ValueError("Parity must be '+' or '-'")
    m0 = nucleon_m0(sigma, args[2], *args[3:])
    a = args[0]
    b = args[1] * mult # introduce parity into the b coupling
    return 0.5 * (np.sqrt(4 * m0 ** 2 + a ** 2 * sigma ** 2) - b * sigma)

    
def nucleon_mass_derivative_parity_model(sigma, parity, *args):
    """Derivative of nucleon mass with respect to sigma
    Args:
        [0] (float): a coupling
        [1] (float): b coupling
        [2] (float): mass amplitude
        [3+] (float): arguments for nucleon_m0 function 
    """
    mult = None
    if parity == '+':
        mult = 1.0
    elif parity == '-':
        mult = -1.0
    else:
        raise ValueError("Parity must be '+' or '-'")
    m0_derivative = nucleon_m0_derivative(sigma, args[2], *args[3:])
    m0 = nucleon_m0(sigma, args[2], *args[3:])
    a = args[0]
    b = args[1] * mult # introduce parity into the b coupling
    return 0.5 * ( (a ** 2 * sigma + 4 * m0 * m0_derivative) / (2 * m0 + b * sigma) - b)

@dataclass
class EoSParameters:
    """Data class to hold EoS parameters."""
    instantiated: bool
    instantiation_error: float
    vacuum_quark_condensate: float
    m_q_bare: float
    sigma_vac: float
    sigma_sat: float
    momentum_scale: float
    g_coupling: float
    a_coupling: float
    b_coupling: float
    mass_amplitude: float

# fix parameters based on the properties of nuclear matter
def fix_parameters(vacuum_quark_condensate, m_q_bare, m_q_vac, m_N,  
                   m_N_star, n_sat, chem_pot_sat, rel_tol, guess_moment_scale):
    n_f = 2
    n_c = 3
    # sigma field in vacuum
    sigma_vac = m_q_bare - m_q_vac
    # sigma coupling
    g_coupling = sigma_vac / (2 * vacuum_quark_condensate)

    
    def score_minimize1(x):
        moment_scale = x[0]

        # Self-consistency for vacuum quark condensate
        def eq1():
            def integrand(k):
                return k ** 2 / (2 * np.pi ** 2) * np.exp(-(k / moment_scale) ** 2) / np.sqrt(1 + (k / m_q_vac) ** 2)
            integral = integrate(integrand, 0, np.inf)[0]
            return 1 + 2 * n_c * n_f * integral / vacuum_quark_condensate
        return [eq1()]
    
    res1 = root(score_minimize1, x0=[guess_moment_scale], tol=rel_tol)
    
    # Time to recreate properties of symmetric nuclear matter at saturation density
    k_fermi_at_sat = (1.5 * np.pi ** 2 * n_sat / mev3_fm3) ** (1 / 3)
    m_N_at_sat = np.sqrt(chem_pot_sat ** 2 - k_fermi_at_sat ** 2)
    
    def score_minimize2(x):
        sigma_sat = x[0]
        a_coupling = x[1]
        b_coupling = x[2]
        mass_amplitude = x[3]

        # 1. sigma field at saturation density
        def eq1():
            # LHS
            res = sigma_sat / (4 * g_coupling)

            quark_mass_sat = m_q_bare - sigma_sat
            # nucl_mass_sat = 0.5 * (np.sqrt(4 * nucleon_m0(sigma_sat, mass_amplitude, m_q_bare) ** 2 + a_coupling ** 2 * sigma_sat ** 2) - b_coupling * sigma_sat)
            nucl_mass_sat = m_N_at_sat
            nucl_mass_sat_derivative = nucleon_mass_derivative_parity_model(sigma_sat, '+', a_coupling, b_coupling, mass_amplitude, m_q_bare)
            
            # quark contribution to sigma field at saturation density
            def integrand1(k):
                return k ** 2 / (2 * np.pi ** 2) * np.exp(-(k / res1.x[0]) ** 2) / np.sqrt(1 + (k / quark_mass_sat) ** 2)
            integral1 = integrate(integrand1, 0, np.inf)[0]
            res += 2 * n_c * integral1

            # hadron contribution to sigma field at saturation density
            def integrand2(k):
                return k ** 2 / (2 * np.pi ** 2) / np.sqrt(1 + (k / nucl_mass_sat) ** 2) * nucl_mass_sat_derivative
            integral2 = integrate(integrand2, 0, k_fermi_at_sat)[0]
            res += 2 * n_f * integral2
            return res / vacuum_quark_condensate
        
        # 2. nucleon mass at saturation density
        def eq2():
            #res = m_N_at_sat - 0.5 * (np.sqrt(4 * nucleon_m0(sigma_sat, mass_amplitude, m_q_bare) ** 2 + a_coupling ** 2 * sigma_sat ** 2) - b_coupling * sigma_sat)
            res = m_N_at_sat - nucleon_mass_parity_model(sigma_sat, '+', a_coupling, b_coupling, mass_amplitude, m_q_bare)
            return res / m_N_at_sat
        
        # 3. nucleon mass in vacuum
        def eq3():
            res = m_N - nucleon_mass_parity_model(sigma_vac, '+', a_coupling, b_coupling, mass_amplitude, m_q_bare)
            return res / m_N
        
        # 4. nucleon partner mass in vacuum
        def eq4():
            res = m_N_star - nucleon_mass_parity_model(sigma_vac, '-', a_coupling, b_coupling, mass_amplitude, m_q_bare)
            return res / m_N_star
        
        return [eq1(), eq2(), eq3(), eq4()]
    
    res2 = root(score_minimize2, x0=[0.0, 0.1, 0.5, m_N / 2], tol=rel_tol)

    errors = [*res1.fun, *res2.fun]
    return EoSParameters(
        instantiated=(res1.success and res2.success),
        instantiation_error=np.abs(errors),
        vacuum_quark_condensate=vacuum_quark_condensate,
        m_q_bare=m_q_bare,
        sigma_vac=sigma_vac,
        sigma_sat=res2.x[0],
        momentum_scale=res1.x[0],
        g_coupling=g_coupling,
        a_coupling=res2.x[1],
        b_coupling=res2.x[2],
        mass_amplitude=res2.x[3]
    )
        