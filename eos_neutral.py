
import eos_module
import argparse

parser = argparse.ArgumentParser(description='Calculate neutral beta-equilibrated EoS for u, d, N+, N-',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

required = parser.add_argument_group('Required arguments')
required.add_argument('--m_q_vac', type=float, required=True,
                        help='Vacuum quark mass in MeV (order of 500 MeV)')

constants = parser.add_argument_group('Constants')
constants.add_argument('--f_pi', type=float, default=90.0,
                       help='Pion decay constant in MeV')
constants.add_argument('--m_pi', type=float,
                       default=140.0, help='Pion mass in MeV')
constants.add_argument('--m_e', type=float,
                       default=0.511, help='Electron mass in MeV')
constants.add_argument('--q_cond_vac', type=float, default=-
                       250.0, help='Vacuum quark condensate per flavour in MeV')
constants.add_argument('--n_sat', type=float, default=0.16,
                       help='Saturation density in fm^-3')
constants.add_argument('--m_N', type=float, default=939.0,
                          help='Nucleon mass in MeV')
constants.add_argument('--m_N_star', type=float, default=1535.0,
                          help='Nucleon\'s partner mass in MeV')
constants.add_argument('--chem_pot_sat', type=float, default=923.0,
                          help='Chemical potential of nuclear matter at saturation density in MeV')

optimization = parser.add_argument_group('Optimization')
optimization.add_argument('--rel_tol', type=float, default=1e-10,
                          help='Relative tolerance for the optimizer')
optimization.add_argument('--guess_moment_scale', type=float,
                          default=500, help='Guess for the moment scale in MeV')
optimization.add_argument('--guess_charge_chemical_potential', type=float,
                          default=-100, help='Initial guess for the charge chemical potential in MeV')
optimization.add_argument('--start', type=float,
                          default=700, help='mu_b to start calculations with in MeV')
optimization.add_argument('--finish', type=float,
                          default=2000, help='mu_b to finish calculations with in MeV')
optimization.add_argument('--num_points', type=int,
                          default=100, help='Number of EoS points, linear in mu_b')

interface = parser.add_argument_group('Interface')
interface.add_argument('--non_verbose', action='store_true',
                       help='Suppress verbose output', default=False)
interface.add_argument('--output', type=str, default='',
                       help='Output file name')

args = parser.parse_args()

def main():
    # conversion factor from density to energy
    mev3_fm3 = eos_module.mev3_fm3
    # vacuum quark condensate value
    vacuum_quark_condensate = 2 * args.q_cond_vac ** 3
    # GMOR relates bare quark mass to pion mass and decay constant
    m_q_bare = -args.m_pi ** 2 * args.f_pi ** 2 / vacuum_quark_condensate

    # Fix the rest of the parameters from the properties of nuclear matter
    parameters = eos_module.fix_parameters(vacuum_quark_condensate, m_q_bare, args.m_q_vac, args.m_N,  
                                            args.m_N_star,  args.n_sat, args.chem_pot_sat, args.rel_tol, args.guess_moment_scale)
    if not args.non_verbose:
        print('Fixed properties (MeV powers): ')
        for fields in parameters.__dataclass_fields__:
            print(f'{fields}: {getattr(parameters, fields)}')
    
main()