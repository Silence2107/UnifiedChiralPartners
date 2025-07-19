
import numpy as np
import matplotlib.pyplot as plt
import sys
import eos_module
import argparse

parser = argparse.ArgumentParser(description='Calculate neutral beta-equilibrated EoS for u, d, N+, N-',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('Required arguments')
required.add_argument('--eta_v', type=float, required=True,
                        help='Vector to scalar coupling ratio')

constants = parser.add_argument_group('Constants')
constants.add_argument('--m_q_vac', type=float, default=300.0,
                        help='Vacuum quark mass in MeV')
constants.add_argument('--f_pi', type=float, default=90.0,
                       help='Pion decay constant in MeV')
constants.add_argument('--m_pi', type=float,
                       default=140.0, help='Pion mass in MeV')
constants.add_argument('--q_cond_vac', type=float, default=-
                       242.0, help='Vacuum quark condensate per flavour in MeV')
constants.add_argument('--n_sat', type=float, default=0.16,
                       help='Saturation density in fm^-3')
constants.add_argument('--m_N', type=float, default=939.0,
                          help='Nucleon mass in MeV')
constants.add_argument('--m_N_star', type=float, default=1535.0,
                          help='Nucleon\'s partner mass in MeV')
constants.add_argument('--chem_pot_sat', type=float, default=923.0,
                          help='Chemical potential of nuclear matter at saturation density in MeV')
constants.add_argument('--g_w', type=float, default=3.0,
                          help='Coupling of nucleon chemical potential to vector field')

optimization = parser.add_argument_group('Optimization')
optimization.add_argument('--rel_tol', type=float, default=1e-10,
                          help='Relative tolerance for the optimizer')
optimization.add_argument('--guess_moment_scale', type=float,
                          default=500, help='Guess for the moment scale in MeV')
optimization.add_argument('--start', type=float,
                          default=700, help='mu_b to start calculations with in MeV')
optimization.add_argument('--finish', type=float,
                          default=1500, help='mu_b to finish calculations with in MeV')
optimization.add_argument('--num_points', type=int,
                          default=100, help='Number of EoS points, linear in mu_b')

interface = parser.add_argument_group('Interface')
interface.add_argument('--non_verbose', action='store_true',
                       help='Suppress verbose output', default=False)
interface.add_argument('--output', type=str, default='',
                       help='Output file name')

args = parser.parse_args(
)#['--eta_v', '0.01' ,'--m_q_vac', '900', '--start', '900', '--finish', '1800'])

def main():
    # conversion factor from density to energy
    mev3_fm3 = eos_module.mev3_fm3
    # vacuum quark condensate value
    vacuum_quark_condensate = 2 * args.q_cond_vac ** 3
    # GMOR relates bare quark mass to pion mass and decay constant
    m_q_bare = -args.m_pi ** 2 * args.f_pi ** 2 / vacuum_quark_condensate

    # Fix the rest of the parameters from the properties of nuclear matter
    parameters = eos_module.resolve_parameters(vacuum_quark_condensate, m_q_bare, args.m_q_vac, args.m_N,  
                                            args.m_N_star,  args.n_sat, args.chem_pot_sat, args.eta_v, args.g_w, args.rel_tol, args.guess_moment_scale)
    if not args.non_verbose:
        print('Fixed properties (MeV powers): ')
        for fields in parameters.__dataclass_fields__:
            print(f'{fields}: {getattr(parameters, fields)}')

    if not args.output == '':
        output = open(args.output, 'w')
    else:
        output = sys.stdout

    # muBs = np.linspace(args.start, args.finish, args.num_points)
    # sigmas = []
    # sigmas_err = []
    # omegas = []
    # omegas_err = []
    # for muB in muBs:
    #     # 
    #     guesses =  [[m_q_bare / 2, parameters.omega_sat], [parameters.sigma_sat, 0.0]]
    #     chempots = [muB / 3, muB / 3, muB, muB, muB, muB] # u, d, n, p, n*, p*
    #     # Resolve the sigma field
    #     meson_res = eos_module.resolve_meson_fields(chempots, parameters,
    #                                                         guesses, args.rel_tol)
    #     print(meson_res)
    #     sigmas.append(meson_res[0][0])
    #     sigmas_err.append(meson_res[1][0] * meson_res[0][0])
    #     omegas.append(meson_res[0][1])
    #     omegas_err.append(meson_res[1][1] * meson_res[0][1])

    # sigmas = np.array(sigmas)
    # sigmas_err = np.array(sigmas_err)
    # omegas = np.array(omegas)
    # omegas_err = np.array(omegas_err)

    # # plot the results
    # plt.figure(figsize=(10, 6))
    # plt.errorbar(muBs, m_q_bare - sigmas, yerr=sigmas_err,
    #              label='Quark mass', fmt='o', markersize=3)
    # plt.errorbar(muBs, omegas, yerr=omegas_err,
    #              label=r'$\omega$ field', fmt='o', markersize=3)
    # plt.xlabel(r'$\mu_B$ (MeV)')
    # plt.ylabel(r'(MeV)')
    # plt.axvline(x=args.chem_pot_sat, color='red', linestyle='--', label='Chemical potential at saturation')
    # plt.legend()
    # plt.savefig('meson_fields.pdf')

    muBs = np.linspace(args.start, args.finish, args.num_points)
    for muB in muBs:
        #
        guesses =  [[m_q_bare / 2, parameters.omega_sat], [parameters.sigma_sat, 0.0]]
        chempots = [muB / 3, muB / 3, muB, muB, muB, muB] # u, d, n, p, n*, p*
        # Resolve the sigma field
        meson_res = eos_module.resolve_meson_fields(chempots, parameters,
                                                            guesses, args.rel_tol)
        sigma, omega = meson_res[0]

        pressure = eos_module.evaluate_pressure(chempots, parameters, meson_res[0]) - \
            parameters.vacuum_pressure
        
        chempots_eff = [mu - omega for mu in chempots]

        densities = 6 * [0]
        # u, d
        quark_mass = m_q_bare - sigma
        for i in range(2):
            n_c = 3
            k_f_sqr = chempots_eff[i] ** 2 - quark_mass ** 2
            if k_f_sqr > 0:
                densities[i] = n_c * (k_f_sqr ** 1.5) / (3 * np.pi ** 2)
        # n+, p+
        nucleon_mass = eos_module.nucleon_mass_parity_model(
            sigma, '+', parameters.a_coupling, parameters.b_coupling, parameters.mass_amplitude, m_q_bare)
        for i in range(2, 4):
            k_f_sqr = chempots_eff[i] ** 2 - nucleon_mass ** 2
            if k_f_sqr > 0:
                densities[i] = (k_f_sqr ** 1.5) / (3 * np.pi ** 2)
        # n*, p*
        nucleon_mass_star = eos_module.nucleon_mass_parity_model(
            sigma, '-', parameters.a_coupling, parameters.b_coupling, parameters.mass_amplitude, m_q_bare)
        for i in range(4, 6):
            k_f_sqr = chempots_eff[i] ** 2 - nucleon_mass_star ** 2
            if k_f_sqr > 0:
                densities[i] = (k_f_sqr ** 1.5) / (3 * np.pi ** 2)
        
        densities = np.array(densities)
        nB = 1 / 3 * np.sum(densities[:2]) + np.sum(densities[2:6])

        energy_density = nB * muB - pressure

        # Convert units
        energy_density = energy_density * mev3_fm3  # convert to MeV/fm^3
        pressure = pressure * mev3_fm3  # convert to MeV/fm^3
        densities = densities * mev3_fm3  # convert to 1/fm^3
        nB = nB * mev3_fm3  # convert to 1/fm^3
        output.write(
            f" {energy_density:20.10e} {pressure:20.10e} {nB:20.10e} {muB:20.10e} {densities[0]:20.10e} {densities[1]:20.10e} {densities[2]:20.10e} {densities[3]:20.10e} {densities[4]:20.10e} {densities[5]:20.10e} {quark_mass:20.10e} {nucleon_mass:20.10e} {nucleon_mass_star:20.10e} {sigma:20.10e} {omega:20.10e}\n")

    # muBs = np.linspace(args.start, args.finish, args.num_points)
    # sigmas = []
    # pressures = []
    # new_muBs = []
    # guesses_sigma = [m_q_bare / 2]#, parameters.sigma_sat]
    # for muB in muBs:
    #     chempots = [muB / 3, muB / 3, muB, muB, muB, muB] # u, d, n, p, n*, p*
    #     # Resolve the sigma field
    #     sigma_res = eos_module.resolve_sigma_field(chempots, parameters,
    #                                                         guesses_sigma, args.rel_tol)
    #     if sigma_res[2] > args.rel_tol:
    #         print (f"At muB {muB} MeV sigma resolution failed with error {sigma_res[2]}, status: {sigma_res[1]}")
        
    #     # if np.abs(sigma_res[0]) < 0.8 * np.abs(parameters.sigma_vac):
    #     #     guesses_sigma[1] = sigma_res[0]
    #     if sigma_res[0] is None:
    #         continue
        
    #     pressure = eos_module.evaluate_pressure(chempots, parameters, sigma_res[0]) - \
    #         parameters.vacuum_pressure
    #     sigmas.append(sigma_res)
    #     pressures.append(pressure)
    #     new_muBs.append(muB)
    # sigmas = np.array(sigmas)
    # pressures = np.array(pressures)

    # plot the results
    # plt.figure(figsize=(10, 6))
    # plt.errorbar(new_muBs, m_q_bare - sigmas[:, 0], yerr=sigmas[:, 2]
    #              * sigmas[:, 0], label='Sigma field', fmt='o', markersize=3)
    # plt.xlabel(r'$\mu_B$ (MeV)')
    # plt.ylabel(r'Quark constituent mass (MeV)')
    # plt.axvline(x=args.chem_pot_sat, color='red', linestyle='--', label='Chemical potential at saturation')
    # plt.savefig('sigma.pdf')

    # plt.figure(figsize=(10, 6))
    # plt.plot(new_muBs, pressures * mev3_fm3, label='Pressure', color='orange')
    # plt.xlabel(r'$\mu_B$ (MeV)')
    # plt.ylabel(r'Pressure (MeV/fm$^3$)')
    # plt.axvline(x=args.chem_pot_sat, color='red', linestyle='--', label='Chemical potential at saturation')
    # plt.savefig('pressure.pdf')


main()