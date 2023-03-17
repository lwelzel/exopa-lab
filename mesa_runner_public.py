from inlist_editor import inlist_editor
from os import mkdir
from pathlib import Path
from shutil import copyfile
import numpy as np

def run_experiment(ie, _m_p_core, _f):
    """
    Run a MESA experiment for a specific core mass and envelope mass fraction.
    :param ie: InlistEditor instance
    :param _m_p_core:
    :param _f:
    """
    # TODO: remove NotImplementedError at EOF for full run
    _dir = Path(f"LOGS/run-m_core-{_m_p_core}-f-{_f}/".replace(".", "_"))
    mkdir(_dir)
    _name = f"m_core-{_m_p_core}-f-{_f}-".replace(".", "_")

    _file_path = Path(_dir / _name)

    print(f"\nRunning {_name}\n"
          f"In directory: {_dir}\n\n")

    try:
        #### STEP 1 - CREATE A PLANET ####
        M_init = 30  # earth mass
        save_model_filename1 = str(_dir / f'{_name}planet_1_create.mod')
        star_history_name1 = f'{_name}history_1_create.data'
        ie.create_planet(M_init, save_model_filename1, star_history_name1, log_directory=str(_dir))

        #### STEP 2 - ADD A CORE TO YOUR PLANET ####
        M_core = _m_p_core  # earth mass
        R_eff = M_core ** 0.27
        # Rho_core = 6.771 # g / cm3 # TODO: estimate mean core density
        # TODO: EQUATION FOR RHO_CORE GOES HERE # corr for M_e / R_e^3 to g / cm3
        save_model_filename2 = str(_dir / f'{_name}planet_2_core.mod')
        star_history_name2 = f'{_name}history_2_core.data'
        ie.add_core(M_core, Rho_core, save_model_filename1, save_model_filename2, star_history_name2, log_directory=str(_dir))

        #### STEP 3 - CHANGE THE ENVELOPE MASS FRACTION ####
        # M_new = 1.001e-05 # solar mass
        f_env = _f
        # TODO: EQUATION FOR M_new GOES HERE  # corr for M_new in solar masses
        save_model_filename3 = str(_dir / f'{_name}planet_3_reducemass.mod')
        star_history_name3 = f'{_name}history_3_reducemass.data'
        ie.reduce_mass(M_new, save_model_filename2, save_model_filename3, star_history_name3, log_directory=str(_dir))

        #### STEP 4 - CHANGE THE ENTROPY AT PLANET BASE ####
        new_L_center = 2e27  # erg/s
        save_model_filename4 = str(_dir / f'{_name}planet_4_setS.mod')
        star_history_name4 = f'{_name}history_4_setS.data'
        ie.set_entropy(new_L_center, save_model_filename3, save_model_filename4, star_history_name4, log_directory=str(_dir))

        #### STEP 5 - EVOLVE THE PLANET ####
        # final_L_center = 8.964e20 # erg/s # TODO: change
        # TODO: EQUATION FOR FINAL_L_CENTER GOES HERE  # corr for M_core in g
        save_model_filename5 = str(_dir / f'{_name}planet_5_evolve.mod')
        star_history_name5 = f'{_name}history_5_evolve.data'
        ie.evolve_planet(final_L_center, save_model_filename4, save_model_filename5, star_history_name5, log_directory=str(_dir))

        # creates a copy of star_history_name5 (evolve) to better fit with existing py modules
        copyfile(_dir / star_history_name5, _dir / "history.data")

    except FileNotFoundError as e:
        with open(_dir / 'FileNotFoundError.txt', 'w') as f:
            f.write(str(e))

    except FileExistsError as e:
        with open(_dir / 'FileExistsError.txt', 'w') as f:
            f.write(str(e))

    # REMOVE FOR FULL RUN
    # raise NotImplementedError

def run_grid_search():
    ie = inlist_editor()

    m_p_core_list = [3., 5., 7., 10., 12.]
    f_list = [0.1, 0.01]

    # m_p_core_list = [0.1, 0.5, 1., 1.2, 1.5, 2., 3., 5., 7., 10., 12., 15., 20., 25., 30.]
    # f_list = [0.5, 0.2, 0.05, 0.02, 0.005, 0.002, 0.001]

    for _m_p_core in m_p_core_list:
        for _f in f_list:
            run_experiment(ie, _m_p_core, _f)


def run_random_search(budget=100):
    ie = inlist_editor()

    # assignment
    # m_p_core_bounds = [3., 12.]
    # f_bounds = [0.01, 0.1]

    # wide
    m_p_core_bounds = [1.1, 15.]
    f_bounds = [0.005, 0.2]

    rng = np.random.default_rng()

    m_p_core_list = rng.uniform(m_p_core_bounds[0], m_p_core_bounds[1], budget)
    f_list = np.power(10, rng.uniform(np.log10(f_bounds[0]), np.log10(f_bounds[1]), budget))

    for (_m_p_core, _f) in zip(m_p_core_list, f_list):
        run_experiment(ie, _m_p_core, _f)

if __name__ == "__main__":
    run_grid_search()






