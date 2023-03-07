from inlist_editor import inlist_editor
from os import mkdir
from pathlib import Path
from shutil import copyfile

# Importing the inlist editor 
ie = inlist_editor()

m_p_core_list = [3., 5., 7., 10., 12.]
f_list = [0.1, 0.01]

# DONE: remove NotImplementedError at EOF for full run

for _m_p_core in m_p_core_list:
    for _f in f_list:
        _dir = Path(f"LOGS/run-m_core-{_m_p_core}-f-{_f}/".replace(".", "_"))
        mkdir(_dir)
        _name = f"m_core-{_m_p_core}-f-{_f}-".replace(".", "_")

        _file_path = Path(_dir / _name)

        print(f"\nRunning {_name}\n"
              f"In directory: {_dir}\n\n")

        #### STEP 1 - CREATE A PLANET ####
        M_init = 30  # earth mass
        save_model_filename1 = str(_dir / f'{_name}planet_1_create.mod')
        star_history_name1 = f'{_name}history_1_create.data'
        ie.create_planet(M_init, save_model_filename1, star_history_name1, log_directory=str(_dir))

        #### STEP 2 - ADD A CORE TO YOUR PLANET ####
        M_core = _m_p_core  # earth mass
        R_eff = M_core ** 0.27
        # Rho_core = 6.771 # g / cm3 # TODO: estimate mean core density
        Rho_core = M_core * 5.927e27 / (4 / 3 * 3.14 * (R_eff * 6.378e8) ** 3)  # corr for M_e / R_e^3 to g / cm3
        save_model_filename2 = str(_dir / f'{_name}planet_2_core.mod')
        star_history_name2 = f'{_name}history_2_core.data'
        ie.add_core(M_core, Rho_core, save_model_filename1, save_model_filename2, star_history_name2, log_directory=str(_dir))

        #### STEP 3 - CHANGE THE ENVELOPE MASS FRACTION ####
        # M_new = 1.001e-05 # solar mass
        f_env = _f
        M_new = M_core / (1 - f_env) * 3.0027e-6  # corr for M_new in solar masses
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
        final_L_center = 5e-8 * M_core * 5.927e27  # corr for M_core in g
        save_model_filename5 = str(_dir / f'{_name}planet_5_evolve.mod')
        star_history_name5 = f'{_name}history_5_evolve.data'
        ie.evolve_planet(final_L_center, save_model_filename4, save_model_filename5, star_history_name5, log_directory=str(_dir))

        # creates a copy of star_history_name5 (evolve) to better fit with existing py modules
        copyfile(_dir / star_history_name5, _dir / "history.data")

        # REMOVE FOR FULL RUN
        # raise NotImplementedError