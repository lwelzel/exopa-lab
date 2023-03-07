import os
import time
import shutil
import numpy as np


class inlist_editor():

    def __init__(self):
        # Dir names
        self.inlist_templates = 'inlist_templates/'
        self.inlist_hist = 'inlist_history/'

        # File names
        self.inlist1_template = 'inlist_1_create_template'
        self.inlist2_template = 'inlist_2_core_template'
        self.inlist3_template = 'inlist_3_reducemass_template'
        self.inlist4_template = 'inlist_4_setS_template'
        self.inlist5_template = 'inlist_5_evolve_template'

        # Constants
        self.msun = 1.98855e33
        self.rsun = 6.9598e10
        self.mjup = 1.8986e30
        self.rjup = 7.14e9
        self.Lsun = 3.9e33
        self.sigma = 5.67e-5
        self.au = 1.496e13
        self.mearth = 5.9722e27
        self.rearth = 6.378e8

    def create_planet(self, M_init, save_model_filename1, star_history_name1, log_directory):
        start_time = time.time()
        print('Creating initial planet')
        print(f'Initial mass set to: {M_init} M_earth')

        # Open the inlist file template
        template_location = self.inlist_templates + self.inlist1_template
        print(f'Using inlist template found in:')
        print(f'{template_location}')
        f = open(template_location, 'r')
        g = f.read()
        f.close()

        # Change parameters
        g = g.replace('<<save_model_filename1>>', f'\"{save_model_filename1}\"')
        g = g.replace('<<initial_mass>>', str(M_init * self.mearth))
        g = g.replace('<<star_history_name1>>', f'\"{star_history_name1}\"')
        g = g.replace('<<log_directory>>', f'\"{log_directory}\"')

        # Copy to inlist file from which MESA will run
        inlist_hist_location = self.inlist_hist + f'inlist_1_create_{M_init}Me'
        h = open(inlist_hist_location, 'w')
        h.write(g)
        h.close()
        shutil.copyfile(inlist_hist_location, "inlist")

        # Run MESA
        # os.system('bash')
        os.system('./clean')
        os.system('./mk')
        os.system('./rn')
        run_time = time.time() - start_time
        print("Run time for create_planets in sec = ", run_time)

    def add_core(self, M_core, Rho_core, save_model_filename1, save_model_filename2, star_history_name2, log_directory):
        start_time = time.time()
        print('Adding core to planet')
        print(f'Core mass set to: {M_core} M_earth')
        print(f'Core density set to: {Rho_core} g/cm3')

        # Open the inlist file template
        template_location = self.inlist_templates + self.inlist2_template
        print(f'Using inlist template found in:')
        print(f'{template_location}')
        f = open(template_location, 'r')
        g = f.read()
        f.close()

        # Change parameters
        g = g.replace('<<save_model_filename1>>', f'\"{save_model_filename1}\"')
        g = g.replace('<<save_model_filename2>>', f'\"{save_model_filename2}\"')
        g = g.replace('<<M_core>>', str(np.format_float_scientific(M_core * self.mearth / self.msun, 3)))
        g = g.replace('<<Rho_core>>', str(Rho_core))
        g = g.replace('<<star_history_name2>>', f'\"{star_history_name2}\"')
        g = g.replace('<<log_directory>>', f'\"{log_directory}\"')

        # Copy to inlist file from which MESA will run
        inlist_hist_location = self.inlist_hist + f'inlist_2_core_{M_core}Me'
        h = open(inlist_hist_location, 'w')
        h.write(g)
        h.close()
        shutil.copyfile(inlist_hist_location, "inlist")

        # Run MESA
        # os.system('bash')
        os.system('./clean')
        os.system('./mk')
        os.system('./rn')
        run_time = time.time() - start_time
        print("Run time for add_core in sec = ", run_time)

    def reduce_mass(self, M_new, save_model_filename2, save_model_filename3, star_history_name3, log_directory):
        start_time = time.time()
        print('Reducing planet mass')
        print(f'New planet total mass set to: {M_new} M_sun')

        # Open the inlist file template
        template_location = self.inlist_templates + self.inlist3_template
        print(f'Using inlist template found in:')
        print(f'{template_location}')
        f = open(template_location, 'r')
        g = f.read()
        f.close()

        # Change parameters
        g = g.replace('<<save_model_filename2>>', f'\"{save_model_filename2}\"')
        g = g.replace('<<save_model_filename3>>', f'\"{save_model_filename3}\"')
        g = g.replace('<<M_new>>', str(M_new))
        g = g.replace('<<star_history_name3>>', f'\"{star_history_name3}\"')
        g = g.replace('<<log_directory>>', f'\"{log_directory}\"')

        # Copy to inlist file from which MESA will run
        inlist_hist_location = self.inlist_hist + f'inlist_3_reducemass_{M_new}Sun'
        h = open(inlist_hist_location, 'w')
        h.write(g)
        h.close()
        shutil.copyfile(inlist_hist_location, "inlist")

        # Run MESA
        # os.system('bash')
        os.system('./clean')
        os.system('./mk')
        os.system('./rn')
        run_time = time.time() - start_time
        print("Run time for reduce_mass in sec = ", run_time)

    def set_entropy(self, new_L_center, save_model_filename3, save_model_filename4, star_history_name4, log_directory):
        start_time = time.time()
        print('Setting artificial entropy')
        print(f'Artificial luminosity of center of planet set to: {new_L_center} erg/s')

        # Open the inlist file template
        template_location = self.inlist_templates + self.inlist4_template
        print(f'Using inlist template found in:')
        print(f'{template_location}')
        f = open(template_location, 'r')
        g = f.read()
        f.close()

        # Change parameters
        g = g.replace('<<save_model_filename3>>', f'\"{save_model_filename3}\"')
        g = g.replace('<<save_model_filename4>>', f'\"{save_model_filename4}\"')
        g = g.replace('<<new_L_center>>', str(new_L_center))
        g = g.replace('<<star_history_name4>>', f'\"{star_history_name4}\"')
        g = g.replace('<<log_directory>>', f'\"{log_directory}\"')

        # Copy to inlist file from which MESA will run
        inlist_hist_location = self.inlist_hist + f'inlist_4_newS_{new_L_center}'
        h = open(inlist_hist_location, 'w')
        h.write(g)
        h.close()
        shutil.copyfile(inlist_hist_location, "inlist")

        # Run MESA
        # os.system('bash')
        os.system('./clean')
        os.system('./mk')
        os.system('./rn')
        run_time = time.time() - start_time
        print("Run time for set_entropy in sec = ", run_time)

    def evolve_planet(self, final_L_center, save_model_filename4, save_model_filename5, star_history_name5, log_directory):
        start_time = time.time()
        print('Evolving planet')
        print(f'Luminosity of center of planet set to: {final_L_center} erg/s')

        # Open the inlist file template
        template_location = self.inlist_templates + self.inlist5_template
        print(f'Using inlist template found in:')
        print(f'{template_location}')
        f = open(template_location, 'r')
        g = f.read()
        f.close()

        # Change parameters
        g = g.replace('<<save_model_filename4>>', f'\"{save_model_filename4}\"')
        g = g.replace('<<save_model_filename5>>', f'\"{save_model_filename5}\"')
        g = g.replace('<<final_L_center>>', str(final_L_center))
        g = g.replace('<<star_history_name5>>', f'\"{star_history_name5}\"')
        g = g.replace('<<profile_data_prefix>>', f'\"{save_model_filename5}\"')
        g = g.replace('<<log_directory>>', f'\"{log_directory}\"')

        # Copy to inlist file from which MESA will run
        inlist_hist_location = self.inlist_hist + f'inlist_5_evolve_{final_L_center}'
        h = open(inlist_hist_location, 'w')
        h.write(g)
        h.close()
        shutil.copyfile(inlist_hist_location, "inlist")

        # Run MESA
        # os.system('bash')
        os.system('./clean')
        os.system('./mk')
        os.system('./rn')
        run_time = time.time() - start_time
        print("Run time for evolve_planet in sec = ", run_time)
