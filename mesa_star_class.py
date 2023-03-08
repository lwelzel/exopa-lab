import numpy as np
import pandas as pd
import multiprocessing as mp
from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm
from pprint import pformat, pprint
from textwrap import indent
import gc
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
import matplotlib.ticker as ticker
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400
import matplotlib.pyplot as plt
from gc import collect

collect()

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]

class MESATimeStep(object):
    def __init__(self, global_data, zone_data):
        """
        wrapper class for dealing with MESA timesteps
        one MESATimeStep object equals one output file (profile)
        :param global_data:
        :param zone_data:
        """
        self.global_data = global_data
        self.zone_data = zone_data
        # TODO: give this unpacked properties


class MESAStar(object):
    def __init__(self, loc=None,
                 which_global=None, which_zone=None,
                 use_multiprocessing=True):
        """
        wrapper class for dealing with MESA data
        :param loc:
        :param which_global:
        :param which_zone:
        :param use_multiprocessing:
        """
        # region SETUP
        self.data_loc = loc
        self.which_global = which_global
        self.which_zone = which_zone

        if self.which_global is None:
            self.which_global = ['model_number',
                                 "num_zones",
                                 "star_age",
                                 "Teff",
                                 "photosphere_L",
                                 "star_mass",
                                 "power_nuc_burn"]

        if self.which_zone is None:
            self.which_zone = ["zone",
                               "logT",
                               "logRho",
                               "logP",
                               "logR",
                               "mu",
                               "gradr",
                               "grada",
                               # "x",
                               # "y",
                               # "z",
                               # "pp",
                               # "cno",
                               # 'tri_alfa'
                               ]

        print("---=== Available MESA Data ===---")
        av_global_headers, av_zone_headers = self.read_file(which_global=None, which_zone=None)
        print("\tGlobal Data Available:")
        print(indent(pformat(av_global_headers.columns.tolist(), compact=True),
                     '        '))
        print("\tZone Data Available:")
        print(indent(pformat(av_zone_headers.columns.tolist(), compact=True),
                     '        '))
        print(f"---=== Available MESA Data ===---\n")
        del av_global_headers, av_zone_headers
        gc.collect()
        # endregion

        # region DATA READING
        print("---=== Reading MESA LOGS ===---")
        self.mesa_time_steps = self.read_data()
        print("\tSelected Global Data:")
        print(indent(pformat(self.mesa_time_steps[0].global_data.columns.tolist(), compact=True),
                     '        '))
        print("\tSelected Zone Data:")
        print(indent(pformat(self.mesa_time_steps[0].zone_data.columns.tolist(), compact=True),
                     '        '))
        print(f"---=== Finished Reading MESA LOGS ===---")

        self.global_properties = pd.concat([ts.global_data for ts in self.mesa_time_steps]).reset_index(drop=True)
        self.global_properties.reset_index(inplace=True)
        self.global_properties.rename(columns={'index': 'step'}, inplace=True)

        # TODO: look at concat docu, check indexing and automatic dim collapse
        # TODO: master df? this is already scary with RAM.
        #  attach global prop as head to zone? Could select any to extract info
        self.zone_properties = pd.concat([ts.zone_data for ts in self.mesa_time_steps],
                                         keys=self.global_properties['step'])

        self.mesa_time_steps = None

        # endregion

        # region DATA prep
        # indexing in hierarchical DFs: https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
        # print(self.global_properties.head(5).to_string())
        # print(self.zone_properties.head(5).to_string())
        # print(self.zone_properties.loc[0, "logR"])

        self.indices_core = [tuple(ind) for ind in
                             zip(self.global_properties['step'],
                                 self.global_properties['num_zones']-1)]

        self.ms_steps = self.zone_properties.loc[self.indices_core, "cno"].index[
            self.zone_properties.loc[self.indices_core, "cno"]>1e-3].to_numpy()
        self.ms_steps = np.array([ind[0] for ind in self.ms_steps])
        self.pms_steps =  self.indices_core[:self.ms_steps[0]]
        self.pms_steps = np.array([ind[0] for ind in self.pms_steps])
        self.subgiant_step =  self.zone_properties.loc[self.indices_core, "x"].index[
            self.zone_properties.loc[self.indices_core, "x"]<1e-10].to_numpy()[0]
        # RGB phase in log tc log rho c plotting


        self.he_flash_steps = self.zone_properties.loc[self.indices_core, "cno"].index[
            self.zone_properties.loc[self.indices_core, "tri_alfa"] > 10].to_numpy()
        self.he_flash_steps = np.array([ind[0] for ind in self.he_flash_steps])

        self.wd_steps = self.global_properties["power_nuc_burn"].index[
            self.global_properties["power_nuc_burn"] < 1e0].to_numpy()[-2:] # TODO: hardcoded, fix


         # TODO: define other stages similarly
        # endregion


    def read_data(self, loc='./LOGS/run-m_core-3_0-f-0_1'):
        files = [f for f in listdir(loc)
                 if isfile(join(loc, f))
                 if "profile" in f
                 if ".data" in f]
        files.sort(key=natural_sort_key)
        cores = mp.cpu_count()

        print(f"\tReading {len(files)} files using {cores} threads.")
        mesa_time_steps = []
        with mp.Pool(cores) as pool:
            with tqdm(total=len(files), desc="\tCollecting MESA data from LOGS") as pbar:
                for (g_data, z_data) in pool.imap(self.read_file, files):
                    mesa_time_steps.append(MESATimeStep(
                        global_data=g_data,
                        zone_data=z_data
                    ))
                    pbar.update()

        mesa_time_steps = np.array(mesa_time_steps)
        return mesa_time_steps

    def read_file(self, name=None,
                  which_global="custom", which_zone="custom",
                  loc='./LOGS/run-m_core-3_0-f-0_1/'):
        if which_global == "custom":
            which_global = self.which_global
        if which_zone == "custom":
            which_zone = self.which_zone
        if name is None:
            # TODO: rewrite so that if name=None return None
            name = "profile1.data"
        file_path = loc + name

        global_data = pd.read_csv(file_path, delim_whitespace=True,
                                  header=1, usecols=which_global,
                                  nrows=1)

        zone_data = pd.read_csv(file_path, delim_whitespace=True,
                                  header=1, usecols=which_zone,
                                  skiprows=[0,1,2]).reset_index(drop=True)

        # TODO: clean column headers
        return global_data, zone_data

    def get_zone_prop(self, name=None, zone=None):
        # TODO: check indexing for concat dfs
        prop_df = self.zone_properties[name, zone]
        return prop_df

    def plot_logtc_logrhoc(self):
        fig, ax = plt.subplots(tight_layout=True)

        x = self.zone_properties.loc[self.indices_core,"logT"]
        y = self.zone_properties.loc[self.indices_core,"logRho"]
        x = x.to_numpy()
        y = y.to_numpy()
        x = np.power(10, x)
        y = np.power(10, y)

        star_age = self.global_properties["star_age"].to_numpy()

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = Normalize(vmin=star_age.min(), vmax=star_age.max())

        lc = LineCollection(segments, cmap='viridis', norm=norm,
                            capstyle="round", joinstyle="round", alpha=1)
        lc.set_array(star_age)
        lc.set_linewidth(10)
        line = ax.add_collection(lc)
        # fig.colorbar(line, ax=ax, label=r"age [yr]", format=ticker.FuncFormatter(fmt))

        ax.loglog(x, y, c="white", solid_joinstyle="round", linewidth=2)
        ax.loglog(x, y, c="black", dash_joinstyle="round", linewidth=0.5, linestyle='dashed', label="Star Track")
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        # tc_cross =
        # ideal_track_deg = np.like()



        # region EOS
        # TODO: mean should be avg weighted by the time associated with the time
        timesteps = [star_age[step+1] - star_age[step]
                     for step in self.global_properties['step'].to_numpy()[:-1]]
        timesteps.insert(0, star_age[0])
        timesteps = np.array(timesteps)

        mu = self.zone_properties.loc[self.indices_core, "mu"]
        mu_avg = np.average(mu, weights=timesteps)

        _X = self.zone_properties.loc[self.indices_core, "x"]
        mu_e = 2 / (1 + _X)
        del _X
        mu_e_avg =  np.average(mu_e, weights=timesteps)

        func_rho_rad_ideal = lambda _T, _mu, _mu_e : (_T / (3.2e7 * _mu ** (-1 / 3))) ** 3
        func_rho_ideal_deg = lambda _T, _mu, _mu_e : (_T / (1.21e5 * _mu * _mu_e ** (-5 / 3))) ** (3 / 2)
        func_rho_deg_n_e = lambda _T, _mu, _mu_e : np.full_like(_T, fill_value=9.7e5 * _mu_e)
        func_rho_ideal_deg_e = lambda _T, _mu, _mu_e : (_T / (1.5e7 * _mu * _mu_e ** (-4 / 3))) ** (3)

        funcs = [func_rho_rad_ideal, func_rho_ideal_deg, func_rho_deg_n_e]
        names = ["Radiation-Ideal Gas", "Ideal Gas-NR Degenerate", "NR-ER Degenerate", "Ideal Gas-ER Degenerate"]
        colors = ["blue", "blueviolet", "deeppink", "crimson"]
        hatches = ["+++", "////", "....", "\\\\\\\\"]

        # find degenerate core -> RGB phase
        self.steps_deg_core =  np.flatnonzero(y > func_rho_ideal_deg(x, mu_avg, mu_e_avg))
        self.step_rgb = self.steps_deg_core[0]

        zams = np.vstack((x,y)).T[self.ms_steps[0]]
        ax.annotate("ZAMS", xy=zams, xytext=zams * np.array([1, 0.2]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))
        sub_giant = np.vstack((x,y)).T[self.ms_steps[-1]+2]
        ax.annotate("Subgiant\nbranch", xy=sub_giant, xytext=sub_giant * np.array([2, 0.3]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))


        sun = [1.5e7, 150]
        ax.scatter(x=sun[0], y=sun[1], s=50, marker="*", edgecolor='k', facecolor='w', linewidth= 0.5)
        ax.annotate("Solar Core", xy=sun, xytext=sun * np.array([0.3, 1]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))

        rgb = np.vstack((x,y)).T[self.step_rgb]
        ax.annotate("RGB", xy=rgb, xytext=rgb * np.array([0.15, 1]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))


        he_flash = np.vstack((x,y)).T[self.he_flash_steps[0]]
        ax.annotate("He-Flashes", xy=he_flash, xytext=he_flash * np.array([0.2, 2]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))

        agb = np.vstack((x,y)).T[self.he_flash_steps[-1]+1]
        ax.annotate("AGB", xy=agb, xytext=agb * np.array([1.2, 8]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))

        wd = np.vstack((x,y)).T[self.wd_steps[0]]
        ax.annotate("WD", xy=wd, xytext=wd * np.array([2, 1.5]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))

        print(mu.to_string)
        print(mu_e.to_string)

        c_range = np.linspace(0.48, 0.36, num=3)
        __mu_e = 1.17
        __mu = 0.62
        __c = 0.48
        r = 8.314e7
        knr = 1.0036e13
        G = 6.6725e-8
        m = 4E33
        Tcmax = __c**2*(G)**2*__mu*__mu_e**(5/3)*(m)**(4/3)/(r * knr)
        ideal_x_range = np.linspace(np.min(x), Tcmax, 20)

        ideal_const_rho = np.full_like(ideal_x_range, fill_value=((__c*G/knr)**3*__mu_e**5)*(m)**2)
        ideal_var_rho = ideal_x_range**3*(r)**3/(__c**3*(G)**3*(m)**2*__mu**3)

        ax.loglog(ideal_x_range, ideal_const_rho, c="black", solid_joinstyle="round", linewidth=2, alpha=0.6, solid_capstyle="round")
        ax.loglog(ideal_x_range, ideal_var_rho, c="black", solid_joinstyle="round", linewidth=2, alpha=0.6, solid_capstyle="round",
                  label="Ideal Track")

        ys = []
        x_rho = np.logspace(start=np.log10(0.1*xlim[0]), stop=np.log10(10*xlim[1]), num=10)
        for rho_func, name, color, hatch in zip(funcs, names, colors, hatches):
            # mean
            y = rho_func(x_rho, mu_avg, mu_e_avg)
            ax.loglog(x_rho, y, zorder=0,
                      solid_joinstyle="round", linewidth=1, linestyle='solid', c=color)

            extremes = self.get_extremes_1_dim_invariants(rho_func, x, mu, mu_e)
            _y = []
            for _mu, _mu_e in zip(extremes[0], extremes[1]):
                __y = rho_func(x_rho, _mu, _mu_e)
                _y.append(__y)
            # TODO: hatch and give label to this instead of mean line
            ys.append(_y)
            ax.fill_between(x_rho, *_y, alpha=0.25, facecolor=color, hatch=hatch, edgecolor=color, linewidth=0.5,
                            zorder=0, interpolate=True)

        names = ["Radiation", "Ideal Gas", "NR Degenerate", "ER Degenerate"]
        colors = ["blue", "blueviolet", "deeppink", "crimson"]
        hatches = ["+++", "....", "////", "\\\\\\\\"]

        ys.insert(0, np.zeros_like(_y))
        ys.append(np.full_like(_y, fill_value=1.1 * xlim[1]))

        for i, (y_pair, name, color, hatch) in enumerate(zip(ys, names, colors, hatches)):
            ax.fill_between(x_rho, y_pair[0], ys[i+1][1], label=name,
                            alpha=0.1, facecolor=color, hatch=hatch,
                            edgecolor=color, linewidth=0.0,
                            zorder=0, interpolate=True)

        # endregion

        ax.legend(fancybox=True, fontsize="small", framealpha=1.)
        ax.set_xlim(xlim[0], Tcmax*1.05)
        ax.set_ylim(ylim)

        # ax.set(title=r'$log(T_c)$ vs $log(\rho_c)$',
        #        xlabel=r"$log(T_c)$",
        #        ylabel=r"$log(\rho_c)$")
        ax.set(xlabel=r"$log(T_c)~[K]$",
               ylabel=r"$log(\rho_c)~[g~cm^{-1}]$")
        ax.grid(axis="both", which="major", alpha=0.5)
        ax.grid(axis="both",which="minor", alpha=0.1)
        fig.savefig("logtc_logrhoc.png", dpi=500)
        fig.show()

    def plot_hzsp_russ(self):
        fig, ax = plt.subplots(tight_layout=True)
        x = self.global_properties["Teff"]
        y = self.global_properties["photosphere_L"]
        x = x.to_numpy()
        y = y.to_numpy()
        star_age = self.global_properties["star_age"].to_numpy()

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = Normalize(vmin=star_age.min(), vmax=star_age.max())

        lc = LineCollection(segments, cmap='viridis', norm=norm,
                            capstyle="round", joinstyle="round", alpha=1)
        lc.set_array(star_age)
        lc.set_linewidth(10)
        line = ax.add_collection(lc)
        # fig.colorbar(line, ax=ax, label=r"age [yr]")
        fig.colorbar(line, ax=ax, label=r"age [yr]", format=ticker.FuncFormatter(fmt))

        ax.loglog(x, y, c="white", solid_joinstyle="round", linewidth=2)
        ax.loglog(x, y, c="black", dash_joinstyle="round", linewidth=0.5, linestyle='dashed')



        zams = np.vstack((x,y)).T[self.ms_steps[0]]
        ax.annotate("ZAMS", xy=zams, xytext=zams * np.array([1, 0.2]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))
        sub_giant = np.vstack((x,y)).T[self.ms_steps[-1]+1]
        ax.annotate("Subgiant\nbranch", xy=sub_giant, xytext=sub_giant * np.array([2, 1]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))


        sun = [5778, 1]
        ax.scatter(x=sun[0], y=sun[1], s=50, marker="*", edgecolor='k', facecolor='w', linewidth= 0.5)
        ax.annotate("Sun", xy=sun, xytext=sun * np.array([2, 0.5]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))

        rgb = np.vstack((x,y)).T[self.step_rgb]
        ax.annotate("RGB", xy=rgb, xytext=rgb * np.array([2, 1.2]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))


        he_flash = np.vstack((x,y)).T[self.he_flash_steps[0]]
        ax.annotate("He-Flashes", xy=he_flash, xytext=he_flash * np.array([3, 0.5]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))

        agb = np.vstack((x,y)).T[self.he_flash_steps[-1]+1]
        ax.annotate("AGB", xy=agb, xytext=agb * np.array([2, 2]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))

        wd = np.vstack((x,y)).T[self.wd_steps[0]]
        ax.annotate("WD", xy=wd, xytext=wd * np.array([0.7, 1]), c="black", fontsize="x-small",
                    arrowprops={"arrowstyle": 'simple', "fc": "w", "ec": "k", "linewidth": 0.5},
                    bbox=dict(boxstyle="square", fc="w", linewidth= 0.5))


        ax.invert_xaxis()
        # ax.set(title=r'Hertzsprung-Russell Diagram',
        #        xlabel=r"$T_{eff}~ [K]$",
        #        ylabel=r"$L_{photosphere}~ [L_\odot]$")
        ax.set(xlabel=r"$T_{eff}~ [K]$",
               ylabel=r"$L_{photosphere}~ [L_\odot]$")
        ax.grid(axis="both", which="major", alpha=0.5)
        ax.grid(axis="both", which="minor", alpha=0.1)
        fig.savefig("Hertzsprung-Russell.png", dpi=500)
        fig.show()

    def plot_conv_grads(self, age=None, steps=None, zoom=False, title=""):
        if np.logical_and(age is None, steps is None):
            raise KeyError("Either age or step has to be set.")
        elif age is not None:
            steps = self.global_properties.iloc[(self.global_properties['star_age']
                                                   - age).abs().argsort()[0]].index.tolist()
        elif type(steps) == int:
            steps = np.array([steps])
        if zoom:
            fig, axes = plt.subplots(3,1, tight_layout=True, figsize=(10,10))
            ax = axes[0]
        else:
            fig, ax = plt.subplots(tight_layout=True)
        # TODO: plotting of multiple timesteps simultaneously
        try:
            star_age = self.global_properties["star_age"].loc[steps].to_numpy() # check
        except AttributeError:
            star_age = np.array([self.global_properties["star_age"].loc[steps]])
        cmap = mpl.cm.get_cmap('cool')
        norm = LogNorm(vmin=star_age.min(), vmax=star_age.max()) # potentially use LogNorm
        colors =  np.array([cmap(norm(age)) for age in star_age])

        cmap = mpl.cm.get_cmap('autumn')
        norm = LogNorm(vmin=star_age.min(), vmax=star_age.max()) # potentially use LogNorm
        colors2 =  np.array([cmap(norm(age)) for age in star_age])

        for step, color, color2, name in zip(steps, colors, colors2, star_age):
            x = self.zone_properties.loc[step, "logR"]
            y1 = self.zone_properties.loc[step, "grada"]
            y2 = self.zone_properties.loc[step, "gradr"]
            x = x.to_numpy()
            y1 = y1.to_numpy()
            y2 = y2.to_numpy()
            x = np.power(10, x)
            # ax.loglog(x, y1, linestyle="solid", c=color,
            #             label=f"Star age: {fmt(name, 1)} yr")
            ax.loglog(x, y1, linestyle="solid", c=color)
            ax.loglog(x, y2,  linestyle="dashed", c=color2)
            ax.fill_between(x, y1, y2, where=(y1 > y2), color=color, alpha=0.3, hatch="++++", linewidth=0.0)
            ax.fill_between(x, y1, y2, where=(y1 < y2), color=color2, alpha=0.3, hatch="////", linewidth=0.0)

            ax.set(xlabel=r"$R~[R_\odot]$",
                   ylabel=r"$\nabla$")
            ax.grid(b=True, axis="both", which="major", alpha=0.5, color="gray")
            ax.grid(b=True, axis="both",which="minor", alpha=0.1, color="gray")

            if zoom:
                xlims = [[0, 0.1*np.max(x)], [0.9*np.max(x), 1.001*np.max(x)]]
                for axs, xlim in zip(axes[1:], xlims):
                    axs.loglog(x, y1, linestyle="solid", c=color)
                    axs.loglog(x, y2, linestyle="dashed", c=color2)
                    axs.fill_between(x, y1, y2, where=(y1 > y2), color=color, alpha=0.3, hatch="++++", linewidth=0.0)
                    axs.fill_between(x, y1, y2, where=(y1 < y2), color=color2, alpha=0.3, hatch="////", linewidth=0.0)
                    axs.set_xlim(xlim)
                    axs.set(xlabel="R",
                           ylabel="grad")
                    # axs.legend()
                    axs.grid(b=True, axis="both", which="major", alpha=0.5, color="gray")
                    axs.grid(b=True, axis="both", which="minor", alpha=0.1, color="gray")

        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        y1 = np.full_like(xlim, fill_value=ylim[1]*10)
        y2 = np.full_like(xlim, fill_value=ylim[1]*10) + [-1, 1]
        ax.loglog(xlim, y1, linestyle="solid", label=r"$\nabla_{ad}$", c=color)
        ax.loglog(xlim, y2, linestyle="dashed", label=r"$\nabla_{rad}$", c=color2)

        ax.fill_between(xlim, y1, y2, where=(y1 > y2), color=color, alpha=0.3, hatch="++++", linewidth=0.0,
                        label="Radiative")
        ax.fill_between(xlim, y1, y2, where=(y1 < y2), color=color2, alpha=0.3, hatch="////", linewidth=0.0,
                        label="Convective")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(title=f"Star age: {fmt(name, 1)} yr")
        # TODO: mention where in title
        # plt.suptitle(f'logR vs grada, gradr') # \nage: {fmt(self.global_properties.loc[step, "star_age"], 0)}yr
        fig.savefig(f"convection_"+title+".png", dpi=500) # age_{self.global_properties.loc[step, 'star_age']:.1e} yr
        fig.show()

    @staticmethod
    def get_extremes_1_dim_invariants(func, xs, *args):
        """

        :param func: some function which takes n arguments
        of which k are invariants and l are are variables varying in the same dim
        :param xs: array of invariants of len k or int k
        :param args: n arrays of shape (m)
        :return:
        """
        if type(xs) != int:
            ys = func(np.ones_like(xs), *args)
        elif type(xs) == int:
            ys = func(np.ones(xs), *args)
        else:
            raise TypeError

        if type(ys) == type(pd.DataFrame()):
            ys = ys.to_numpy()

        ind_min = np.argmin(ys)
        ind_max = np.argmax(ys)
        extremes = np.array([[arg[ind_min], arg[ind_max]] for arg in args])
        return extremes


if __name__ == "__main__":
    gc.collect()

    # region 2 solar masses
    ms2 = MESAStar()
    ms2.plot_logtc_logrhoc()
    ms2.plot_hzsp_russ()
    # ms2.plot_conv_grads(steps=3, zoom=False, title="PMS")
    # ms2.plot_conv_grads(steps=6, zoom=False, title="MS")
    # ms2.plot_conv_grads(steps=ms2.pms_steps, zoom=True)
    # ms2.plot_conv_grads(steps=ms2.ms_steps, zoom=True)

    # endregion
    del ms2
    gc.collect()

