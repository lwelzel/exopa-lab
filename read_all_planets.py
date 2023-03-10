import numpy as np
from mesa_planet_class import MESAPlanet
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import h5py
from copy import deepcopy
from pathos import multiprocessing as mp

def get_all_configs(loc=None):
    """
    Returns a configuration dictionary for all subdirectories in LOGS (or other folder).
    Run data is extracted from folder name.
    Success is inferred from the presence of a FileNotFoundError.txt file that is placed into the directory if error.

    config = {
        'name':     'run-m_core-0_1-f-0_001',
        'path':     Path(),
        'success':  0,      # 0 -> failed, 1 -> success
        'm_core':   0.1,
        'f':        0.001
    }

    """
    if loc is None:
        loc = "./LOGS/"
    loc = Path(loc)

    dirs = [x for x in loc.iterdir() if x.is_dir()]
    vals = np.array([[item.replace("_", ".") for item in list(map(d.name.split("-").__getitem__, [2, 4]))] for d in dirs], dtype=float)
    success = [1 if not (d / "FileNotFoundError.txt").is_file() else 0 for d in dirs]

    experiments = [{
        "name": key.name,
        "path": str(key),
        "success": success[i],
        "m_core": vals[i, 0],
        "f": vals[i, 1],
    } for i, key in enumerate(dirs)]

    print(f"\n{len(success) - np.sum(success)} experiments out of {len(success)} failed.\n")

    return experiments

def get_experiment_data(loc=None):
    if loc is None:
        loc = "./LOGS/"
    loc = Path(loc)

    configs = get_all_configs(loc=loc)
    successful_configs = [config for config in configs if config["success"] == 1]
    failed_configs = [config for config in configs if config["success"] == 0]
    cores = int(0.9 * mp.cpu_count())
    with mp.Pool(cores) as pool:
        with tqdm(total=len(successful_configs), desc="\tCollecting MESA data from LOGS") as pbar:
            i_arr = np.arange(len(successful_configs))
            for (i, config) in pool.imap(make_planet_data, zip(i_arr, successful_configs)):
                successful_configs[i] = z = {**successful_configs[i], **config}
                pbar.update()
        pool.close()

    return successful_configs, failed_configs

def make_planet_data(theta):
    i, config = theta
    this_config = {}
    try:
        mesa_planet = MESAPlanet(loc=str(config["path"]))

        for k, v in mesa_planet.hist_header.items():
            this_config[k] = v
        this_config["history"] = mesa_planet.hist_data
        this_config["global_properties"] = mesa_planet.global_properties
        this_config["zone_properties"] = mesa_planet.zone_properties
    except BaseException:
        pass
    return i, this_config

def save_experiments(successful_configs, failed_configs, file_name=None):
    if file_name is None:
        file_name = 'exoplanet_A_MESA_lab_GS_RS.hdf5'

    meta_data = [
        'name', 'success', 'm_core', 'f', 'version_number', 'initial_mass', 'initial_z',
        'burn_min1', 'burn_min2'
    ]

    with h5py.File(file_name, 'w') as f:
        for i, config in enumerate(successful_configs):
            grp = f.create_group(config["name"])

            # make meta data dict
            attrs = dict((k, config[k]) for k in meta_data if k in config)
            attrs["path"] = str(config["path"])

            grp.attrs.update(attrs)

    with pd.HDFStore(file_name, mode="a") as hdf:
        for config in successful_configs:
            hdf.put(value=config["history"], key=config["name"]+"/history", data_columns=True)
            hdf.put(value=config["global_properties"], key=config["name"]+"/global_properties", data_columns=True)
            hdf.put(value=config["zone_properties"], key=config["name"]+"/zone_properties", data_columns=True)

    df_failed = pd.DataFrame.from_dict(failed_configs)
    df_failed.to_hdf(file_name, key="failed_configs", mode="a")

def read_experiments_hdf5(loc=None):
    if loc is None:
        loc = 'exoplanet_A_MESA_lab_GS.hdf5'

    def print_attrs(name, obj):
        # Create indent
        shift = name.count('/') * '    '
        item_name = name.split("/")[-1]
        print(shift + item_name)

    data = {}

    # TODO: the file is open two times which is really bad, once as hdf and once as f
    # pd cant read attrs so I dont see a way around it
    hdf = pd.HDFStore(loc, mode="r")
    data["failed_configs"] = hdf.get("failed_configs")

    entries = list(hdf.walk())[0][1]
    # print(hdf.info())

    f = h5py.File(loc, 'r')
    for i, entry in tqdm(enumerate(entries),
                         desc="Reading headers from HDF5 file: ", total=len(entries), leave=False):
        data[entry] = {}
        data[entry]["header"] = {key:value for (key, value) in f[entry].attrs.items()}

    for i, entry in tqdm(enumerate(entries),
                         desc="Reading data from HDF5 file: ", total=len(entries), leave=False):
        # data[entry]["info"] = hdf.info()
        data[entry]["history"] = hdf.get(entry + "/history")
        data[entry]["global_properties"] = hdf.get(entry + "/global_properties")
        data[entry]["zone_properties"] = hdf.get(entry + "/zone_properties")

    # deepcopy the dict to save to memory (not ref to files) then close the two open files
    data = deepcopy(data)
    hdf.close()
    f.close()

    return data



def test_num_experiments():
    from itertools import product

    m_p_core_list = [0.1, 0.5, 1., 1.2, 1.5, 2., 3., 5., 7., 10., 12., 15., 20., 25., 30.]
    f_list = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

    should = list(product(m_p_core_list, f_list))
    print(len(should))

if __name__ == "__main__":
    # successful_configs, failed_configs = get_experiment_data()
    # save_experiments(successful_configs, failed_configs)
    __ = read_experiments_hdf5(loc="exoplanet_A_MESA_lab_GS_RS.hdf5")