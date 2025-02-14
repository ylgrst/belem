import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import math as m
from simcoon import simmit as sim
from simcoon.parameter import Parameter
from simcoon.data import Data, write_input_and_tab_files, write_files_exp, write_files_num
from simcoon import parameter, data
from typing import List, Tuple, Union
import os
import shutil
import glob
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import importlib.resources

DEFAULT_CONSTANTS_FILE = (importlib.resources.files(belem) / "default_simcoon_files" / "constants.inp").as_posix()
DEFAULT_WEIGHTS_FILE = (importlib.resources.files(belem) / "default_simcoon_files" / "file_weights.inp").as_posix()
DEFAULT_GEN_FILE = (importlib.resources.files(belem) / "default_simcoon_files" / "gen0.inp").as_posix()
DEFAULT_IDENT_CONTROL_FILE = (importlib.resources.files(belem) / "default_simcoon_files" / "ident_control.inp").as_posix()
DEFAULT_IDENT_ESSENTIALS_FILE = (importlib.resources.files(belem) / "default_simcoon_files" / "ident_essentials.inp").as_posix()
DEFAULT_MATERIAL_FILE = (importlib.resources.files(belem) / "default_simcoon_files" / "material.dat").as_posix()
DEFAULT_SOLVER_CONTROL_FILE = (importlib.resources.files(belem) / "default_simcoon_files" / "solver_control.inp").as_posix()
DEFAULT_SOLVER_ESSENTIALS_FILE = (importlib.resources.files(belem) / "default_simcoon_files" / "solver_essentials.inp").as_posix()

def prepare_epchg_identification(data_to_identify: List[Data], list_columns_to_compare: List[List[Union[str, int]]],
                                 parameters_to_optimize: List[Parameter],
                                 elastic_params: npt.NDArray[np.float_], n_iso_hard: int, n_kin_hard: int, criteria: str,
                                 criteria_params: npt.NDArray[np.float_], basedir: str) -> None:

    exp_dir = basedir + "/exp_data"
    data_dir = basedir + "/data"
    keys_dir = basedir + "/keys"

    _create_ident_folders(basedir)
    write_input_and_tab_files(data_to_identify, exp_dir, data_dir)
    write_files_exp(data_to_identify, exp_dir, data_dir)
    write_files_num(data_to_identify, list_columns_to_compare, data_dir)
    _copy_all_default_files(data_dir)
    _write_parameter_input_file(parameters_to_optimize, data_dir)
    _write_epchg_material_input_file(elastic_params, criteria, criteria_params, n_iso_hard, n_kin_hard,
                                     parameters_to_optimize, keys_dir)
    i = 1
    for sim_type in ["tension", "bitension", "shear"]:
        _write_path_id_file(f"path_id_{i:02}.txt", sim_type, data_dir)
        i += 1

def compute_epchg_loss(parameters_to_optimize: List[Parameter], elastic_params: npt.NDArray[np.float_],
                       n_iso_hard: int, n_kin_hard: int, criteria: str,
                       criteria_params: npt.NDArray[np.float_], path_dir: str = "data/", num_dir: str = "num_data/",
                       results_dir: str = "results_id/") -> float:

    list_path_file = glob.glob("path_id_*.txt", root_dir=path_dir)
    list_output_file = []

    umat_name = 'EPCHG'  # This is the 5 character code for the elastic-plastic subroutine
    nstatev = 9 + 12*n_kin_hard  # The number of scalar variables required, only the initial temperature is stored here

    psi_rve = 0.
    theta_rve = 0.
    phi_rve = 0.

    match criteria:
        case "mises":
            criteria_id = 0
        case "hill":
            criteria_id = 1
        case "dfa":
            criteria_id = 2
        case "anisotropic":
            criteria_id = 3
        case _:
            raise ValueError("Unknown plasticity criterion. Available options are mises, hill, dfa and anisotropic")

    props_elast_yield = np.append(elastic_params, np.array([parameters_to_optimize[0]]))
    props_elast_yield_gen_params = np.append(props_elast_yield, np.array([n_iso_hard, n_kin_hard, criteria_id]))
    props_temp = np.append(props_elast_yield_gen_params, np.array([parameters_to_optimize[i+1] for i in range(len(parameters_to_optimize)-1)]))
    props = np.append(props_temp, criteria_params)

    for i in range(len(list_path_file)):
        output_file = f"results_EPCHG{i:02}.txt"
        copied_output_file = f"results_EPCHG{i:02}_global-0.txt"
        list_output_file.append(copied_output_file)
        sim.solver(umat_name, props, nstatev, psi_rve, theta_rve, phi_rve, 0, 2, path_dir, results_dir, list_path_file[i],
                   output_file)
        shutil.copy(results_dir + copied_output_file, num_dir)

    c = sim.calc_cost(len(list_output_file), list_output_file)
    print(c)

    return c

def _write_parameter_input_file(list_parameters: List[Parameter], path: str = "data/") -> None:
    with open(path + "parameter.inp", "w+") as file:
        file.write("#Number\t#min\t#max\t#key\t#number_of_files\t#files\n")
        for parameter in list_parameters:
            file.write(str(parameter.number) + "\t" + str(parameter.bounds[0]) + "\t" + str(parameter.bounds[1]) + "\t"
                       + parameter.key + "\t" + str(len(parameter.sim_input_files)) + "\t")
            file.write(' '.join(str(val) for val in parameter.sim_input_files))
            file.write("\n")

def _write_dfa_material_input_file(dfa_params: npt.NDArray[np.float_], elastic_params: npt.NDArray[np.float_],
                                   list_parameters: List[Parameter], path: str = "keys/") -> None:
    with open(path + "material.dat", "w+") as file:
        file.write("Material\nName\tEPDFA\nNumber_of_material_parameters\t17\nNumber_of_internal_variables\t33\n\n#Orientation\npsi\t0\ntheta\t0\nphi\t0\n\n#Mechanical\n")
        file.write("E\t" + str(elastic_params[0]) + "\n")
        file.write("nu\t0.3\n")
        file.write("G\t" + str(elastic_params[1]) + "\n")
        file.write("alpha_iso\t1.E-6\n")
        file.write("sigmaY\t" + list_parameters[0].key + "\n")
        file.write("Q\t" + list_parameters[1].key + "\n")
        file.write("b\t" + list_parameters[2].key + "\n")
        file.write("C_1\t" + list_parameters[3].key + "\n")
        file.write("D_1\t" + list_parameters[4].key + "\n")
        file.write("C_2\t" + list_parameters[5].key + "\n")
        file.write("D_2\t" + list_parameters[6].key + "\n")
        file.write("F_dfa\t" + str(dfa_params[0]) + "\n")
        file.write("G_dfa\t" + str(dfa_params[1]) + "\n")
        file.write("H_dfa\t" + str(dfa_params[2]) + "\n")
        file.write("L_dfa\t" + str(dfa_params[3]) + "\n")
        file.write("M_dfa\t" + str(dfa_params[4]) + "\n")
        file.write("N_dfa\t" + str(dfa_params[5]) + "\n")
        file.write("K_dfa\t" + str(dfa_params[6]) + "\n")

def _write_dfa_epchg_material_input_file(dfa_params: npt.NDArray[np.float_], elastic_params: npt.NDArray[np.float_],
                                         n_iso_hard: int, n_kin_hard: int,
                                         list_parameters: List[Parameter], path: str = "keys/") -> None:
    with open(path + "material.dat", "w+") as file:
        file.write("Material\nName\tEPCHG\nNumber_of_material_parameters\t" + str(8+2*n_iso_hard+2*n_kin_hard+7) +
                   "\nNumber_of_internal_variables\t" + str(9 + 12*n_kin_hard) + "\n\n#Orientation\npsi\t0\ntheta\t0\nphi\t0\n\n#Mechanical\n")
        file.write("E\t" + str(elastic_params[0]) + "\n")
        file.write("nu\t0.3\n")
        file.write("G\t" + str(elastic_params[1]) + "\n")
        file.write("alpha_iso\t1.E-6\n")
        file.write("sigmaY\t" + list_parameters[0].key + "\n\n")
        file.write("N_iso_hard\t" + str(n_iso_hard) + "\n")
        file.write("N_kin_hard\t" + str(n_kin_hard) + "\n")
        file.write("criteria\t2\n\n")
        for i in range(n_iso_hard):
            file.write("Q\t" + list_parameters[1+2*i].key + "\n")
            file.write("b\t" + list_parameters[2+2*i].key + "\n")
        file.write("\n")
        for i in range(n_kin_hard):
            file.write("C_" + str(i+1) + "\t" + list_parameters[1+n_iso_hard*2+i*2].key + "\n")
            file.write("D_" + str(i+1) + "\t" + list_parameters[1+n_iso_hard*2+i*2+1].key + "\n")
        file.write("\n")
        file.write("F_dfa\t" + str(dfa_params[0]) + "\n")
        file.write("G_dfa\t" + str(dfa_params[1]) + "\n")
        file.write("H_dfa\t" + str(dfa_params[2]) + "\n")
        file.write("L_dfa\t" + str(dfa_params[3]) + "\n")
        file.write("M_dfa\t" + str(dfa_params[4]) + "\n")
        file.write("N_dfa\t" + str(dfa_params[5]) + "\n")
        file.write("K_dfa\t" + str(dfa_params[6]) + "\n")

def _write_epchg_material_input_file(elastic_params: npt.NDArray[np.float_], criteria: str,
                                     criteria_params: Optional[npt.NDArray[np.float_]],
                                     n_iso_hard: int, n_kin_hard: int,
                                     list_parameters: List[Parameter], path: str = "keys/") -> None:

    match criteria:
        case "mises":
            if criteria_params:
                raise ValueError("criteria_params must not be specified or must be an empty list for Mises criterion")
            criteria_id = 0
            criteria_param_names = []
        case "hill":
            if not len(criteria_params) == 6:
                raise ValueError("criteria_params must contain 6 parameters for Hill criterion")
            criteria_id = 1
            criteria_param_names = ["F_hill", "G_hill", "H_hill", "L_hill", "M_hill", "N_hill"]
        case "dfa":
            if not len(criteria_params) == 7:
                raise ValueError("criteria_params must contain 7 parameters for DFA criterion")
            criteria_id = 2
            criteria_param_names = ["F_dfa", "G_dfa", "H_dfa", "L_dfa", "M_dfa", "N_dfa", "K_dfa"]
        case "anisotropic":
            if not len(criteria_params) == 9:
                raise ValueError("criteria_params must contain 9 parameters for anisotropic criterion")
            criteria_id = 3
            criteria_param_names = ["P11", "P22", "P33", "P12", "P13", "P23", "P44", "P55", "P66"]
        case _:
            raise ValueError("Unknown plasticity criterion. Available options are mises, hill, dfa and anisotropic")

    with open(path + "material.dat", "w+") as file:
        file.write("Material\nName\tEPCHG\nNumber_of_material_parameters\t"
                   + str(8+2*n_iso_hard+2*n_kin_hard+len(criteria_param_names))
                   + "\nNumber_of_internal_variables\t"
                   + str(9 + 12*n_kin_hard)
                   + "\n\n#Orientation\npsi\t0\ntheta\t0\nphi\t0\n\n#Mechanical\n")
        file.write("E\t" + str(elastic_params[0]) + "\n")
        file.write("nu\t"+ str(elastic_params[1]) + "\n")
        file.write("G\t" + str(elastic_params[2]) + "\n")
        file.write("alpha_iso\t"+ str(elastic_params[3]) + "\n")
        file.write("sigmaY\t" + list_parameters[0].key + "\n\n")
        file.write("N_iso_hard\t" + str(n_iso_hard) + "\n")
        file.write("N_kin_hard\t" + str(n_kin_hard) + "\n")
        file.write("criteria\t" + str(criteria_id) + "\n\n")
        for i in range(n_iso_hard):
            file.write("Q\t" + list_parameters[1+2*i].key + "\n")
            file.write("b\t" + list_parameters[2+2*i].key + "\n")
        file.write("\n")
        for i in range(n_kin_hard):
            file.write("C_" + str(i+1) + "\t" + list_parameters[1+n_iso_hard*2+i*2].key + "\n")
            file.write("D_" + str(i+1) + "\t" + list_parameters[1+n_iso_hard*2+i*2+1].key + "\n")
        file.write("\n")
        if len(criteria_param_names) != 0:
            for i in range(len(criteria_param_names)):
                file.write(criteria_param_names[i] + "\t" + str(criteria_params[i]) + "\n")

def _write_path_id_file(filename: str, sim_type: str, path: str = "data/") -> None:
    intro = "#Initial_temperature\n290\n#Number_of_blocks\n1\n\n#Block\n1\n#Loading_type\n1\n#Control_type(NLGEOM)\n1\n#Repeat\n1\n#Steps\n3\n\n"
    step_mode = "#Mode\n1\n#Dn_init 1.\n#Dn_mini 1.\n#Dn_inc 0.01\n#time\n1\n"
    step_th_load = "#Consigne_T\nT 290\n\n"

    if sim_type == "tension":
        step_load = "#Consigne\nE 0.05\nS 0 S 0\nS 0 S 0 S 0\n"
        step_unload = "#Consigne\nE 0\nS 0 S 0\nS 0 S 0 S 0\n"
    elif sim_type == "bitension":
        step_load = "#Consigne\nE 0.05\nS 0 E 0.05\nS 0 S 0 S 0\n"
        step_unload = "#Consigne\nE 0\nS 0 E 0\nS 0 S 0 S 0\n"
    elif sim_type == "shear":
        step_load = "#Consigne\nS 0\nE 0.1 S 0\nS 0 S 0 S 0\n"
        step_unload = "#Consigne\nS 0\nE 0 S 0\nS 0 S 0 S 0\n"
    else:
        raise ValueError("sim_type not implemented. Choose between tension, bitension or shear.")
    with open(path + filename, "w+") as file:
        file.write(intro)
        file.write(step_mode)
        file.write(step_load)
        file.write(step_th_load)
        file.write(step_mode)
        file.write(step_unload)
        file.write(step_th_load)
        file.write(step_mode)
        file.write(step_load)
        file.write(step_th_load)

def _create_ident_folders(basedir: str) -> None:
    os.makedirs(basedir + "data/", exist_ok=True)
    os.makedirs(basedir + "exp_data/", exist_ok=True)
    os.makedirs(basedir + "num_data/", exist_ok=True)
    os.makedirs(basedir + "keys/", exist_ok=True)
    os.makedirs(basedir + "results_id/", exist_ok=True)

def _copy_all_default_files(path: str = "data/") -> None:
    shutil.copy(DEFAULT_CONSTANTS_FILE, path)
    shutil.copy(DEFAULT_WEIGHTS_FILE, path)
    shutil.copy(DEFAULT_GEN_FILE, path)
    shutil.copy(DEFAULT_IDENT_CONTROL_FILE, path)
    shutil.copy(DEFAULT_IDENT_ESSENTIALS_FILE, path)
    shutil.copy(DEFAULT_SOLVER_CONTROL_FILE, path)
    shutil.copy(DEFAULT_SOLVER_ESSENTIALS_FILE, path)
    shutil.copy(DEFAULT_MATERIAL_FILE, path)


def compute_epdfa_loss(parameters_to_optimize: List[Parameter], elastic_params: npt.NDArray[np.float_],
                 dfa_params: npt.NDArray[np.float_], path_dir: str = "data/", num_dir: str = "num_data/",
                 results_dir: str = "results_id/") -> float:

    list_path_file = glob.glob("path_id_*.txt", root_dir=path_dir)
    list_output_file = []

    umat_name = 'EPDFA'  # This is the 5 character code for the elastic-plastic subroutine
    nstatev = 33  # The number of scalar variables required, only the initial temperature is stored here

    nu = 0.3
    alpha = 1.E-6
    young_modulus, shear_modulus = elastic_params
    psi_rve = 0.
    theta_rve = 0.
    phi_rve = 0.

    props_elast = np.array([young_modulus, nu, shear_modulus, alpha])
    props_temp = np.append(props_elast, np.array([parameters_to_optimize[i] for i in range(len(parameters_to_optimize))]))
    props = np.append(props_temp, dfa_params)

    for i in range(len(list_path_file)):
        output_file = f"results_EPDFA{i:02}.txt"
        copied_output_file = f"results_EPDFA{i:02}_global-0.txt"
        list_output_file.append(copied_output_file)
        sim.solver(umat_name, props, nstatev, psi_rve, theta_rve, phi_rve, 0, 2, path_dir, results_dir, list_path_file[i],
                   output_file)
        shutil.copy(results_dir + copied_output_file, num_dir)

    c = sim.calc_cost(len(list_output_file), list_output_file)
    print(c)

    return c

def _add_zero_to_equalize_array_length(array_x, array_y):
    x = np.copy(array_x)
    y = np.copy(array_y)
    if len(x) > len(y):
        np.insert(y, 0, [0.0])
    elif len(x) < len(y):
        np.insert(x, 0, [0.0])
    else:
        return x, y
    return x, y

def plot_graph(sim_list: List[str], ident_data_columns_to_plot: List[Tuple[int]],
               exp_data_columns_to_plot: List[Tuple[int]],
               path_results_id: str = "results_id/", path_exp: str = "exp_data/",
               graph_filename: str = "Figure_results_dfa_chaboche.png") -> None:

    if len(sim_list) != len(ident_data_columns_to_plot) or len(sim_list) != len(exp_data_columns_to_plot) or len(ident_data_columns_to_plot) != len(exp_data_columns_to_plot):
        raise IndexError("sim_list, ident_data_columns and exp_data_columns must be of same length")

    list_ident_data_file = glob.glob("results_*_global-0.txt", root_dir=path_results_id)
    list_exp_data_file = glob.glob("input_data_*.txt", root_dir=path_exp)

    if len(list_ident_data_file) != len(sim_list) or len(list_exp_data_file) != len(sim_list):
        raise Exception("Number of files found does not correspond to sim_list length")

    linestyle_list = ["-", "--", ":", "-."]

    plt.figure()
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel(r'Strain', size=15)
    plt.ylabel(r'Stress (MPa)', size=15)

    for i in range(len(sim_list)):
        exp_strain, exp_stress = np.loadtxt(path_exp + list_exp_data_file[i], usecols=exp_data_columns_to_plot[i], unpack=True, skiprows=1)
        ident_strain, ident_stress = np.loadtxt(path_results_id + list_ident_data_file[i], usecols=ident_data_columns_to_plot[i], unpack=True)
        plt.plot(exp_strain, exp_stress, c="black", ls=linestyle_list[i%len(linestyle_list)], label=sim_list[i] + " simulation")
        plt.plot(ident_strain, ident_stress, c="red", ls=linestyle_list[i%len(linestyle_list)],
                 label=sim_list[i] + " identification")

    plt.legend()
    plt.savefig(path_results_id + graph_filename, bbox_inches='tight', format='png')
    plt.close()

def plot_error(sim_list: List[str], ident_data_columns_to_plot: List[Tuple[int]],
               exp_data_columns_to_plot: List[Tuple[int]],
               path_results_id: str = "results_id/", path_exp: str = "exp_data/",
               graph_filename: str = "Figure_error_dfa_chaboche.png") -> None:

    if len(sim_list) != len(ident_data_columns_to_plot) or len(sim_list) != len(exp_data_columns_to_plot) or len(ident_data_columns_to_plot) != len(exp_data_columns_to_plot):
        raise IndexError("sim_list, ident_data_columns and exp_data_columns must be of same length")

    list_ident_data_file = glob.glob("results_*_global-0.txt", root_dir=path_results_id)
    list_exp_data_file = glob.glob("input_data_*.txt", root_dir=path_exp)

    if len(list_ident_data_file) != len(sim_list) or len(list_exp_data_file) != len(sim_list):
        raise Exception("Number of files found does not correspond to sim_list length")

    linestyle_list = ["-", "--", ":", "-."]

    plt.figure()
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel(r'Time', size=15)
    plt.ylabel(r'MSE', size=15)

    for i in range(len(sim_list)):
        exp_strain, exp_stress = np.loadtxt(path_exp + list_exp_data_file[i], usecols=exp_data_columns_to_plot[i], unpack=True, skiprows=1)
        ident_strain, ident_stress = np.loadtxt(path_results_id + list_ident_data_file[i], usecols=ident_data_columns_to_plot[i], unpack=True)

        exp_strain, ident_stress = _add_zero_to_equalize_array_length(exp_strain, ident_stress)
        mse = np.zeros(len(ident_stress))
        iterations = np.array([i for i in range(len(mse))])
        for j in range(len(mse)):
            mse[j] = mean_squared_error([exp_stress[j]], [ident_stress[j]])
        mse_r2_score = round(r2_score(exp_stress.flatten(), ident_stress.flatten()), 3)
        plt.plot(iterations, mse, ls=linestyle_list[i%len(linestyle_list)], label=sim_list[i] + ' MSE (r2 score: ' + str(mse_r2_score) + ')')

    plt.legend()
    plt.savefig(path_results_id + graph_filename, bbox_inches='tight', format='png')
    plt.close()

def plot_nrmse(sim_list: List[str], ident_data_columns_to_plot: List[Tuple[int]],
               exp_data_columns_to_plot: List[Tuple[int]],
               path_results_id: str = "results_id/", path_exp: str = "exp_data/",
               graph_filename: str = "Figure_nrmse_dfa_chaboche.png") -> None:

    if len(sim_list) != len(ident_data_columns_to_plot) or len(sim_list) != len(exp_data_columns_to_plot) or len(ident_data_columns_to_plot) != len(exp_data_columns_to_plot):
        raise IndexError("sim_list, ident_data_columns and exp_data_columns must be of same length")

    list_ident_data_file = glob.glob("results_*_global-0.txt", root_dir=path_results_id)
    list_exp_data_file = glob.glob("input_data_*.txt", root_dir=path_exp)

    if len(list_ident_data_file) != len(sim_list) or len(list_exp_data_file) != len(sim_list):
        raise Exception("Number of files found does not correspond to sim_list length")

    linestyle_list = ["-", "--", ":", "-."]

    plt.figure()
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel(r'Time', size=15)
    plt.ylabel(r'NRMSE', size=15)

    for i in range(len(sim_list)):
        exp_strain, exp_stress = np.loadtxt(path_exp + list_exp_data_file[i], usecols=exp_data_columns_to_plot[i], unpack=True, skiprows=1)
        ident_strain, ident_stress = np.loadtxt(path_results_id + list_ident_data_file[i], usecols=ident_data_columns_to_plot[i], unpack=True)

        exp_strain, ident_stress = _add_zero_to_equalize_array_length(exp_strain, ident_stress)
        nrmse = np.zeros(len(ident_stress))
        ground_truth_mean = np.mean(exp_stress)
        iterations = np.array([i for i in range(len(nrmse))])
        for j in range(len(nrmse)):
            nrmse[j] = root_mean_squared_error([exp_stress[j]], [ident_stress[j]])/ground_truth_mean
        reg_score = round(r2_score(exp_stress.flatten(), ident_stress.flatten()), 3)
        plt.plot(iterations, nrmse, ls=linestyle_list[i%len(linestyle_list)], label=sim_list[i] + ' NRMSE (r2 score: ' + str(reg_score) + ')')

    plt.legend()
    plt.savefig(path_results_id + graph_filename, bbox_inches='tight', format='png')
    plt.close()