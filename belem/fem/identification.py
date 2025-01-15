import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import math as m
from simcoon import simmit as sim
from simcoon import parameter, data
from typing import List, Tuple
import os
import shutil
import glob
from sklearn.metrics import mean_squared_error, r2_score

def get_dfa_parameters(filename: str) -> npt.NDArray[np.float_]:
    dfa_params = np.loadtxt(filename)

    return dfa_params

def compute_young_shear_moduli_from_stiffness_tensor(stiffness_tensor: npt.NDArray[np.float_], young_filename: str, shear_filename: str) -> None:
    young, _, shear = sim.L_cubic_props(stiffness_tensor)
    np.savetxt(young_filename, young)
    np.savetxt(shear_filename, shear)

def get_shape_young_modulus(filename: str) -> float:
    young_modulus = np.loadtxt(filename)

    return young_modulus

def get_shape_shear_modulus(filename: str) -> float:
    shear_modulus = np.loadtxt(filename)

    return shear_modulus

def compute_elastic_material_parameters(bulk_material_young_modulus: float,
                                        shape_young_filename, shape_shear_filename) -> npt.NDArray[np.float_]:

    shape_young_modulus = get_shape_young_modulus(shape_young_filename)
    shape_shear_modulus = get_shape_shear_modulus(shape_shear_filename)

    structure_young = bulk_material_young_modulus*shape_young_modulus
    structure_shear = bulk_material_young_modulus*shape_shear_modulus

    return np.array([structure_young, structure_shear])

def write_parameter_input_file(list_parameters: List[parameter.Parameter], path: str = "data/") -> None:
    with open(path + "parameter.inp", "w+") as file:
        file.write("#Number\t#min\t#max\t#key\t#number_of_files\t#files\n")
        for parameter in list_parameters:
            file.write(str(parameter.number) + "\t" + str(parameter.bounds[0]) + "\t" + str(parameter.bounds[1]) + "\t"
                       + parameter.key + "\t" + str(len(parameter.sim_input_files)) + "\t")
            file.write(' '.join(str(val) for val in parameter.sim_input_files))
            file.write("\n")

def write_dfa_material_input_file(dfa_params: npt.NDArray[np.float_], elastic_params: npt.NDArray[np.float_],
                                  list_parameters: List[parameter.Parameter], path: str = "keys/") -> None:
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

def write_dfa_epchg_material_input_file(dfa_params: npt.NDArray[np.float_], elastic_params: npt.NDArray[np.float_],
                                        n_iso_hard: int, n_kin_hard: int,
                                        list_parameters: List[parameter.Parameter], path: str = "keys/") -> None:
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

def write_path_id_file(filename: str, sim_type: str, path: str = "data/") -> None:
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

def create_ident_folders(basedir: str) -> None:
    os.makedirs(basedir + "data/", exist_ok=True)
    os.makedirs(basedir + "exp_data/", exist_ok=True)
    os.makedirs(basedir + "num_data/", exist_ok=True)
    os.makedirs(basedir + "keys/", exist_ok=True)
    os.makedirs(basedir + "results_id/", exist_ok=True)

def generate_constants_file(filename: str = "constants.inp", path: str = "data/") -> None:
    with open(path + filename, "w+") as file:
        file.write("#Number\t#key\t#input_values\t#number_of_files\t#files\n\n")

def generate_weight_file(filename: str = "files_weights.inp", path: str = "data/") -> None:
    template = """Weight_1_per_file
type_0_no_weight_1_weight_per_file
0
#if_1_list

Weight_2_per_columns
type_0_no_weight_1_sum_per_columns_2_weight_per_columns_plus_sum_3_weight_per_columns_no_sum
1
#if_2_or_3_list

Weight_3_or_uncertainty_per_point
type_0_no_weight_1_weight_column_for_each_exp_col
0
#if_1_list_per_file_of_columns






#Below is a reminder only

Weight_1_per_file
type_0_no_weight_1_weight_per_file
1
#if_1_list
2.0

Weight_2_per_columns
type_0_no_weight_1_sum_per_columns_2_weight_per_columns_plus_sum_3_weight_per_columns_no_sum
2
#if_2_or_3_list
4.0	1.0

Weight_3_or_uncertainty_per_point
type_0_no_weight_1_weight_column_for_each_exp_col
1
#if_1_list_per_file_of_columns
9	10


"""
    with open(path + filename, "w+") as file:
        file.write(template)

def generate_generation_file(filename: str = "gen0.inp", path: str = "data/") -> None:
    template = """g	nindividual	cost	p(0)	p(1)	p(2)	p(3)	p(4)	
0	1	0.538474	32000	5144.24	0.0900987	0.211601	3190.88
0	2	0.538474	30196.8	5144.24	0.0900987	0.211601	3190.88
0	3	0.538474	30196.8	5144.24	0.0900987	0.211601	3190.88
0	4	0.538474	30196.8	5144.24	0.0900987	0.211601	3190.88
"""

    with open(path + filename, "w+") as file:
        file.write(template)

def generate_ident_control_file(filename: str = "ident_control.inp", path: str = "data/") -> None:
    template = """Number_of_generations
100
Aleatory_Space_Population_0=mesh_1=meshlimit_2=random_3=defined
2
Space_Population
10
Golden_boys
1
Max_population_per_subgeneration
5
Mutation_probability_pourcentage
5
Perturbation
0.001
Lagrange_parameters
0.001	10
Lambda_LM
0.01
phiEps
1.
"""

    with open(path + filename, "w+") as file:
        file.write(template)

def generate_ident_essentials_file(filename: str = "ident_essentials.inp", path: str = "data/") -> None:
    template = """Number_of_parameters
5
Number_of_consts
2
Number_of_files
3
"""

    with open(path + filename, "w+") as file:
        file.write(template)

def generate_solver_control_file(filename: str = "solver_control.inp", path: str = "data/") -> None:
    template = """div_tnew_dt_solver
0.5

mul_tnew_dt_solver
2

miniter_solver
10

maxiter_solver
100

inforce_solver
1

precision_solver
1.E-6

lambda_solver
10000.
"""
    with open(path + filename, "w+") as file:
        file.write(template)

def generate_solver_essentials_file(filename: str = "solver_essentials.inp", path: str = "data/") -> None:
    template = """Solver_type_0_Newton_tangent_1_RNL
0"""

    with open(path + filename, "w+") as file:
        file.write(template)

def generate_default_material_file(filename: str = "material.dat", path: str = "data/") -> None:
    template = """Material
Name	EPDFA
Number_of_material_parameters	17
Number_of_internal_variables	33

#Orientation
psi	0
theta	0
phi	0

#Mechanical
E	2405.040177
nu	0.3
G	2405.040177
alpha_iso	1.E-6
sigmaY	66.775157
Q	2441.737289
b	481.737801
C_1	145959.140084
D_1	782.541931
C_2	46511.503691
D_2	210.544501
F_dfa	0.36
G_dfa	0.36
H_dfa	0.36
L_dfa	1.56
M_dfa	1.56
N_dfa	1.56
K_dfa	0.2
"""
    with open(path + filename, "w+") as file:
        file.write(template)

def generate_all_default_files(path: str = "data/") -> None:
    generate_constants_file(path=path)
    generate_weight_file(path=path)
    generate_generation_file(path=path)
    generate_ident_control_file(path=path)
    generate_ident_essentials_file(path=path)
    generate_solver_control_file(path=path)
    generate_solver_essentials_file(path=path)
    generate_default_material_file(path=path)


def compute_epdfa_loss(parameters_to_optimize: List[parameter.Parameter], elastic_params: npt.NDArray[np.float_],
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

def compute_loss(parameters_to_optimize: List[parameter.Parameter], elastic_params: npt.NDArray[np.float_], n_iso_hard: int, n_kin_hard: int,
                 dfa_params: npt.NDArray[np.float_], path_dir: str = "data/", num_dir: str = "num_data/",
                 results_dir: str = "results_id/") -> float:

    list_path_file = glob.glob("path_id_*.txt", root_dir=path_dir)
    list_output_file = []

    umat_name = 'EPCHG'  # This is the 5 character code for the elastic-plastic subroutine
    nstatev = 9 + 12*n_kin_hard  # The number of scalar variables required, only the initial temperature is stored here

    nu = 0.3
    alpha = 1.E-6
    young_modulus, shear_modulus = elastic_params
    psi_rve = 0.
    theta_rve = 0.
    phi_rve = 0.
    criteria = 2

    props_elast = np.array([young_modulus, nu, shear_modulus, alpha])
    props_elast_yield = np.append(props_elast, np.array([parameters_to_optimize[0]]))
    props_elast_yield_gen_params = np.append(props_elast_yield, np.array([n_iso_hard, n_kin_hard, criteria]))
    props_temp = np.append(props_elast_yield_gen_params, np.array([parameters_to_optimize[i+1] for i in range(len(parameters_to_optimize)-1)]))
    props = np.append(props_temp, dfa_params)

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


def add_zero_to_equalize_array_length(array_x, array_y):
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

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
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

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel(r'Time', size=15)
    plt.ylabel(r'MSE', size=15)

    for i in range(len(sim_list)):
        exp_strain, exp_stress = np.loadtxt(path_exp + list_exp_data_file[i], usecols=exp_data_columns_to_plot[i], unpack=True, skiprows=1)
        ident_strain, ident_stress = np.loadtxt(path_results_id + list_ident_data_file[i], usecols=ident_data_columns_to_plot[i], unpack=True)

        exp_strain, ident_stress = add_zero_to_equalize_array_length(exp_strain, ident_stress)
        mse = np.zeros(len(ident_stress))
        iterations = np.array([i for i in range(len(mse))])
        for j in range(len(mse)):
            mse[j] = mean_squared_error([exp_stress[j]], [ident_stress[j]])
        mse_r2_score = round(r2_score(exp_stress, ident_stress), 2)
        plt.plot(iterations, mse, ls=linestyle_list[i%len(linestyle_list)], label=sim_list[i] + ' MSE (r2 score: ' + str(mse_r2_score) + ')')

    plt.legend()
    plt.savefig(path_results_id + graph_filename, bbox_inches='tight', format='png')
    plt.close()