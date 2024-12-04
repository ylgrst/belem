import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import math as m
from simcoon import simmit as sim
from simcoon import parameter, data
import os


def get_dfa_parameters(filename: str) -> npt.NDArray[np.float_]:
    dfa_params = np.loadtxt(filename)

    return dfa_params

def compute_young_shear_moduli_from_stiffness_tensor(stiffness_tensor: npt.NDArray[np.float_], young_filename: str, shear_filename: str) -> None:
    young, _, shear = sim.L_cubic_props(stiffness_tensor)
    np.savetxt(young_filename, young)
    np.savetxt(shear_filename)

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

    structure_young = bulk_material_young_modulus*shape_young_modulus/1000.0
    structure_shear = bulk_material_young_modulus*shape_shear_modulus/1000.0

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
        file.write("Material\nName\tEPDFA\nNumber_of_material_parameters\t17\nNumber_of_internal_variables\t33\n#Orientation\npsi\t0\ntheta\t0\nphi\t0\n#Mechanical\n")
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