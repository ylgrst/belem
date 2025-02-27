import numpy as np
from simcoon import simmit as sim
from simcoon.parameter import Parameter
from simcoon.data import Data
from belem.fem.identification import *
from belem.utils import ResultsColumnHeader, InputColumnHeader
import os
from pathlib import Path

basedir = str(Path(__file__).parent) + "/identification/"
os.makedirs(basedir, exist_ok=True)

bulk_material_young = 200.0
stiffness_tensor = np.loadtxt(str(Path(os.path.realpath(__file__)).parent.parent) + "/data/cuboct20_effective_stiffness_tensor.txt")
shape_young, shape_poisson, shape_shear = sim.L_cubic_props(stiffness_tensor)
thermal_expansion_coefficient = 1e-6
elastic_params = np.array([bulk_material_young*shape_young[0], shape_poisson[0],
                           bulk_material_young*shape_shear[0], thermal_expansion_coefficient])

criteria_params = np.loadtxt(str(Path(__file__).parent) + "/dfa_params.txt")

sigma_y = Parameter(0, (1.0, 400.0), "@0p", ["material.dat"])
Q = Parameter(1, (1.0, 500.0), "@1p", ["material.dat"])
b = Parameter(2, (1.0, 1000.0), "@2p", ["material.dat"])
C_1 = Parameter(3, (100.0, 300000.0), "@3p", ["material.dat"])
D_1 = Parameter(4, (5.0, 1000.0), "@4p", ["material.dat"])
C_2 = Parameter(5, (50.0, 100000.0), "@5p", ["material.dat"])
D_2 = Parameter(6, (10.0, 1000.0), "@6p", ["material.dat"])
list_parameters_to_identify = [sigma_y, Q, b, C_1, D_1, C_2, D_2]

data_tension = Data(control=np.loadtxt(str(Path(__file__).parent) + "/tension_cycle/strain.txt") / 100.0,
                    observation=np.loadtxt(str(Path(__file__).parent) + "/tension_cycle/stress_component.txt"))
data_biaxial_tension = Data(control=np.loadtxt(str(Path(__file__).parent) + "/biaxial_tension_cycle/strain.txt") / 100.0,
                            observation=np.loadtxt(str(Path(__file__).parent) + "/biaxial_tension_cycle/stress_component.txt"))
data_shear = Data(control=np.loadtxt(str(Path(__file__).parent) + "/shear_cycle/strain.txt") / 100.0,
                  observation=np.loadtxt(str(Path(__file__).parent) + "/shear_cycle/stress_component.txt"))
list_data = [data_tension, data_biaxial_tension, data_shear]

prepare_epchg_identification(data_to_identify=list_data,
                             list_columns_to_compare=[[ResultsColumnHeader.S11],
                                                      [ResultsColumnHeader.S11],
                                                      [ResultsColumnHeader.S12]],
                             parameters_to_optimize=list_parameters_to_identify,
                             elastic_params=elastic_params,
                             n_iso_hard=1, n_kin_hard=2, criteria="dfa", criteria_params=criteria_params,
                             basedir=basedir)

os.chdir(basedir)

homogenized_law_params = run_epchg_identification(parameters_to_optimize=list_parameters_to_identify,
                                                  elastic_params=elastic_params, n_iso_hard=1, n_kin_hard=2,
                                                  criteria="dfa", criteria_params=criteria_params,
                                                  path_dir=basedir + "/data/", num_dir=basedir + "/num_data/",
                                                  results_dir=basedir + "/results_id/", popsize=10, tol=0.0001,
                                                  maxiter=100, disp=True)
print(homogenized_law_params)

plot_graph(sim_list=["tension", "biaxial tension", "shear"],
           ident_data_columns_to_plot=[[ResultsColumnHeader.E11, ResultsColumnHeader.S11],
                                       [ResultsColumnHeader.E11, ResultsColumnHeader.S11],
                                       [ResultsColumnHeader.E12, ResultsColumnHeader.S12]],
           exp_data_columns_to_plot=[[InputColumnHeader.STRAIN, InputColumnHeader.STRESS],
                                     [InputColumnHeader.STRAIN, InputColumnHeader.STRESS],
                                     [InputColumnHeader.STRAIN, InputColumnHeader.STRESS]],
           path_results_id=basedir + "/results_id/",
           path_exp=basedir + "/exp_data/",
           )

plot_nrmse(sim_list=["tension", "biaxial tension", "shear"],
           ident_data_columns_to_plot=[[ResultsColumnHeader.E11, ResultsColumnHeader.S11],
                                       [ResultsColumnHeader.E11, ResultsColumnHeader.S11],
                                       [ResultsColumnHeader.E12, ResultsColumnHeader.S12]],
           exp_data_columns_to_plot=[[InputColumnHeader.STRAIN, InputColumnHeader.STRESS],
                                     [InputColumnHeader.STRAIN, InputColumnHeader.STRESS],
                                     [InputColumnHeader.STRAIN, InputColumnHeader.STRESS]],
           path_results_id=basedir + "/results_id/",
           path_exp=basedir + "/exp_data/",
           )