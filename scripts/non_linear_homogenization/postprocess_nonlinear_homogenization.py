from pathlib import Path
import numpy as np
import os
from belem.fem.postprocess import *
from simcoon import simmit as sim

basedir = str(Path(__file__).parent) + "/"

print("Postprocessing non-linear homogenization FEA computations")
processed_nonlinear_homogenization_results = postprocess_all_homogenization_computations(basedir)

print("Plotting response and hardening curves")
plot_all_stress_strain_from_all_results(processed_nonlinear_homogenization_results, basedir + "all_stress_strain.png")
plot_all_vm_stress_vm_strain_from_all_results(processed_nonlinear_homogenization_results,
                                              basedir + "all_vm_stress_vm_strain.png")
plot_all_hardening_from_all_results(processed_nonlinear_homogenization_results, basedir + "all_hardening.png")


print("Identifying yield surface parameters")
dfa_params = identify_plasticity_criterion_parameters(criterion="dfa",
                                                      all_results_dict=processed_nonlinear_homogenization_results)
np.savetxt(basedir + "dfa_params.txt", dfa_params)

print("Plotting yield surface")
plot_criteria_yield_surface(criterion="dfa",
                            all_results_dict=processed_nonlinear_homogenization_results,
                            criteria_params=dfa_params,
                            figname=basedir + "dfa_yield_surface.png")
plot_criteria_shear_yield_surface(criterion="dfa",
                                  all_results_dict=processed_nonlinear_homogenization_results,
                                  criteria_params=dfa_params,
                                  figname=basedir + "dfa_shear_yield_surface.png")
