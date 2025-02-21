from pathlib import Path
import numpy as np
import fedoo as fd
from belem.fem.fea import Load, run_fea_computation
import os

meshfile = str(Path(__file__).parent.parent / "data/cuboct20_unit_cell.vtk")

material_law = "EPICP"

E = 200e3
nu = 0.3
alpha = 1e-5  # CTE
Re = 800.0
k = 700.0
m = 0.1
props = np.array([E, nu, alpha, Re, k, m])

tensile_load = Load("Dirichlet", [0], ["DispX"], [0.05]) #5% tensile strain along x
biaxial_tension_load = Load("Dirichlet", [0], ["DispX", "DispY"], [0.05, 0.05]) #5% tensile strain along x and y
compression_load = Load("Dirichlet", [0], ["DispX"], [-0.05]) #5% compression strain along x
biaxial_compression_load = Load("Dirichlet", [0], ["DispX", "DispY"], [-0.05, -0.05]) #5% compression strain along x and y
tension_compression_load = Load("Dirichlet", [0], ["DispX", "DispY"], [0.05, -0.05]) #5% tensile strain along x compression along y
shear_load = Load("Dirichlet", [1], ["DispX"], [0.1]) #5% shear strain in plane (x,y) (/!\ 2*gamma!!!)

tensile_zero = Load("Dirichlet", [0], ["DispX"], [0.0])
biaxial_tension_zero = Load("Dirichlet", [0], ["DispX", "DispY"], [0.0, 0.0])
shear_zero = Load("Dirichlet", [1], ["DispX"], [0.0])

tension_cycle_loads = [tensile_load, tensile_zero, tensile_load]
biaxial_tension_cycle_loads = [biaxial_tension_load, biaxial_tension_zero, biaxial_tension_load]
shear_cycle_loads = [shear_load, shear_zero, shear_load]

typesim_to_loads = {"tension": [tensile_load],
                    "biaxial_tension": [biaxial_tension_load],
                    "compression": [compression_load],
                    "biaxial_compression": [biaxial_compression_load],
                    "tencomp": [tension_compression_load],
                    "shear": [shear_load],
                    "tension_cycle": tension_cycle_loads,
                    "biaxial_tension_cycle": biaxial_tension_cycle_loads,
                    "shear_cycle": shear_cycle_loads}

for typesim in typesim_to_loads.keys():
    print("Running " + typesim + " FE computation")
    results_dir = str(Path(__file__).parent / typesim)
    output_file = typesim
    if not (os.path.isdir(results_dir)):
        os.mkdir(results_dir)

    run_fea_computation(mesh_filename=meshfile, material_law=material_law, props=props, results_dir=results_dir,
                        output_file_name=output_file, load_list=typesim_to_loads[typesim])