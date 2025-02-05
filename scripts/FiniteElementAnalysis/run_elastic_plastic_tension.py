from pathlib import Path
import numpy as np
import fedoo as fd
from belem.fem.fea import Load, run_fea_computation

meshfile = str(Path(__file__).parent / "cuboct20_unit_cell.vtk")
results_dir = str(Path(__file__).parent / "results/")
output_file = "fea_example"


material_law = "EPICP"

E = 200e3
nu = 0.3
alpha = 1e-5  # CTE
Re = 500.0
k = 600.0
m = 0.2
props = np.array([E, nu, alpha, Re, k, m])

tensile_load = Load("Dirichlet", [0], ["DispX"], [0.5]) #5% tensile strain along x
print(type(tensile_load.boundary_condition_type))

run_fea_computation(mesh_filename=meshfile, material_law=material_law, props=props, results_dir=results_dir,
                    output_file_name=output_file, load_list=[tensile_load])

