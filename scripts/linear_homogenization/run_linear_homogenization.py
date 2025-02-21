from pathlib import Path
import numpy as np
import os
import elana
from belem.fem.fea import run_linear_homogenization

basedir = str(Path(__file__).parent) + "/"
meshfile = str(Path(__file__).parent.parent / "data/cuboct20_unit_cell.vtk")

effective_stiffness_tensor = run_linear_homogenization(mesh_filename=meshfile, young_modulus=1.0e3, poisson_ratio=0.3)

np.savetxt(basedir + "effective_stiffness_tensor.txt", effective_stiffness_tensor)

stiffness = elana.AnisotropicStiffnessTensor(effective_stiffness_tensor)

young_modulus_x, young_modulus_y, young_modulus_z = stiffness.young_xyz()
elana.plot_young_2d(stiffness, output_png_name=basedir + 'young2d.png')
elana.plot_young_3d(stiffness, output_png_name=basedir + 'young3d.png')
mean_young = np.mean(stiffness.data_young_3d)
min_young = np.min(stiffness.data_young_3d)
max_young = np.max(stiffness.data_young_3d)
aniso_young = np.max(stiffness.data_young_3d) / np.min(stiffness.data_young_3d)


poisson_xy, poisson_yz, poisson_xz = stiffness.poisson_xyz()
elana.plot_poisson_2d(stiffness, output_png_name=basedir + "poisson2d.png")
elana.plot_poisson_3d(stiffness, output_png_name=basedir + "poisson3d.png")

shear_xy, shear_yz, shear_xz = stiffness.shear_xyz()
elana.plot_shear_modulus_2d(stiffness, output_png_name=basedir + "shear2d.png")
elana.plot_shear_modulus_3d(stiffness, output_png_name=basedir + "shear3d.png")

elana.plot_linear_compressibility_2d(stiffness, output_png_name=basedir + "linear_compressibility_2d.png")
elana.plot_linear_compressibility_3d(stiffness, output_png_name=basedir + "linear_compressibility_3d.png")

#averages for bulk, shear, young moduli and poisson ratio
voigt_averages = stiffness.voigt_averages()
reuss_averages = stiffness.reuss_averages()
hill_averages = stiffness.hill_averages()