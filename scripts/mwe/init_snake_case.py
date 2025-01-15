from microgen import Tpms, Rve
from microgen.shape.surface_functions import gyroid
from microgen.operations import rotate_pv_euler
import numpy as np

density = 0.2
cell_size = 1.0
center = (0.0, 0.0, 0.0)
psi, theta, phi = np.pi/4.0, np.pi/4.0, np.pi/4.0
rve = Rve(dim=cell_size, center=center)

shape = Tpms(surface_function=gyroid, density=density, cell_size=cell_size, resolution=30).sheet
rotated_shape = rotate_pv_euler(obj=shape, center=center, angles=(psi, theta, phi))