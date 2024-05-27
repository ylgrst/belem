from microgen import Tpms, Rve, Phase, meshPeriodic
from microgen.shape.surface_functions import gyroid
import pyvista as pv
import cadquery as cq

density = 0.45
cell_size = 4.0
center = (0.0, 0.0, 0.0)
rve = Rve(dim=cell_size, center=center)

initial_shape = Tpms(surface_function=gyroid, density=density, cell_size=cell_size, resolution=55).generate(type_part="sheet")
cq.exporters.export(initial_shape, 'initial_tpms.step')
phase = [Phase(initial_shape)]
meshPeriodic(mesh_file='initial_tpms.step', rve=rve, listPhases=phase, size=0.06, order=1, output_file='initial_tpms.vtk')
mesh = pv.read('initial_tpms.vtk')
mesh.plot(color='white', show_edges=True)