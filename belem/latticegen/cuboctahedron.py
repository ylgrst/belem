import numpy as np
import numpy.typing as npt
import math as m
from scipy.spatial.transform import Rotation
from typing import Optional, Union, Sequence
import cadquery as cq
import pyvista as pv
from microgen import (
    Rve,
    Cylinder,
    periodic,
    cutPhases,
    Phase,
)


class Cuboctahedron:
    """
    Class to create a unit cuboctahedron lattice of given cell size and density or strut radius
    """

    def __init__(self,
                 strut_radius: float = 0.05,
                 # density: Optional[float] = None,
                 cell_size: float = 1.0,
                 repeat_cell: Union[int, Sequence[int]] = 1,
                 center: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 ) -> None:
        """
        Class to generate a unit cuboctahedron lattice.
        The lattice will be created in a cube which size can be modified with 'cell_size'.
        The number of repetitions in each direction of the created geometry can be modified with 'repeat_cell'.

        :param strut_radius: radius of the struts
        :param repeat_cell: integer or list of integers to repeat the geometry in each dimension
        :param center: center of the geometry
        :param orientation: orientation of the geometry
        """

        self.strut_radius = strut_radius
        # self.density = density
        self.cell_size = cell_size
        self.repeat_cell = repeat_cell
        self.center = center
        self.orientation = orientation

        self.rve = Rve(dim_x=self.cell_size, dim_y=self.cell_size, dim_z=self.cell_size, center=self.center)

        self.vertices = self._compute_vertices()
        self.strut_centers = self._compute_strut_centers()
        self.strut_directions_cartesian = self._compute_strut_directions()
        self.strut_directions_euler = self._compute_euler_angles()

        # @classmethod

    # def radius_from_density(
    #         cls,
    #         density: float,
    # ) -> float:
    #     """
    #     Returns the strut radius corresponding to the required density
    #
    #     :param density: Required density, 0.5 for 50%
    #
    #     :return: corresponding strut radius value
    #     """
    #
    #     if not isinstance(density, (int, float)):
    #         raise ValueError("density must be a float between 0 and 1")
    #
    #     cuboctahedron = Cuboctahedron(density=density)
    #     cuboctahedron._compute_strut_radius_to_fit_density()
    #
    #     if isinstance(cuboctahedron.offset, float):
    #         return cuboctahedron.strut_radius
    #     raise ValueError("strut radius must be a float")
    #
    # def _compute_strut_radius_to_fit_density(self) -> None:
    #     if self.density is None:
    #         raise ValueError("density must be given to compute strut radius")
    #
    #     temp_cuboctahedron = Cuboctahedron()
    #
    #     cell_volume = temp_cuboctahedron.cell_size * temp_cuboctahedron.cell_size * temp_cuboctahedron.cell_size
    #
    #     polydata_func = getattr(temp_cuboctahedron, "vtk_lattice")

    def _compute_vertices(self) -> npt.NDArray[np.float_]:
        vertices_array = self.center + self.cell_size * np.array([
            [0.5, 0.5, 0.0],
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [-0.5, 0.0, -0.5],
            [0.5, 0.0, -0.5],
            [-0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.0, -0.5, -0.5],
            [0.0, 0.5, -0.5],
            [0.0, -0.5, 0.5]
        ])

        return vertices_array

    def _compute_strut_centers(self) -> npt.NDArray[np.float_]:
        centers_array = np.array([
            (self.vertices[8] + self.vertices[0]),
            (self.vertices[4] + self.vertices[0]),
            (self.vertices[8] + self.vertices[4]),
            (self.vertices[11] + self.vertices[4]),
            (self.vertices[2] + self.vertices[4]),
            (self.vertices[11] + self.vertices[2]),
            (self.vertices[8] + self.vertices[7]),
            (self.vertices[7] + self.vertices[11]),
            (self.vertices[1] + self.vertices[11]),
            (self.vertices[7] + self.vertices[1]),
            (self.vertices[9] + self.vertices[2]),
            (self.vertices[1] + self.vertices[9]),
            (self.vertices[6] + self.vertices[0]),
            (self.vertices[2] + self.vertices[6]),
            (self.vertices[9] + self.vertices[6]),
            (self.vertices[9] + self.vertices[5]),
            (self.vertices[1] + self.vertices[5]),
            (self.vertices[5] + self.vertices[3]),
            (self.vertices[7] + self.vertices[3]),
            (self.vertices[8] + self.vertices[3]),
            (self.vertices[5] + self.vertices[10]),
            (self.vertices[10] + self.vertices[3]),
            (self.vertices[0] + self.vertices[10]),
            (self.vertices[6] + self.vertices[10])
        ]) / 2.0

        return centers_array

    def _compute_strut_directions(self) -> npt.NDArray[np.float_]:
        directions_array = np.array([
            (self.vertices[8] - self.vertices[0]) / np.linalg.norm((self.vertices[8] - self.vertices[0])),
            (self.vertices[4] - self.vertices[0]) / np.linalg.norm((self.vertices[4] - self.vertices[0])),
            (self.vertices[8] - self.vertices[4]) / np.linalg.norm((self.vertices[8] - self.vertices[4])),
            (self.vertices[11] - self.vertices[4]) / np.linalg.norm((self.vertices[11] - self.vertices[4])),
            (self.vertices[2] - self.vertices[4]) / np.linalg.norm((self.vertices[2] - self.vertices[4])),
            (self.vertices[11] - self.vertices[2]) / np.linalg.norm((self.vertices[11] - self.vertices[2])),
            (self.vertices[8] - self.vertices[7]) / np.linalg.norm((self.vertices[8] - self.vertices[7])),
            (self.vertices[7] - self.vertices[11]) / np.linalg.norm((self.vertices[7] - self.vertices[11])),
            (self.vertices[1] - self.vertices[11]) / np.linalg.norm((self.vertices[1] - self.vertices[11])),
            (self.vertices[7] - self.vertices[1]) / np.linalg.norm((self.vertices[7] - self.vertices[1])),
            (self.vertices[9] - self.vertices[2]) / np.linalg.norm((self.vertices[9] - self.vertices[2])),
            (self.vertices[1] - self.vertices[9]) / np.linalg.norm((self.vertices[1] - self.vertices[9])),
            (self.vertices[6] - self.vertices[0]) / np.linalg.norm((self.vertices[6] - self.vertices[0])),
            (self.vertices[2] - self.vertices[6]) / np.linalg.norm((self.vertices[2] - self.vertices[6])),
            (self.vertices[9] - self.vertices[6]) / np.linalg.norm((self.vertices[9] - self.vertices[6])),
            (self.vertices[9] - self.vertices[5]) / np.linalg.norm((self.vertices[9] - self.vertices[5])),
            (self.vertices[1] - self.vertices[5]) / np.linalg.norm((self.vertices[1] - self.vertices[5])),
            (self.vertices[5] - self.vertices[3]) / np.linalg.norm((self.vertices[5] - self.vertices[3])),
            (self.vertices[7] - self.vertices[3]) / np.linalg.norm((self.vertices[7] - self.vertices[3])),
            (self.vertices[8] - self.vertices[3]) / np.linalg.norm((self.vertices[8] - self.vertices[3])),
            (self.vertices[5] - self.vertices[10]) / np.linalg.norm((self.vertices[5] - self.vertices[10])),
            (self.vertices[10] - self.vertices[3]) / np.linalg.norm((self.vertices[10] - self.vertices[3])),
            (self.vertices[0] - self.vertices[10]) / np.linalg.norm((self.vertices[0] - self.vertices[10])),
            (self.vertices[6] - self.vertices[10]) / np.linalg.norm((self.vertices[6] - self.vertices[10]))
        ])

        return directions_array

    def _compute_euler_angles(self) -> npt.NDArray[np.float]:
        """Computes euler angles from default (1.0, 0.0, 0.0) oriented cylinder for all struts in the lattice"""

        default_dir = np.array([1.0, 0.0, 0.0])

        rotation_vector_array = np.zeros((24, 3))
        euler_angles_array = np.zeros((24, 3))

        for i in range(24):
            axis = np.cross(default_dir, self.strut_directions_cartesian[i])
            axis /= np.linalg.norm(axis)
            angle = np.arccos(np.dot(default_dir, self.strut_directions_cartesian[i]))
            rotation_vector_array[i] = angle * axis
            euler_angles_array[i] = Rotation.from_rotvec(rotation_vector_array[i]).as_euler('zxz', degrees=True)

        return euler_angles_array

    def generate(self) -> cq.Compound:
        listPhases = []
        listPeriodicPhases = []

        for i in range(24):
            elem = Cylinder(
                center=tuple(self.strut_centers[i]),
                orientation=(self.strut_directions_euler[i, 2], self.strut_directions_euler[i, 1], self.strut_directions_euler[i, 0]),
                height=self.cell_size * m.sqrt(2.0) / 2.0,
                radius=self.strut_radius,
            )
            listPhases.append(Phase(shape=elem.generate()))

        for phase_elem in listPhases:
            periodicPhase = periodic(phase=phase_elem, rve=self.rve)
            listPeriodicPhases.append(periodicPhase)

        phases_cut = cutPhases(
            phaseList=listPeriodicPhases, reverseOrder=False
        )
        compound = cq.Compound.makeCompound([phase.shape for phase in phases_cut])

        fused_compound = cq.Compound.fuse(compound)

        return fused_compound

    def generateVtk(self) -> pv.PolyData:
        listStruts = []

        for i in range(24):
            elem = Cylinder(
                center=tuple(self.strut_centers[i]),
                orientation=(self.strut_directions_euler[i, 2], self.strut_directions_euler[i, 1],
                             self.strut_directions_euler[i, 0]),
                height=self.cell_size * m.sqrt(2.0) / 2.0,
                radius=self.strut_radius,
            )
            listStruts.append(elem.generateVtk().triangulate())

        merged_struts = pv.Polydata()
        for strut in listStruts:
            merged_struts.boolean_union(strut)

        return merged_struts





