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
    Box,
    periodic,
    cutPhases,
    Phase,
)


class BodyCenteredCubic:
    """
    Class to create a unit body-centered cubic lattice of given cell size and density or strut radius
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
        Class to generate a unit body-centered cubic lattice.
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

    def _compute_vertices(self) -> npt.NDArray[np.float_]:
        vertices_array = self.center + self.cell_size * np.array([
            [0.0, 0.0, 0.0],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5]
        ])

        return vertices_array

    def _compute_strut_centers(self) -> npt.NDArray[np.float_]:
        centers_array = np.array([
            (self.vertices[1] + self.vertices[0]),
            (self.vertices[2] + self.vertices[0]),
            (self.vertices[3] + self.vertices[0]),
            (self.vertices[4] + self.vertices[0]),
            (self.vertices[5] + self.vertices[0]),
            (self.vertices[6] + self.vertices[0]),
            (self.vertices[7] + self.vertices[0]),
            (self.vertices[8] + self.vertices[0]),
        ]) / 2.0

        return centers_array

    def _compute_strut_directions(self) -> npt.NDArray[np.float_]:
        directions_array = np.array([
            (self.vertices[1] - self.vertices[0]) / np.linalg.norm((self.vertices[1] - self.vertices[0])),
            (self.vertices[2] - self.vertices[0]) / np.linalg.norm((self.vertices[2] - self.vertices[0])),
            (self.vertices[3] - self.vertices[0]) / np.linalg.norm((self.vertices[3] - self.vertices[0])),
            (self.vertices[4] - self.vertices[0]) / np.linalg.norm((self.vertices[4] - self.vertices[0])),
            (self.vertices[5] - self.vertices[0]) / np.linalg.norm((self.vertices[5] - self.vertices[0])),
            (self.vertices[6] - self.vertices[0]) / np.linalg.norm((self.vertices[6] - self.vertices[0])),
            (self.vertices[7] - self.vertices[0]) / np.linalg.norm((self.vertices[7] - self.vertices[0])),
            (self.vertices[8] - self.vertices[0]) / np.linalg.norm((self.vertices[8] - self.vertices[0])),
        ])

        return directions_array

    def _compute_euler_angles(self) -> npt.NDArray[np.float_]:
        """Computes euler angles from default (1.0, 0.0, 0.0) oriented cylinder for all struts in the lattice"""

        default_dir = np.array([1.0, 0.0, 0.0])

        rotation_vector_array = np.zeros((8, 3))
        euler_angles_array = np.zeros((8, 3))

        for i in range(8):
            if np.all(self.strut_directions_cartesian[i] == default_dir) or np.all(self.strut_directions_cartesian[i] == -default_dir):
                euler_angles_array[i] = np.zeros(3)
            else:
                axis = np.cross(default_dir, self.strut_directions_cartesian[i])
                axis /= np.linalg.norm(axis)
                angle = np.arccos(np.dot(default_dir, self.strut_directions_cartesian[i]))
                rotation_vector_array[i] = angle * axis
                euler_angles_array[i] = Rotation.from_rotvec(rotation_vector_array[i]).as_euler('zxz', degrees=True)

        return euler_angles_array

    def generate(self) -> cq.Shape:
        listPhases = []

        for i in range(8):
            elem = Cylinder(
                center=tuple(self.strut_centers[i]),
                orientation=(self.strut_directions_euler[i, 2], self.strut_directions_euler[i, 1],
                             self.strut_directions_euler[i, 0]),
                height=self.cell_size * m.sqrt(3.0) / 2.0,
                radius=self.strut_radius,
            )
            listPhases.append(Phase(shape=elem.generate()))

        compound = cq.Compound.makeCompound([phase.shape for phase in listPhases])

        fused_compound = cq.Compound.fuse(compound)

        bounding_box = Box(center=self.center, orientation=self.orientation, dim_x=self.cell_size, dim_y=self.cell_size, dim_z=self.cell_size).generate()

        lattice = bounding_box.intersect(fused_compound)

        return lattice

    @property
    def volume(self) -> float:
        volume = self.generate().Volume()/(self.cell_size * self.cell_size * self.cell_size)

        return volume