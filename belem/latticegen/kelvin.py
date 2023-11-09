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


class Kelvin:
    """
    Class to create a unit kelvin lattice of given cell size and density or strut radius
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
        Class to generate a unit kelvin lattice.
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
            [1.0, (1.0 + m.sqrt(2)), (1.0 + 2.0*m.sqrt(2))],
            [1.0, -(1.0 + m.sqrt(2)), (1.0 + 2.0 * m.sqrt(2))],
            [1.0, (1.0 + m.sqrt(2)), -(1.0 + 2.0 * m.sqrt(2))],
            [1.0, -(1.0 + m.sqrt(2)), -(1.0 + 2.0 * m.sqrt(2))],
            [1.0, (1.0 + 2.0 * m.sqrt(2)), (1.0 + m.sqrt(2))],
            [1.0, -(1.0 + 2.0 * m.sqrt(2)), (1.0 + m.sqrt(2))],
            [1.0, (1.0 + 2.0 * m.sqrt(2)), -(1.0 + m.sqrt(2))],
            [1.0, -(1.0 + 2.0 * m.sqrt(2)), -(1.0 + m.sqrt(2))],
            [-1.0, (1.0 + m.sqrt(2)), (1.0 + 2.0 * m.sqrt(2))],
            [-1.0, -(1.0 + m.sqrt(2)), (1.0 + 2.0 * m.sqrt(2))],
            [-1.0, (1.0 + m.sqrt(2)), -(1.0 + 2.0 * m.sqrt(2))],
            [-1.0, -(1.0 + m.sqrt(2)), -(1.0 + 2.0 * m.sqrt(2))],
            [-1.0, (1.0 + 2.0 * m.sqrt(2)), (1.0 + m.sqrt(2))],
            [-1.0, -(1.0 + 2.0 * m.sqrt(2)), (1.0 + m.sqrt(2))],
            [-1.0, (1.0 + 2.0 * m.sqrt(2)), -(1.0 + m.sqrt(2))],
            [-1.0, -(1.0 + 2.0 * m.sqrt(2)), -(1.0 + m.sqrt(2))],
            [(1.0 + 2.0 * m.sqrt(2)), 1.0, (1.0 + m.sqrt(2))],
            [(1.0 + 2.0 * m.sqrt(2)), -1.0, (1.0 + m.sqrt(2))],
            [(1.0 + 2.0 * m.sqrt(2)), 1.0, -(1.0 + m.sqrt(2))],
            [(1.0 + 2.0 * m.sqrt(2)), -1.0, -(1.0 + m.sqrt(2))],
            [(1.0 + 2.0 * m.sqrt(2)), (1.0 + m.sqrt(2)), 1.0],
            [(1.0 + 2.0 * m.sqrt(2)), -(1.0 + m.sqrt(2)), 1.0],
            [(1.0 + 2.0 * m.sqrt(2)), (1.0 + m.sqrt(2)), -1.0],
            [(1.0 + 2.0 * m.sqrt(2)), -(1.0 + m.sqrt(2)), -1.0],
            [-(1.0 + 2.0 * m.sqrt(2)), 1.0, (1.0 + m.sqrt(2))],
            [-(1.0 + 2.0 * m.sqrt(2)), -1.0, (1.0 + m.sqrt(2))],
            [-(1.0 + 2.0 * m.sqrt(2)), 1.0, -(1.0 + m.sqrt(2))],
            [-(1.0 + 2.0 * m.sqrt(2)), -1.0, -(1.0 + m.sqrt(2))],
            [-(1.0 + 2.0 * m.sqrt(2)), (1.0 + m.sqrt(2)), 1.0],
            [-(1.0 + 2.0 * m.sqrt(2)), -(1.0 + m.sqrt(2)), 1.0],
            [-(1.0 + 2.0 * m.sqrt(2)), (1.0 + m.sqrt(2)), -1.0],
            [-(1.0 + 2.0 * m.sqrt(2)), -(1.0 + m.sqrt(2)), -1.0],
            [(1.0 + m.sqrt(2)), 1.0, (1.0 + 2.0 * m.sqrt(2))],
            [(1.0 + m.sqrt(2)), -1.0, (1.0 + 2.0 * m.sqrt(2))],
            [(1.0 + m.sqrt(2)), 1.0, -(1.0 + 2.0 * m.sqrt(2))],
            [(1.0 + m.sqrt(2)), -1.0, -(1.0 + 2.0 * m.sqrt(2))],
            [(1.0 + m.sqrt(2)), (1.0 + 2.0 * m.sqrt(2)), 1.0],
            [(1.0 + m.sqrt(2)), -(1.0 + 2.0 * m.sqrt(2)), 1.0],
            [(1.0 + m.sqrt(2)), (1.0 + 2.0 * m.sqrt(2)), -1.0],
            [(1.0 + m.sqrt(2)), -(1.0 + 2.0 * m.sqrt(2)), -1.0],
            [-(1.0 + m.sqrt(2)), 1.0, (1.0 + 2.0 * m.sqrt(2))],
            [-(1.0 + m.sqrt(2)), -1.0, (1.0 + 2.0 * m.sqrt(2))],
            [-(1.0 + m.sqrt(2)), 1.0, -(1.0 + 2.0 * m.sqrt(2))],
            [-(1.0 + m.sqrt(2)), -1.0, -(1.0 + 2.0 * m.sqrt(2))],
            [-(1.0 + m.sqrt(2)), (1.0 + 2.0 * m.sqrt(2)), 1.0],
            [-(1.0 + m.sqrt(2)), -(1.0 + 2.0 * m.sqrt(2)), 1.0],
            [-(1.0 + m.sqrt(2)), (1.0 + 2.0 * m.sqrt(2)), -1.0],
            [-(1.0 + m.sqrt(2)), -(1.0 + 2.0 * m.sqrt(2)), -1.0],
        ]) / (2.0 + 4.0*m.sqrt(2))

        return vertices_array

    def _compute_strut_centers(self) -> npt.NDArray[np.float_]:
        centers_array = np.array([
            (self.vertices[2] + self.vertices[6]),
            (self.vertices[6] + self.vertices[38]),
            (self.vertices[38] + self.vertices[22]),
            (self.vertices[22] + self.vertices[18]),
            (self.vertices[18] + self.vertices[34]),
            (self.vertices[34] + self.vertices[2]),
            (self.vertices[6] + self.vertices[14]),
            (self.vertices[14] + self.vertices[10]),
            (self.vertices[10] + self.vertices[2]),
            (self.vertices[38] + self.vertices[36]),
            (self.vertices[20] + self.vertices[36]),
            (self.vertices[22] + self.vertices[20]),
            (self.vertices[34] + self.vertices[35]),
            (self.vertices[35] + self.vertices[19]),
            (self.vertices[18] + self.vertices[19]),
            (self.vertices[10] + self.vertices[42]),
            (self.vertices[42] + self.vertices[43]),
            (self.vertices[43] + self.vertices[11]),
            (self.vertices[11] + self.vertices[3]),
            (self.vertices[3] + self.vertices[35]),
            (self.vertices[3] + self.vertices[7]),
            (self.vertices[7] + self.vertices[39]),
            (self.vertices[39] + self.vertices[23]),
            (self.vertices[23] + self.vertices[19]),
            (self.vertices[23] + self.vertices[21]),
            (self.vertices[21] + self.vertices[37]),
            (self.vertices[39] + self.vertices[37]),
            (self.vertices[21] + self.vertices[17]),
            (self.vertices[17] + self.vertices[16]),
            (self.vertices[16] + self.vertices[20]),
            (self.vertices[16] + self.vertices[32]),
            (self.vertices[17] + self.vertices[33]),
            (self.vertices[33] + self.vertices[32]),
            (self.vertices[32] + self.vertices[0]),
            (self.vertices[0] + self.vertices[4]),
            (self.vertices[36] + self.vertices[4]),
            (self.vertices[37] + self.vertices[5]),
            (self.vertices[5] + self.vertices[1]),
            (self.vertices[1] + self.vertices[33]),
            (self.vertices[7] + self.vertices[15]),
            (self.vertices[11] + self.vertices[15]),
            (self.vertices[15] + self.vertices[47]),
            (self.vertices[47] + self.vertices[45]),
            (self.vertices[45] + self.vertices[13]),
            (self.vertices[13] + self.vertices[5]),
            (self.vertices[47] + self.vertices[31]),
            (self.vertices[43] + self.vertices[27]),
            (self.vertices[27] + self.vertices[31]),
            (self.vertices[42] + self.vertices[26]),
            (self.vertices[27] + self.vertices[26]),
            (self.vertices[26] + self.vertices[30]),
            (self.vertices[46] + self.vertices[30]),
            (self.vertices[14] + self.vertices[46]),
            (self.vertices[46] + self.vertices[44]),
            (self.vertices[44] + self.vertices[12]),
            (self.vertices[12] + self.vertices[4]),
            (self.vertices[12] + self.vertices[8]),
            (self.vertices[8] + self.vertices[0]),
            (self.vertices[8] + self.vertices[40]),
            (self.vertices[41] + self.vertices[40]),
            (self.vertices[41] + self.vertices[9]),
            (self.vertices[9] + self.vertices[1]),
            (self.vertices[9] + self.vertices[13]),
            (self.vertices[44] + self.vertices[28]),
            (self.vertices[28] + self.vertices[30]),
            (self.vertices[28] + self.vertices[24]),
            (self.vertices[24] + self.vertices[40]),
            (self.vertices[24] + self.vertices[25]),
            (self.vertices[25] + self.vertices[41]),
            (self.vertices[25] + self.vertices[29]),
            (self.vertices[29] + self.vertices[31]),
            (self.vertices[29] + self.vertices[45])
        ]) / 2.0

        return centers_array

    def _compute_strut_directions(self) -> npt.NDArray[np.float_]:
        directions_array = np.array([
            (self.vertices[2] - self.vertices[6]) / np.linalg.norm((self.vertices[2] - self.vertices[6])),
            (self.vertices[6] - self.vertices[38]) / np.linalg.norm((self.vertices[6] - self.vertices[38])),
            (self.vertices[38] - self.vertices[22]) / np.linalg.norm((self.vertices[38] - self.vertices[22])),
            (self.vertices[22] - self.vertices[18]) / np.linalg.norm((self.vertices[22] - self.vertices[18])),
            (self.vertices[18] - self.vertices[34]) / np.linalg.norm((self.vertices[18] - self.vertices[34])),
            (self.vertices[34] - self.vertices[2]) / np.linalg.norm((self.vertices[34] - self.vertices[2])),
            (self.vertices[6] - self.vertices[14]) / np.linalg.norm((self.vertices[6] - self.vertices[14])),
            (self.vertices[14] - self.vertices[10]) / np.linalg.norm((self.vertices[14] - self.vertices[10])),
            (self.vertices[10] - self.vertices[2]) / np.linalg.norm((self.vertices[10] - self.vertices[2])),
            (self.vertices[38] - self.vertices[36]) / np.linalg.norm((self.vertices[38] - self.vertices[36])),
            (self.vertices[20] - self.vertices[36]) / np.linalg.norm((self.vertices[20] - self.vertices[36])),
            (self.vertices[22] - self.vertices[20]) / np.linalg.norm((self.vertices[22] - self.vertices[20])),
            (self.vertices[34] - self.vertices[35]) / np.linalg.norm((self.vertices[34] - self.vertices[35])),
            (self.vertices[35] - self.vertices[19]) / np.linalg.norm((self.vertices[35] - self.vertices[19])),
            (self.vertices[18] - self.vertices[19]) / np.linalg.norm((self.vertices[18] - self.vertices[19])),
            (self.vertices[10] - self.vertices[42]) / np.linalg.norm((self.vertices[10] - self.vertices[42])),
            (self.vertices[42] - self.vertices[43]) / np.linalg.norm((self.vertices[42] - self.vertices[43])),
            (self.vertices[43] - self.vertices[11]) / np.linalg.norm((self.vertices[43] - self.vertices[11])),
            (self.vertices[11] - self.vertices[3]) / np.linalg.norm((self.vertices[11] - self.vertices[3])),
            (self.vertices[3] - self.vertices[35]) / np.linalg.norm((self.vertices[3] - self.vertices[35])),
            (self.vertices[3] - self.vertices[7]) / np.linalg.norm((self.vertices[3] - self.vertices[7])),
            (self.vertices[7] - self.vertices[39]) / np.linalg.norm((self.vertices[7] - self.vertices[39])),
            (self.vertices[39] - self.vertices[23]) / np.linalg.norm((self.vertices[39] - self.vertices[23])),
            (self.vertices[23] - self.vertices[19]) / np.linalg.norm((self.vertices[23] - self.vertices[19])),
            (self.vertices[23] - self.vertices[21]) / np.linalg.norm((self.vertices[23] - self.vertices[21])),
            (self.vertices[21] - self.vertices[37]) / np.linalg.norm((self.vertices[21] - self.vertices[37])),
            (self.vertices[39] - self.vertices[37]) / np.linalg.norm((self.vertices[39] - self.vertices[37])),
            (self.vertices[21] - self.vertices[17]) / np.linalg.norm((self.vertices[21] - self.vertices[17])),
            (self.vertices[17] - self.vertices[16]) / np.linalg.norm((self.vertices[17] - self.vertices[16])),
            (self.vertices[16] - self.vertices[20]) / np.linalg.norm((self.vertices[16] - self.vertices[20])),
            (self.vertices[16] - self.vertices[32]) / np.linalg.norm((self.vertices[16] - self.vertices[32])),
            (self.vertices[17] - self.vertices[33]) / np.linalg.norm((self.vertices[17] - self.vertices[33])),
            (self.vertices[33] - self.vertices[32]) / np.linalg.norm((self.vertices[33] - self.vertices[32])),
            (self.vertices[32] - self.vertices[0]) / np.linalg.norm((self.vertices[32] - self.vertices[0])),
            (self.vertices[0] - self.vertices[4]) / np.linalg.norm((self.vertices[0] - self.vertices[4])),
            (self.vertices[36] - self.vertices[4]) / np.linalg.norm((self.vertices[36] - self.vertices[4])),
            (self.vertices[37] - self.vertices[5]) / np.linalg.norm((self.vertices[37] - self.vertices[5])),
            (self.vertices[5] - self.vertices[1]) / np.linalg.norm((self.vertices[5] - self.vertices[1])),
            (self.vertices[1] - self.vertices[33]) / np.linalg.norm((self.vertices[1] - self.vertices[33])),
            (self.vertices[7] - self.vertices[15]) / np.linalg.norm((self.vertices[7] - self.vertices[15])),
            (self.vertices[11] - self.vertices[15]) / np.linalg.norm((self.vertices[11] - self.vertices[15])),
            (self.vertices[15] - self.vertices[47]) / np.linalg.norm((self.vertices[15] - self.vertices[47])),
            (self.vertices[47] - self.vertices[45]) / np.linalg.norm((self.vertices[47] - self.vertices[45])),
            (self.vertices[45] - self.vertices[13]) / np.linalg.norm((self.vertices[45] - self.vertices[13])),
            (self.vertices[13] - self.vertices[5]) / np.linalg.norm((self.vertices[13] - self.vertices[5])),
            (self.vertices[47] - self.vertices[31]) / np.linalg.norm((self.vertices[47] - self.vertices[31])),
            (self.vertices[43] - self.vertices[27]) / np.linalg.norm((self.vertices[43] - self.vertices[27])),
            (self.vertices[27] - self.vertices[31]) / np.linalg.norm((self.vertices[27] - self.vertices[31])),
            (self.vertices[42] - self.vertices[26]) / np.linalg.norm((self.vertices[42] - self.vertices[26])),
            (self.vertices[27] - self.vertices[26]) / np.linalg.norm((self.vertices[27] - self.vertices[26])),
            (self.vertices[26] - self.vertices[30]) / np.linalg.norm((self.vertices[26] - self.vertices[30])),
            (self.vertices[46] - self.vertices[30]) / np.linalg.norm((self.vertices[46] - self.vertices[30])),
            (self.vertices[14] - self.vertices[46]) / np.linalg.norm((self.vertices[14] - self.vertices[46])),
            (self.vertices[46] - self.vertices[44]) / np.linalg.norm((self.vertices[46] - self.vertices[44])),
            (self.vertices[44] - self.vertices[12]) / np.linalg.norm((self.vertices[44] - self.vertices[12])),
            (self.vertices[12] - self.vertices[4]) / np.linalg.norm((self.vertices[12] - self.vertices[4])),
            (self.vertices[12] - self.vertices[8]) / np.linalg.norm((self.vertices[12] - self.vertices[8])),
            (self.vertices[8] - self.vertices[0]) / np.linalg.norm((self.vertices[8] - self.vertices[0])),
            (self.vertices[8] - self.vertices[40]) / np.linalg.norm((self.vertices[8] - self.vertices[40])),
            (self.vertices[41] - self.vertices[40]) / np.linalg.norm((self.vertices[41] - self.vertices[40])),
            (self.vertices[41] - self.vertices[9]) / np.linalg.norm((self.vertices[41] - self.vertices[9])),
            (self.vertices[9] - self.vertices[1]) / np.linalg.norm((self.vertices[9] - self.vertices[1])),
            (self.vertices[9] - self.vertices[13]) / np.linalg.norm((self.vertices[9] - self.vertices[13])),
            (self.vertices[44] - self.vertices[28]) / np.linalg.norm((self.vertices[44] - self.vertices[28])),
            (self.vertices[28] - self.vertices[30]) / np.linalg.norm((self.vertices[28] - self.vertices[30])),
            (self.vertices[28] - self.vertices[24]) / np.linalg.norm((self.vertices[28] - self.vertices[24])),
            (self.vertices[24] - self.vertices[40]) / np.linalg.norm((self.vertices[24] - self.vertices[40])),
            (self.vertices[24] - self.vertices[25]) / np.linalg.norm((self.vertices[24] - self.vertices[25])),
            (self.vertices[25] - self.vertices[41]) / np.linalg.norm((self.vertices[25] - self.vertices[41])),
            (self.vertices[25] - self.vertices[29]) / np.linalg.norm((self.vertices[25] - self.vertices[29])),
            (self.vertices[29] - self.vertices[31]) / np.linalg.norm((self.vertices[29] - self.vertices[31])),
            (self.vertices[29] - self.vertices[45]) / np.linalg.norm((self.vertices[29] - self.vertices[45]))
        ])

        return directions_array

    def _compute_euler_angles(self) -> npt.NDArray[np.float_]:
        """Computes euler angles from default (1.0, 0.0, 0.0) oriented cylinder for all struts in the lattice"""

        default_dir = np.array([1.0, 0.0, 0.0])

        rotation_vector_array = np.zeros((72, 3))
        euler_angles_array = np.zeros((72, 3))

        for i in range(72):
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

        for i in range(72):
            elem = Cylinder(
                center=tuple(self.strut_centers[i]),
                orientation=(self.strut_directions_euler[i, 2], self.strut_directions_euler[i, 1],
                             self.strut_directions_euler[i, 0]),
                height=self.cell_size * (1 / (1.0 + 2.0*m.sqrt(2))),
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