from .abstractLattice import AbstractLattice
import numpy as np
import numpy.typing as npt
import math as m
from typing import Union, Sequence


class OctetTruss(AbstractLattice):
    """
    Class to create a unit octet-truss lattice of given cell size and density or strut radius
    """

    def __init__(self,
                 strut_radius: float = 0.05,
                 cell_size: float = 1.0,
                 repeat_cell: Union[int, Sequence[int]] = 1,
                 center: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 ) -> None:
        super().__init__(strut_radius=strut_radius, cell_size=cell_size, repeat_cell=repeat_cell, center=center,
                         orientation=orientation)

    @property
    def n_struts(self) -> int:
        return 36

    @property
    def strut_height(self) -> float:
        return self.cell_size * m.sqrt(2.0) / 2.0

    def _compute_vertices(self) -> npt.NDArray[np.float_]:
        vertices_array = self.center + self.cell_size * np.array([
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, -0.5],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, -0.5, 0.0]
        ])

        return vertices_array

    def _compute_strut_centers(self) -> npt.NDArray[np.float_]:
        centers_array = np.array([
            (self.vertices[11] + self.vertices[2]),
            (self.vertices[11] + self.vertices[3]),
            (self.vertices[11] + self.vertices[6]),
            (self.vertices[11] + self.vertices[7]),
            (self.vertices[13] + self.vertices[6]),
            (self.vertices[13] + self.vertices[3]),
            (self.vertices[13] + self.vertices[1]),
            (self.vertices[13] + self.vertices[5]),
            (self.vertices[10] + self.vertices[1]),
            (self.vertices[10] + self.vertices[5]),
            (self.vertices[10] + self.vertices[4]),
            (self.vertices[10] + self.vertices[0]),
            (self.vertices[12] + self.vertices[4]),
            (self.vertices[12] + self.vertices[0]),
            (self.vertices[12] + self.vertices[2]),
            (self.vertices[12] + self.vertices[7]),
            (self.vertices[8] + self.vertices[1]),
            (self.vertices[8] + self.vertices[3]),
            (self.vertices[8] + self.vertices[0]),
            (self.vertices[8] + self.vertices[2]),
            (self.vertices[9] + self.vertices[5]),
            (self.vertices[9] + self.vertices[6]),
            (self.vertices[9] + self.vertices[7]),
            (self.vertices[9] + self.vertices[4]),
            (self.vertices[13] + self.vertices[9]),
            (self.vertices[13] + self.vertices[8]),
            (self.vertices[8] + self.vertices[11]),
            (self.vertices[12] + self.vertices[8]),
            (self.vertices[10] + self.vertices[8]),
            (self.vertices[9] + self.vertices[10]),
            (self.vertices[9] + self.vertices[12]),
            (self.vertices[9] + self.vertices[11]),
            (self.vertices[11] + self.vertices[13]),
            (self.vertices[13] + self.vertices[10]),
            (self.vertices[10] + self.vertices[12]),
            (self.vertices[12] + self.vertices[11])
        ]) / 2.0

        return centers_array

    def _compute_strut_directions(self) -> npt.NDArray[np.float_]:
        directions_array = np.array([
            (self.vertices[11] - self.vertices[2]) / np.linalg.norm((self.vertices[11] - self.vertices[2])),
            (self.vertices[11] - self.vertices[3]) / np.linalg.norm((self.vertices[11] - self.vertices[3])),
            (self.vertices[11] - self.vertices[6]) / np.linalg.norm((self.vertices[11] - self.vertices[6])),
            (self.vertices[11] - self.vertices[7]) / np.linalg.norm((self.vertices[11] - self.vertices[7])),
            (self.vertices[13] - self.vertices[6]) / np.linalg.norm((self.vertices[13] - self.vertices[6])),
            (self.vertices[13] - self.vertices[3]) / np.linalg.norm((self.vertices[13] - self.vertices[3])),
            (self.vertices[13] - self.vertices[1]) / np.linalg.norm((self.vertices[13] - self.vertices[1])),
            (self.vertices[13] - self.vertices[5]) / np.linalg.norm((self.vertices[13] - self.vertices[5])),
            (self.vertices[10] - self.vertices[1]) / np.linalg.norm((self.vertices[10] - self.vertices[1])),
            (self.vertices[10] - self.vertices[5]) / np.linalg.norm((self.vertices[10] - self.vertices[5])),
            (self.vertices[10] - self.vertices[4]) / np.linalg.norm((self.vertices[10] - self.vertices[4])),
            (self.vertices[10] - self.vertices[0]) / np.linalg.norm((self.vertices[10] - self.vertices[0])),
            (self.vertices[12] - self.vertices[4]) / np.linalg.norm((self.vertices[12] - self.vertices[4])),
            (self.vertices[12] - self.vertices[0]) / np.linalg.norm((self.vertices[12] - self.vertices[0])),
            (self.vertices[12] - self.vertices[2]) / np.linalg.norm((self.vertices[12] - self.vertices[2])),
            (self.vertices[12] - self.vertices[7]) / np.linalg.norm((self.vertices[12] - self.vertices[7])),
            (self.vertices[8] - self.vertices[1]) / np.linalg.norm((self.vertices[8] - self.vertices[1])),
            (self.vertices[8] - self.vertices[3]) / np.linalg.norm((self.vertices[8] - self.vertices[3])),
            (self.vertices[8] - self.vertices[0]) / np.linalg.norm((self.vertices[8] - self.vertices[0])),
            (self.vertices[8] - self.vertices[2]) / np.linalg.norm((self.vertices[8] - self.vertices[2])),
            (self.vertices[9] - self.vertices[5]) / np.linalg.norm((self.vertices[9] - self.vertices[5])),
            (self.vertices[9] - self.vertices[6]) / np.linalg.norm((self.vertices[9] - self.vertices[6])),
            (self.vertices[9] - self.vertices[7]) / np.linalg.norm((self.vertices[9] - self.vertices[7])),
            (self.vertices[9] - self.vertices[4]) / np.linalg.norm((self.vertices[9] - self.vertices[4])),
            (self.vertices[13] - self.vertices[9]) / np.linalg.norm((self.vertices[13] - self.vertices[9])),
            (self.vertices[13] - self.vertices[8]) / np.linalg.norm((self.vertices[13] - self.vertices[8])),
            (self.vertices[8] - self.vertices[11]) / np.linalg.norm((self.vertices[8] - self.vertices[11])),
            (self.vertices[12] - self.vertices[8]) / np.linalg.norm((self.vertices[12] - self.vertices[8])),
            (self.vertices[10] - self.vertices[8]) / np.linalg.norm((self.vertices[10] - self.vertices[8])),
            (self.vertices[9] - self.vertices[10]) / np.linalg.norm((self.vertices[9] - self.vertices[10])),
            (self.vertices[9] - self.vertices[12]) / np.linalg.norm((self.vertices[9] - self.vertices[12])),
            (self.vertices[9] - self.vertices[11]) / np.linalg.norm((self.vertices[9] - self.vertices[11])),
            (self.vertices[11] - self.vertices[13]) / np.linalg.norm((self.vertices[11] - self.vertices[13])),
            (self.vertices[13] - self.vertices[10]) / np.linalg.norm((self.vertices[13] - self.vertices[10])),
            (self.vertices[10] - self.vertices[12]) / np.linalg.norm((self.vertices[10] - self.vertices[12])),
            (self.vertices[12] - self.vertices[11]) / np.linalg.norm((self.vertices[12] - self.vertices[11]))
        ])

        return directions_array
