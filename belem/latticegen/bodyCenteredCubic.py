from .abstractLattice import AbstractLattice
import numpy as np
import numpy.typing as npt
import math as m


class BodyCenteredCubic(AbstractLattice):
    """
    Class to create a unit body-centered cubic lattice of given cell size and density or strut radius
    """

    def __init__(self,
                 *args, **kwargs
                 ) -> None:

        super().__init__(*args, **kwargs)

    @property
    def n_struts(self) -> int:
        return 8

    @property
    def strut_height(self) -> float:
        return self.cell_size * m.sqrt(3.0) / 2.0

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