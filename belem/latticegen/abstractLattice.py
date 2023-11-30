from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np
from scipy.spatial.transform import Rotation
import cadquery as cq
from microgen import (
    Rve,
    Cylinder,
    Box,
    fuseShapes,
)


class AbstractLattice(ABC):
    """
    Abstract Class to create strut-based lattice
    """

    def __init__(self,
                 strut_radius: float = 0.05,
                 cell_size: float = 1.0,
                 center: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 ) -> None:
        """
        Class to generate a unit body-centered cubic lattice.
        The lattice will be created in a cube which size can be modified with 'cell_size'.
        The number of repetitions in each direction of the created geometry can be modified with 'repeat_cell'.

        :param strut_radius: radius of the struts
        :param cell_size: size of the cubic rve in which the lattice cell is enclosed
        :param center: center of the geometry
        """

        self.strut_radius = strut_radius
        self.cell_size = cell_size
        self.center = center

        self.rve = Rve(dim_x=self.cell_size, dim_y=self.cell_size, dim_z=self.cell_size, center=self.center)

        self.vertices = self._compute_vertices()
        self.strut_centers = self._compute_strut_centers()
        self.strut_directions_cartesian = self._compute_strut_directions()
        self.strut_directions_euler = self._compute_euler_angles()

    @property
    @abstractmethod
    def n_struts(self) -> int: ...

    @property
    @abstractmethod
    def strut_height(self) -> float: ...

    @abstractmethod
    def _compute_vertices(self) -> npt.NDArray[np.float_]: ...

    @abstractmethod
    def _compute_strut_centers(self) -> npt.NDArray[np.float_]: ...

    @abstractmethod
    def _compute_strut_directions(self) -> npt.NDArray[np.float_]: ...

    def _compute_euler_angles(self) -> npt.NDArray[np.float_]:
        """Computes euler angles from default (1.0, 0.0, 0.0) oriented cylinder for all struts in the lattice"""

        default_dir = np.array([1.0, 0.0, 0.0])

        rotation_vector_array = np.zeros((self.n_struts, 3))
        euler_angles_array = np.zeros((self.n_struts, 3))

        for i in range(self.n_struts):
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
        list_struts = []

        for i in range(self.n_struts):
            elem = Cylinder(
                center=tuple(self.strut_centers[i]),
                orientation=(self.strut_directions_euler[i, 2], self.strut_directions_euler[i, 1],
                             self.strut_directions_euler[i, 0]),
                height=self.strut_height,
                radius=self.strut_radius,
            )
            list_struts.append(elem.generate())

        fused_compound = fuseShapes(list_struts, retain_edges=False)

        bounding_box = Box(center=self.center, dim_x=self.cell_size, dim_y=self.cell_size, dim_z=self.cell_size).generate()

        lattice = bounding_box.intersect(fused_compound)

        return lattice

    @property
    def volume(self) -> float:
        volume = self.generate().Volume()

        return volume