import os
import fedoo as fd
from fedoo.core.boundary_conditions import ListBC, BoundaryCondition
import numpy as np
import numpy.typing as npt
import pyvista as pv
from typing import List, NamedTuple, Optional, Sequence, Union

# Fedoo >= 1.0 replaces the pair of virtual "constraint driver" nodes by named
# global dofs holding the macroscopic strain. The legacy (node id, variable)
# encoding is kept in the public API and translated here:
#   node 0 -> (E_xx, E_yy, E_zz), node 1 -> (E_xy, E_xz, E_yz)
# The Voigt shear convention (engineering shear angles, i.e. 2*eps_ij) is the
# same as the one the constraint driver dofs used.
MEAN_STRAIN_VECTOR = "MeanStrain"
_CONSTRAINT_DRIVER_TO_MEAN_STRAIN = {
    (0, "DispX"): "E_xx",
    (0, "DispY"): "E_yy",
    (0, "DispZ"): "E_zz",
    (1, "DispX"): "E_xy",
    (1, "DispY"): "E_xz",
    (1, "DispZ"): "E_yz",
}


class Load(NamedTuple):
    """Class to manage load cases for fea computation
    :param boundary_condition_type: type of boundary condition (Dirichlet or Neumann)
    :param constraint_drivers_node_id: constraint driver node id, 0 or 1, given either
    as an int or as a one element sequence
    :param constraint_drivers_variables: list of constraint drivers variables to which apply values
    :param constraint_drivers_values: list of values to be applied to constraint drivers
    """
    boundary_condition_type: str
    constraint_drivers_node_id: Union[int, Sequence[int]]
    constraint_drivers_variables: List[str]
    constraint_drivers_values: List[float]


def run_fea_computation(mesh_filename: str,
                        material_law: str,
                        props: npt.NDArray[np.float64],
                        results_dir: str,
                        output_file_name: str,
                        load_list: List[Load],
                        output_file_ext: str = "fdz",
                        ) -> None:
    """
    Runs fea computation using Fedoo and Simcoon
    :param mesh_filename: name of mesh file (.mesh or .vtk extension)
    :param material_law: material law in 5 character code string (all upper-case) format. See Simcoon's constitutive law
    library for reference
    :param props: array of material properties. See Simcoon's constitutive law library for reference
    :param results_dir: directory in which the output file is to be written
    :param output_file_name: name of the output file (without extension)
    :param load_list: list of loads to be applied
    :param output_file_ext: output file extension, fdz by default (Fedoo's default output format)
    """

    _reset_memory()

    fd.ModelingSpace("3D")

    mesh = fd.Mesh.read(mesh_filename)
    bounds = mesh.bounding_box
    print(bounds, flush=True)

    material = fd.constitutivelaw.Simcoon(material_law, props)
    weakform = fd.weakform.StressEquilibrium(material, nlgeom=False)
    assembly = fd.Assembly.create(weakform, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_solver("direct")
    pb.set_nr_criterion("Displacement", err0=1, tol=1e-4, max_subiter=10)
    pb.add_output(
        results_dir + "/" + output_file_name,
        assembly,
        ["Disp", "Stress", "Strain", "Fext", "Statev", MEAN_STRAIN_VECTOR],
        file_format=output_file_ext,
        compressed=True
    )
    periodic_bc = fd.constraint.PeriodicBC("small_strain", dim=3, meshperio=True)
    pb.bc.add(periodic_bc)
    # Periodicity alone leaves the rigid body translation free, which makes the
    # system singular. Fedoo < 1.0 removed it along with the constraint driver
    # nodes, so block it explicitly here, as fedoo.homogen does: the nearest node
    # to the RVE centre is pinned, which leaves strains and stresses unchanged.
    center_node = [np.linalg.norm(mesh.nodes - bounds.center, axis=1).argmin()]
    pb.bc.add("Dirichlet", center_node, "Disp", 0)
    pb.nlsolve(dt=1.0, tmax=1, update_dt=True, print_info=1, interval_output=1.0)

    for load in load_list:
        load_boundary_conditions = _create_load_case(load)
        pb.bc.add(load_boundary_conditions)
        pb.nlsolve(dt=0.1, tmax=1, update_dt=True, print_info=1, interval_output=0.01)

    _reset_memory()

def run_linear_homogenization(mesh_filename: str,
                              young_modulus: float = 1.0e3,
                              poisson_ratio: float = 0.3) -> npt.NDArray[np.float64]:

    _reset_memory()

    fd.ModelingSpace("3D")
    mesh = fd.Mesh.read(mesh_filename)
    material = fd.constitutivelaw.ElasticIsotrop(young_modulus, poisson_ratio)
    weakform = fd.weakform.StressEquilibrium(material, nlgeom=False)
    assembly = fd.Assembly.create(weakform, mesh, mesh.elm_type, name="Assembly")

    effective_stiffness_tensor = fd.homogen.get_homogenized_stiffness(assembly)

    _reset_memory()

    return effective_stiffness_tensor


def _reset_memory() -> None:
    if "_perturbation" in fd.Problem.get_all():
        del fd.Problem.get_all()["_perturbation"]
    fd.Assembly.delete_memory()


def _create_load_case(load: Load) -> ListBC:
    # the node id is historically given as a one element list, e.g. [0]
    node_ids = np.asarray(load.constraint_drivers_node_id, dtype=int).reshape(-1)
    if node_ids.size != 1:
        raise ValueError(
            f"A load drives a single constraint driver node, got {node_ids.size} ids."
        )
    node_id = int(node_ids[0])
    mean_strain_variables = [_mean_strain_variable(node_id, variable)
                             for variable in load.constraint_drivers_variables]

    # Global dofs carry a single dof each, addressed with the dof index 0.
    load_case = BoundaryCondition.create(load.boundary_condition_type, [0],
                                         mean_strain_variables, load.constraint_drivers_values)

    return load_case


def _mean_strain_variable(node_id: int, variable: str) -> str:
    try:
        return _CONSTRAINT_DRIVER_TO_MEAN_STRAIN[(node_id, variable)]
    except KeyError:
        raise ValueError(
            f"Unknown constraint driver (node {node_id}, variable '{variable}'). "
            "Node id must be 0 or 1 and variable one of DispX, DispY, DispZ."
        ) from None
