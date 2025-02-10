import os
import fedoo as fd
from fedoo.core.boundary_conditions import ListBC, BoundaryCondition
import numpy as np
import numpy.typing as npt
import pyvista as pv
from typing import List, Optional, NamedTuple
np.float_ = np.float64

class Load(NamedTuple):
    """Class to manage load cases for fea computation
    :param boundary_condition_type: type of boundary condition (Dirichlet or Neumann)
    :param constraint_drivers_nodes_ids: list of constraint drivers nodes ids
    :param constraint_drivers_variables: list of constraint drivers variables to which apply values
    :param constraint_drivers_values: list of values to be applied to constraint drivers
    """
    boundary_condition_type: str
    constraint_drivers_node_id: int
    constraint_drivers_variables: List[str]
    constraint_drivers_values: List[float]


def run_fea_computation(mesh_filename: str,
                        material_law: str,
                        props: npt.NDArray[np.float_],
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

    ref_node = mesh.add_virtual_nodes(2)
    mesh.nodes[ref_node[0], :] = bounds.center.tolist()
    mesh.nodes[ref_node[1], :] = bounds.center.tolist()
    node_cd = [ref_node[0] for i in range(3)] + [ref_node[1] for i in range(3)]
    var_cd = ["DispX", "DispY", "DispZ", "DispX", "DispY", "DispZ"]

    material = fd.constitutivelaw.Simcoon(material_law, props)
    weakform = fd.weakform.StressEquilibrium(material, nlgeom=False)
    assembly = fd.Assembly.create(weakform, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_solver("direct")
    pb.set_nr_criterion("Displacement", err0=1, tol=1e-4, max_subiter=10)
    pb.add_output(
        results_dir + "/" + output_file_name,
        assembly,
        ["Disp", "Stress", "Strain", "Fext", "Statev"],
        file_format=output_file_ext,
        compressed=True
    )
    periodic_bc = fd.constraint.PeriodicBC(node_cd, var_cd, dim=3, meshperio=True)
    pb.bc.add(periodic_bc)
    pb.nlsolve(dt=1.0, tmax=1, update_dt=True, print_info=1, interval_output=1.0)

    for load in load_list:
        load_boundary_conditions = _create_load_case(load, ref_node)
        pb.bc.add(load_boundary_conditions)
        pb.nlsolve(dt=0.1, tmax=1, update_dt=True, print_info=1, interval_output=0.01)

    _reset_memory()

def run_linear_homogenization(mesh_filename: str,
                              young_modulus: float = 1.0e3,
                              poisson_ratio: float = 0.3) -> npt.NDArray[np.float_]:

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


def _create_load_case(load: Load, ref_node: npt.NDArray[int]) -> ListBC:
    load_case = BoundaryCondition.create(load.boundary_condition_type, ref_node[load.constraint_drivers_node_id],
                                         load.constraint_drivers_variables, load.constraint_drivers_values)

    return load_case
