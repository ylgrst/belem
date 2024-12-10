import numpy as np
import numpy.typing as npt
import fedoo as fd
import matplotlib.pyplot as plt
import math as m
from simcoon import simmit as sim
from scipy.optimize import minimize, differential_evolution, Bounds
import pyvista as pv
from typing import Optional, Union, List
from tqdm import tqdm


def compute_average_stress_strain_arrays(dataset: fd.MultiFrameDataSet, component: str, n_incr: int = 100, max_strain: np.float_ = 0.05, cycle: bool = False) -> dict[
    str, npt.NDArray[np.float_]]:
    """Returns average stress and strain arrays"""
    mesh_volume = dataset.mesh.to_pyvista().volume
    rve_volume = dataset.mesh.bounding_box.volume
    density = mesh_volume / rve_volume
    n_iter = dataset.n_iter
    stress_array = np.zeros(n_iter)
    strain_array = 100 * np.linspace(0, max_strain, n_incr + 1)
    if cycle:
        relax_array = strain_array[::-1]
        relax_array = np.delete(relax_array, 0)
        reload_array = np.delete(strain_array, 0)
        relax_reload_array = np.append(relax_array, reload_array)
        strain_array = np.append(strain_array, relax_reload_array)
    for i in range(n_iter):
        dataset.load(i)
        data_stress = dataset.get_data(field="Stress", component=component, data_type="GaussPoint")
        vol_avg_stress = (density / mesh_volume) * dataset.mesh.integrate_field(field=data_stress,
                                                                                type_field="GaussPoint")
        stress_array[i] = vol_avg_stress

    return {"strain": strain_array, "stress": stress_array}


def compute_von_mises_stress(dataset: fd.MultiFrameDataSet) -> npt.NDArray[np.float_]:
    """Returns von mises stress"""
    mesh_volume = dataset.mesh.to_pyvista().volume
    rve_volume = dataset.mesh.bounding_box.volume
    density = mesh_volume / rve_volume
    n_iter = dataset.n_iter
    vm_stress_array = np.zeros(n_iter)
    for i in range(n_iter):
        dataset.load(i)
        data_vm_stress = dataset.get_data(field="Stress", component="vm", data_type="GaussPoint")
        vol_avg_stress = (density / mesh_volume) * dataset.mesh.integrate_field(field=data_vm_stress,
                                                                                type_field="GaussPoint")
        vm_stress_array[i] = vol_avg_stress

    return vm_stress_array


def compute_von_mises_strain(dataset: fd.MultiFrameDataSet) -> npt.NDArray[np.float_]:
    """Returns von mises strain array"""
    mesh_volume = dataset.mesh.to_pyvista().volume
    n_iter = dataset.n_iter
    strain_array = np.zeros(n_iter)
    for i in range(n_iter):
        dataset.load(i)
        data_Ep = dataset.get_data(field="Strain", data_type="GaussPoint")
        data_vm_Ep = np.asarray([sim.Mises_strain(data_Ep[:, i]) for i in range(np.shape(data_Ep)[1])])
        vol_avg_vm_Ep = dataset.mesh.integrate_field(field=data_vm_Ep, type_field="GaussPoint") / mesh_volume
        strain_array[i] = 100 * vol_avg_vm_Ep

    return strain_array


def compute_von_mises_plastic_strain(dataset: fd.MultiFrameDataSet) -> npt.NDArray[np.float_]:
    """Returns von mises plastic strain array"""
    mesh_volume = dataset.mesh.to_pyvista().volume
    n_iter = dataset.n_iter
    plastic_strain_array = np.zeros(n_iter)
    for i in range(n_iter):
        dataset.load(i)
        data_Ep = dataset.get_data(field="Statev", data_type="GaussPoint")[2:8]
        data_vm_Ep = np.asarray([sim.Mises_strain(data_Ep[:, i]) for i in range(np.shape(data_Ep)[1])])
        vol_avg_vm_Ep = dataset.mesh.integrate_field(field=data_vm_Ep, type_field="GaussPoint") / mesh_volume
        plastic_strain_array[i] = 100 * vol_avg_vm_Ep

    return plastic_strain_array


def diagonalize_stress_tensor(voigt_stress_tensor: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Returns diagonalized stress matrix"""
    if np.shape(voigt_stress_tensor) != (6,):
        raise ValueError("Voigt stress tensor is not a 6x1 vector")
    stress_tensor = np.zeros((3, 3))
    for i in range(3):
        stress_tensor[i, i] = voigt_stress_tensor[i]
        for j in range(3):
            if i < j:
                stress_tensor[i, j] = voigt_stress_tensor[i + j + 2]
                stress_tensor[j, i] = stress_tensor[i, j]

    eigenvalues, _ = np.linalg.eig(stress_tensor)
    diag = np.sort(abs(eigenvalues))[::-1]

    return diag


def compute_principal_stresses(dataset: fd.MultiFrameDataSet) -> npt.NDArray[np.float_]:
    """Returns principal stresses arrays"""
    component_to_voigt: dict[str, int] = {"XX": 0, "YY": 1, "ZZ": 2, "XY": 3, "XZ": 4, "YZ": 5}
    component_list = ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]
    mesh_volume = dataset.mesh.to_pyvista().volume
    rve_volume = dataset.mesh.bounding_box.volume
    density = mesh_volume / rve_volume
    n_iter = dataset.n_iter
    principal_stresses_array = np.zeros((n_iter, 3))
    for i in range(n_iter):
        dataset.load(i)
        stress_array = np.zeros(6)
        for component in component_list:
            data_stress = dataset.get_data(field="Stress", component=component, data_type="GaussPoint")
            vol_avg_stress = (density / mesh_volume) * dataset.mesh.integrate_field(field=data_stress,
                                                                                    type_field="GaussPoint")
            stress_array[component_to_voigt[component]] = vol_avg_stress
        principal_stresses = diagonalize_stress_tensor(stress_array)
        principal_stresses_array[i, :] = principal_stresses

    return principal_stresses_array


def compute_all_arrays_from_data_fields(dataset: fd.MultiFrameDataSet, component: str, n_incr: int = 100, max_strain: np.float_ = 0.05, cycle: bool = False) -> dict[str, npt.NDArray[np.float_]]:
    """Returns average stress and strain arrays, von mises stress array, von mises strain array,
    von mises plastic strain array, principal stresses arrays
    :param component: stress array component
    """
    mesh_volume = dataset.mesh.to_pyvista().volume
    rve_volume = dataset.mesh.bounding_box.volume
    density = mesh_volume / rve_volume
    n_iter = dataset.n_iter
    strain_array = 100 * np.linspace(0, max_strain, int(n_incr + 1))
    if cycle:
        relax_array = strain_array[::-1]
        relax_array = np.delete(relax_array, 0)
        reload_array = np.delete(strain_array, 0)
        relax_reload_array = np.append(relax_array, reload_array)
        strain_array = np.append(strain_array, relax_reload_array)
    stress_array = np.zeros(n_iter)
    stress_component_array_all_iter = np.zeros((n_iter, 6))
    vm_stress_array = np.zeros(n_iter)
    vm_strain_array = np.zeros(n_iter)
    plastic_strain_array = np.zeros(n_iter)
    principal_stresses_array = np.zeros((n_iter, 3))
    output_dict = {}
    for i in tqdm(range(n_iter)):
        dataset.load(i)
        data_stress = dataset.get_data(field="Stress", component=component, data_type="GaussPoint")
        vol_avg_stress = (density / mesh_volume) * dataset.mesh.integrate_field(field=data_stress,
                                                                                type_field="GaussPoint")
        stress_array[i] = vol_avg_stress

        data_vm_stress = dataset.get_data(field="Stress", component="vm", data_type="GaussPoint")
        vol_avg_vm_stress = (density / mesh_volume) * dataset.mesh.integrate_field(field=data_vm_stress,
                                                                                type_field="GaussPoint")
        vm_stress_array[i] = vol_avg_vm_stress

        data_Ep = dataset.get_data(field="Strain", data_type="GaussPoint")
        data_vm_Ep = np.asarray([sim.Mises_strain(data_Ep[:, i]) for i in range(np.shape(data_Ep)[1])])
        vol_avg_vm_Ep = dataset.mesh.integrate_field(field=data_vm_Ep, type_field="GaussPoint") / mesh_volume
        vm_strain_array[i] = 100 * vol_avg_vm_Ep

        data_plastic_Ep = dataset.get_data(field="Statev", data_type="GaussPoint")[2:8]
        data_vm_plastic_Ep = np.asarray([sim.Mises_strain(data_plastic_Ep[:, i]) for i in range(np.shape(data_plastic_Ep)[1])])
        vol_avg_vm_plastic_Ep = dataset.mesh.integrate_field(field=data_vm_plastic_Ep, type_field="GaussPoint") / mesh_volume
        plastic_strain_array[i] = 100 * vol_avg_vm_plastic_Ep

        stress_components_array = np.zeros(6)
        component_list = ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]
        component_to_voigt: dict[str, int] = {"XX": 0, "YY": 1, "ZZ": 2, "XY": 3, "XZ": 4, "YZ": 5}
        for comp in component_list:
            data_stress_components = dataset.get_data(field="Stress", component=comp, data_type="GaussPoint")
            vol_avg_stress_components = (density / mesh_volume) * dataset.mesh.integrate_field(field=data_stress_components,
                                                                                    type_field="GaussPoint")
            stress_components_array[component_to_voigt[comp]] = vol_avg_stress_components
        stress_component_array_all_iter[i] = stress_components_array
        principal_stresses = diagonalize_stress_tensor(stress_components_array)
        principal_stresses_array[i] = principal_stresses

    output_dict["stress_component"] = stress_array
    output_dict["stress_tensor"] = stress_component_array_all_iter
    output_dict["strain"] = strain_array
    output_dict["vm_stress"] = vm_stress_array
    output_dict["vm_strain"] = vm_strain_array
    output_dict["vm_plastic_strain"] = plastic_strain_array
    output_dict["principal_stresses"] = principal_stresses_array
    return output_dict


def separate_tension_cycle_data(tension_cycle_dataset: fd.MultiFrameDataSet, n_incr_per_cycle: int = 100, max_strain: np.float_ = 0.05) -> tuple[dict[str, npt.NDArray[np.float_]]]:
    all_cycles_data = compute_all_arrays_from_data_fields(tension_cycle_dataset, component = "XX", n_incr=n_incr_per_cycle, max_strain=max_strain, cycle=True)
    tcycle_tension_data = {}
    tcycle_compression_data = {}
    t_cycle_reload_data = {}
    for key in all_cycles_data.keys():
        tcycle_tension_data[key] = all_cycles_data[key][:n_incr_per_cycle + 1]
        tcycle_compression_data[key] = all_cycles_data[key][n_incr_per_cycle + 1:2 * n_incr_per_cycle + 1]
        t_cycle_reload_data[key] = all_cycles_data[key][2 * n_incr_per_cycle + 1:]

    return tcycle_tension_data, tcycle_compression_data, t_cycle_reload_data

def save_separated_tension_cycle_data(tcycle_tension_data: dict[str, npt.NDArray[np.float_]], tcycle_compression_data: dict[str, npt.NDArray[np.float_]], tcycle_reload_data: dict[str, npt.NDArray[np.float_]]) -> None:
    for key in tcycle_tension_data.keys():
        np.savetxt("cycle_tension_1_" + key + ".txt", tcycle_tension_data[key])
        np.savetxt("cycle_compression_2_" + key + ".txt", tcycle_compression_data[key])
        np.savetxt("cycle_tension_3_" + key + ".txt", tcycle_reload_data[key])

def plot_separated_tension_cycle(tcycle_tension_data: dict[str, npt.NDArray[np.float_]], tcycle_compression_data: dict[str, npt.NDArray[np.float_]], tcycle_reload_data: dict[str, npt.NDArray[np.float_]]) -> None:
    plot_stress_strain(tcycle_tension_data["stress_component"], tcycle_tension_data["strain"], "cycle_tension_1_stress_strain.png")
    plot_stress_strain(tcycle_tension_data["vm_stress"], tcycle_tension_data["vm_strain"], "cycle_tension_1_vm_stress_vm_strain.png")
    plot_hardening(tcycle_tension_data["vm_stress"], tcycle_tension_data["vm_plastic_strain"], "cycle_tension_1_hardening")

    plot_stress_strain(-tcycle_compression_data["stress_component"], -tcycle_compression_data["strain"], "cycle_compression_2_stress_strain.png")
    plot_stress_strain(tcycle_compression_data["vm_stress"], tcycle_compression_data["vm_strain"], "cycle_compression_2_vm_stress_vm_strain.png")
    plot_hardening(tcycle_compression_data["vm_stress"], tcycle_compression_data["vm_plastic_strain"], "cycle_compression_2_hardening")

    plot_stress_strain(tcycle_reload_data["stress_component"], tcycle_reload_data["strain"], "cycle_reload_3_stress_strain.png")
    plot_stress_strain(tcycle_reload_data["vm_stress"], tcycle_reload_data["vm_strain"], "cycle_reload_3_vm_stress_vm_strain.png")
    plot_hardening(tcycle_reload_data["vm_stress"], tcycle_reload_data["vm_plastic_strain"], "cycle_reload_3_hardening")


def create_plastic_strain_and_principal_stress_data(dataset: fd.MultiFrameDataSet) -> tuple[
    npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    p_stresses = compute_principal_stresses(dataset)
    vm_plastic_strain = compute_von_mises_plastic_strain(dataset)

    return vm_plastic_strain, p_stresses


def get_stress_at_plasticity_threshold(stress_array: npt.NDArray[np.float_],
                                       plasticity_array: npt.NDArray[np.float_], plasticity_threshold: float) -> \
        npt.NDArray[np.float_]:
    """Fetches point in stress-strain curve where set plasticity threshold is reached"""
    try:
        index = next(idx for idx, plasticity in enumerate(plasticity_array) if plasticity > plasticity_threshold)
    except StopIteration:
        index = len(plasticity_array) - 1
    stress = stress_array[:, index]

    return stress

def get_stress_tensor_at_plasticity_threshold(stress_array: npt.NDArray[np.float_], plasticity_array: npt.NDArray[np.float_], plasticity_threshold: float) -> npt.NDArray[np.float_]:
    """Fetches stress array where set plasticity threshold is reached"""
    try:
        index = next(idx for idx, plasticity in enumerate(plasticity_array) if plasticity > plasticity_threshold)
    except StopIteration:
        index = len(plasticity_array) - 1
    stress = stress_array[index]

    return stress

def get_vm_stress_at_plasticity_threshold(vm_stress_array: npt.NDArray[np.float_],
                                          vm_plasticity_array: npt.NDArray[np.float_],
                                          plasticity_threshold: float) -> float:
    try:
        index = next(idx for idx, plasticity in enumerate(vm_plasticity_array) if plasticity > plasticity_threshold)
    except StopIteration:
        index = len(vm_plasticity_array) - 1
    vm_stress = vm_stress_array[index]

    return vm_stress


def compute_average_stress_tensor(dataset: fd.MultiFrameDataSet) -> npt.NDArray[np.float_]:
    component_to_voigt: dict[str, int] = {"XX": 0, "YY": 1, "ZZ": 2, "XY": 3, "XZ": 4, "YZ": 5}
    component_list = ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]
    mesh_volume = dataset.mesh.to_pyvista().volume
    rve_volume = dataset.mesh.bounding_box.volume
    density = mesh_volume / rve_volume
    n_iter = dataset.n_iter
    stress_array = np.zeros((6, n_iter))
    for i in range(n_iter):
        dataset.load(i)
        stress = np.zeros(6)
        for component in component_list:
            data_stress = dataset.get_data(field="Stress", component=component, data_type="GaussPoint")
            vol_avg_stress = (density / mesh_volume) * dataset.mesh.integrate_field(field=data_stress,
                                                                                    type_field="GaussPoint")
            stress[component_to_voigt[component]] = vol_avg_stress
        stress_array[:, i] = stress

    return stress_array


def plot_stress_strain(stress_array: npt.NDArray[np.float_], strain_array: npt.NDArray[np.float_],
                       figname: str = "stress_strain.png"):
    plt.figure()
    plt.plot(strain_array, stress_array, 'o-')
    plt.title("Stress vs strain")
    plt.xlabel("Strain (%)")
    plt.ylabel("Stress (MPa)")
    plt.savefig(figname)
    plt.close()

def plot_all_stress_strain_from_all_results(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]],
                                            figname: str = "all_stress_strain.png") -> None:

    sim_to_plot_label = {"tension": "tension (S11 vs E11)", "biaxial_tension": "biaxial tension (S11 vs E11)",
                          "compression": "compression (S11 vs E11)",
                          "biaxial_compression": "biaxial compression (S11 vs E11)",
                          "tencomp": "tension-compression (S11 vs E11)",
                          "shear": "shear (S12 vs E12)"}

    plt.figure()
    plt.title("Stress vs strain")
    plt.xlabel("Strain (%)")
    plt.ylabel("Stress (MPa)")
    plt.grid(True)
    for key in sim_to_plot_label.keys():
        if key in ["compression", "biaxial_compression"]:
            strain = -all_results_dict[key]["strain"]
        else:
            strain = all_results_dict[key]["strain"]
        stress = all_results_dict[key]["stress_component"]
        plt.plot(strain, stress, ls="-", label=sim_to_plot_label[key])
    plt.legend()
    plt.savefig(figname)
    plt.close()


def plot_hardening(stress_array: npt.NDArray[np.float_], plasticity_array: npt.NDArray[np.float_],
                   figname: str = "hardening.png"):
    plt.figure()
    plt.plot(plasticity_array, stress_array, 'o-')
    plt.title("Stress vs plastic strain")
    plt.xlabel("Plastic strain (%)")
    plt.ylabel("Stress (MPa)")
    plt.savefig(figname)
    plt.close()

def plot_all_hardening_from_all_results(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]],
                                            figname: str = "all_hardening.png") -> None:

    sim_to_plot_label = {"tension": "tension (S11 vs E11)", "biaxial_tension": "biaxial tension (S11 vs E11)",
                          "compression": "compression (S11 vs E11)",
                          "biaxial_compression": "biaxial compression (S11 vs E11)",
                          "tencomp": "tension-compression (S11 vs E11)",
                          "shear": "shear (S12 vs E12)"}

    plt.figure()
    plt.title("Stress vs strain")
    plt.xlabel("Strain (%)")
    plt.ylabel("Stress (MPa)")
    plt.grid(True)
    for key in sim_to_plot_label.keys():
        strain = all_results_dict[key]["vm_plastic_strain"]
        stress = all_results_dict[key]["stress_component"]
        plt.plot(strain, stress, ls="-", label=sim_to_plot_label[key])
    plt.legend()
    plt.savefig(figname)
    plt.close()

def build_yield_data_projected_to_all_directions(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]]) -> list[dict[str, tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]]]:
    """USE ONLY ON STRUCTURES WITH AT LEAST CUBIC SYMMETRY"""
    yield_data = {}
    yield_data_stress = {}
    yield_data_pstrain = {}

    def swap_indices(array: npt.NDArray[np.float_], idx1: int, idx2: int):
        swapped_array = np.copy(array)
        for i in range(len(array)):
            swapped_array[i, idx1], swapped_array[i, idx2] = swapped_array[i, idx2], swapped_array[i, idx1]

        return swapped_array

    yield_data["tension_11"] = (all_results_dict["tension"]["stress_tensor"], all_results_dict["tension"]["vm_plastic_strain"])
    yield_data["tension_22"] = (
    swap_indices(all_results_dict["tension"]["stress_tensor"], 0, 1), all_results_dict["tension"]["vm_plastic_strain"])
    yield_data["tension_33"] = (
    swap_indices(all_results_dict["tension"]["stress_tensor"], 0, 2), all_results_dict["tension"]["vm_plastic_strain"])
    yield_data["compression_11"] = (
    all_results_dict["compression"]["stress_tensor"], all_results_dict["compression"]["vm_plastic_strain"])
    yield_data["compression_22"] = (
    swap_indices(all_results_dict["compression"]["stress_tensor"], 0, 1), all_results_dict["compression"]["vm_plastic_strain"])
    yield_data["compression_33"] = (
    swap_indices(all_results_dict["compression"]["stress_tensor"], 0, 2), all_results_dict["compression"]["vm_plastic_strain"])
    yield_data["bitraction_1122"] = (
    all_results_dict["biaxial_tension"]["stress_tensor"], all_results_dict["biaxial_tension"]["vm_plastic_strain"])
    yield_data["bitraction_1133"] = (swap_indices(all_results_dict["biaxial_tension"]["stress_tensor"], 1, 2),
                                    all_results_dict["biaxial_tension"]["vm_plastic_strain"])
    yield_data["bitraction_2233"] = (swap_indices(all_results_dict["biaxial_tension"]["stress_tensor"], 0, 2),
                                    all_results_dict["biaxial_tension"]["vm_plastic_strain"])
    yield_data["bicompression_1122"] = (
    all_results_dict["biaxial_compression"]["stress_tensor"], all_results_dict["biaxial_compression"]["vm_plastic_strain"])
    yield_data["bicompression_1133"] = (swap_indices(all_results_dict["biaxial_compression"]["stress_tensor"], 1, 2),
                                       all_results_dict["biaxial_compression"]["vm_plastic_strain"])
    yield_data["bicompression_2233"] = (swap_indices(all_results_dict["biaxial_compression"]["stress_tensor"], 0, 2),
                                       all_results_dict["biaxial_compression"]["vm_plastic_strain"])
    yield_data["tencomp_1122"] = (all_results_dict["tencomp"]["stress_tensor"], all_results_dict["tencomp"]["vm_plastic_strain"])
    yield_data["tencomp_1133"] = (
    swap_indices(all_results_dict["tencomp"]["stress_tensor"], 1, 2), all_results_dict["tencomp"]["vm_plastic_strain"])
    yield_data["tencomp_2233"] = (
    swap_indices(all_results_dict["tencomp"]["stress_tensor"], 0, 2), all_results_dict["tencomp"]["vm_plastic_strain"])
    yield_data["compten_1122"] = (-all_results_dict["tencomp"]["stress_tensor"], all_results_dict["tencomp"]["vm_plastic_strain"])
    yield_data["compten_1133"] = (-swap_indices(all_results_dict["tencomp"]["stress_tensor"], 1, 2), all_results_dict["tencomp"]["vm_plastic_strain"])
    yield_data["compten_2233"] = (-swap_indices(all_results_dict["tencomp"]["stress_tensor"], 0, 2), all_results_dict["tencomp"]["vm_plastic_strain"])
    yield_data["shear_12"] = (all_results_dict["shear"]["stress_tensor"], all_results_dict["shear"]["vm_plastic_strain"])
    yield_data["shear_13"] = (
    swap_indices(all_results_dict["shear"]["stress_tensor"], 3, 4), all_results_dict["shear"]["vm_plastic_strain"])
    yield_data["shear_23"] = (
    swap_indices(all_results_dict["shear"]["stress_tensor"], 3, 5), all_results_dict["shear"]["vm_plastic_strain"])
    yield_data["neg_shear_12"] = (-all_results_dict["shear"]["stress_tensor"], all_results_dict["shear"]["vm_plastic_strain"])
    yield_data["neg_shear_13"] = (
    -swap_indices(all_results_dict["shear"]["stress_tensor"], 3, 4), all_results_dict["shear"]["vm_plastic_strain"])
    yield_data["neg_shear_23"] = (
    -swap_indices(all_results_dict["shear"]["stress_tensor"], 3, 5), all_results_dict["shear"]["vm_plastic_strain"])

    for key in yield_data.keys():
        yield_data_stress[key] = yield_data[key][0]
        yield_data_pstrain[key] = yield_data[key][1]

    return yield_data, yield_data_stress, yield_data_pstrain


def compute_yield_surface_data_from_all_results(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]], plasticity_threshold: float) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    stress_at_plastic_strain_threshold_tension = get_stress_tensor_at_plasticity_threshold(all_results_dict["tension"]["principal_stresses"], all_results_dict["tension"]["vm_plastic_strain"], plasticity_threshold)
    stress_at_plastic_strain_threshold_compression = get_stress_tensor_at_plasticity_threshold(
        all_results_dict["compression"]["principal_stresses"], all_results_dict["compression"]["vm_plastic_strain"], plasticity_threshold)
    stress_at_plastic_strain_threshold_biaxial_tension = get_stress_tensor_at_plasticity_threshold(
        all_results_dict["biaxial_tension"]["principal_stresses"], all_results_dict["biaxial_tension"]["vm_plastic_strain"], plasticity_threshold)
    stress_at_plastic_strain_threshold_biaxial_compression = get_stress_tensor_at_plasticity_threshold(
        all_results_dict["biaxial_compression"]["principal_stresses"], all_results_dict["biaxial_compression"]["vm_plastic_strain"], plasticity_threshold)
    stress_at_plastic_strain_threshold_tencomp = get_stress_tensor_at_plasticity_threshold(
        all_results_dict["tencomp"]["principal_stresses"], all_results_dict["tencomp"]["vm_plastic_strain"], plasticity_threshold)

    plot_data_s11 = [stress_at_plastic_strain_threshold_tension[0],
                     stress_at_plastic_strain_threshold_biaxial_tension[0], 0.0,
                     -stress_at_plastic_strain_threshold_tencomp[0], -stress_at_plastic_strain_threshold_compression[0],
                     -stress_at_plastic_strain_threshold_biaxial_compression[0], 0.0,
                     stress_at_plastic_strain_threshold_tencomp[0], stress_at_plastic_strain_threshold_tension[0]]
    plot_data_s22 = [0.0, stress_at_plastic_strain_threshold_biaxial_tension[1],
                     stress_at_plastic_strain_threshold_tension[0], stress_at_plastic_strain_threshold_tencomp[1], 0.0,
                     -stress_at_plastic_strain_threshold_biaxial_compression[1],
                     -stress_at_plastic_strain_threshold_tension[0], -stress_at_plastic_strain_threshold_tencomp[1],
                     0.0]

    return plot_data_s11, plot_data_s22

def compute_yield_shear_surface_data_from_all_results(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]], plasticity_threshold: float) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    stress_at_plastic_strain_threshold_tension = get_stress_tensor_at_plasticity_threshold(
        all_results_dict["tension"]["stress_component"], all_results_dict["tension"]["vm_plastic_strain"],
        plasticity_threshold)
    stress_at_plastic_strain_threshold_compression = get_stress_tensor_at_plasticity_threshold(
        all_results_dict["compression"]["stress_component"], all_results_dict["compression"]["vm_plastic_strain"],
        plasticity_threshold)
    stress_at_plastic_strain_threshold_shear = get_stress_tensor_at_plasticity_threshold(
        all_results_dict["shear"]["stress_component"], all_results_dict["shear"]["vm_plastic_strain"],
        plasticity_threshold)

    plot_data_s11 = [stress_at_plastic_strain_threshold_tension, 0.0, stress_at_plastic_strain_threshold_compression, 0.0]
    plot_data_s12 = [0.0, stress_at_plastic_strain_threshold_shear, 0.0, - stress_at_plastic_strain_threshold_shear]

    return plot_data_s11, plot_data_s12

def compute_yield_surface_data(tension_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                               biaxial_tension_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                               tencomp_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                               compression_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                               biaxial_compression_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                               plasticity_threshold: float) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    stress_at_plastic_strain_threshold_tension = get_stress_at_plasticity_threshold(tension_data[1],
                                                                                    tension_data[0],
                                                                                    plasticity_threshold)
    stress_at_plastic_strain_threshold_compression = get_stress_at_plasticity_threshold(compression_data[1],
                                                                                        compression_data[0],
                                                                                        plasticity_threshold)
    stress_at_plastic_strain_threshold_biaxial_tension = get_stress_at_plasticity_threshold(
        biaxial_tension_data[1], biaxial_tension_data[0], plasticity_threshold)
    stress_at_plastic_strain_threshold_biaxial_compression = get_stress_at_plasticity_threshold(
        biaxial_compression_data[1], biaxial_compression_data[0], plasticity_threshold)
    stress_at_plastic_strain_threshold_tencomp = get_stress_at_plasticity_threshold(tencomp_data[1],
                                                                                    tencomp_data[0],
                                                                                    plasticity_threshold)

    plot_data_s11 = [stress_at_plastic_strain_threshold_tension[0],
                     stress_at_plastic_strain_threshold_biaxial_tension[0], 0.0,
                     -stress_at_plastic_strain_threshold_tencomp[0], -stress_at_plastic_strain_threshold_compression[0],
                     -stress_at_plastic_strain_threshold_biaxial_compression[0], 0.0,
                     stress_at_plastic_strain_threshold_tencomp[0], stress_at_plastic_strain_threshold_tension[0]]
    plot_data_s22 = [0.0, stress_at_plastic_strain_threshold_biaxial_tension[1],
                     stress_at_plastic_strain_threshold_tension[0], stress_at_plastic_strain_threshold_tencomp[1], 0.0,
                     -stress_at_plastic_strain_threshold_biaxial_compression[1],
                     -stress_at_plastic_strain_threshold_tension[0], -stress_at_plastic_strain_threshold_tencomp[1],
                     0.0]

    return plot_data_s11, plot_data_s22


def plot_yield_surface_from_all_results(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]], plasticity_threshold: float, figname: str = "stress_at_Ep.png") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel(r"$\sigma_{11}$ (MPa)")
    plt.ylabel(r"$\sigma_{22}$ (MPa)")
    ax.xaxis.set_label_coords(1.05, 0.45)
    ax.yaxis.set_label_coords(0.55, 1.05)

    plot_data_s11, plot_data_s22 = compute_yield_surface_data_from_all_results(all_results_dict,
                                                              plasticity_threshold)

    plt.plot(plot_data_s11, plot_data_s22, "o--")

    plt.savefig(figname)
    plt.close()

def plot_yield_shear_surface_from_all_results(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]], plasticity_threshold: float, figname: str = "stress_at_Ep_shear.png") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel(r"$\sigma_{12}$ (MPa)")
    plt.ylabel(r"$\sigma_{11}$ (MPa)")
    ax.xaxis.set_label_coords(1.05, 0.45)
    ax.yaxis.set_label_coords(0.55, 1.05)

    plot_data_s11, plot_data_s12 = compute_yield_shear_surface_data_from_all_results(all_results_dict,
                                                              plasticity_threshold)

    plt.plot(plot_data_s11, plot_data_s12, "o--")

    plt.savefig(figname)
    plt.close()

def plot_yield_surface(tension_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                       biaxial_tension_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                       tencomp_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                       compression_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                       biaxial_compression_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                       plasticity_threshold: float, figname: str = "stress_at_Ep.png") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel(r"$\sigma_{11}$ (MPa)")
    plt.ylabel(r"$\sigma_{22}$ (MPa)")
    ax.xaxis.set_label_coords(1.05, 0.45)
    ax.yaxis.set_label_coords(0.55, 1.05)

    plot_data_s11, plot_data_s22 = compute_yield_surface_data(tension_data, biaxial_tension_data, tencomp_data,
                                                              compression_data, biaxial_compression_data,
                                                              plasticity_threshold)

    plt.plot(plot_data_s11, plot_data_s22, "o--")

    plt.savefig(figname)
    plt.close()


def plot_yield_surface_evolution_from_all_results(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]], plasticity_threshold_list: list[float],
                                 figname: str = "yield_surface_evolution.png") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel(r"$\sigma_{11}$ (MPa)")
    plt.ylabel(r"$\sigma_{22}$ (MPa)")
    ax.xaxis.set_label_coords(1.05, 0.45)
    ax.yaxis.set_label_coords(0.55, 1.05)

    for plasticity_threshold in plasticity_threshold_list:
        try:
            plot_data_s11, plot_data_s22 = compute_yield_surface_data_from_all_results(all_results_dict, plasticity_threshold)
            plt.plot(plot_data_s11, plot_data_s22, "o--", label=r"$\epsilon^{p} = $" + str(plasticity_threshold) + "%")
        except:
            print("Plasticity threshold ", plasticity_threshold, " not reached.")

    plt.legend()
    plt.savefig(figname)
    plt.close()


def plot_yield_surface_evolution(tension_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                                 biaxial_tension_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                                 tencomp_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                                 compression_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                                 biaxial_compression_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                                 plasticity_threshold_list: list[float],
                                 figname: str = "yield_surface_evolution.png") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel(r"$\sigma_{11}$ (MPa)")
    plt.ylabel(r"$\sigma_{22}$ (MPa)")
    ax.xaxis.set_label_coords(1.05, 0.45)
    ax.yaxis.set_label_coords(0.55, 1.05)

    for plasticity_threshold in plasticity_threshold_list:
        try:
            plot_data_s11, plot_data_s22 = compute_yield_surface_data(tension_data, biaxial_tension_data, tencomp_data,
                                                                      compression_data, biaxial_compression_data,
                                                                      plasticity_threshold)
            plt.plot(plot_data_s11, plot_data_s22, "o--", label=r"$\epsilon^{p} = $" + str(plasticity_threshold) + "%")
        except:
            print("Plasticity threshold ", plasticity_threshold, " not reached.")

    plt.legend()
    plt.savefig(figname)
    plt.close()


def plot_clipped_vm_plastic_strain(dataset: fd.MultiFrameDataSet, global_plasticity_threshold: float,
                                   local_plasticity_threshold: float, figname: str) -> None:
    """Plots the Von Mises plastic strain in a mesh, clipped by local plasticity threshold (given in %),
    at simulation increment corresponding to set global plasticity threshold (given in %)"""
    global_vm_plasticity_array = compute_von_mises_plastic_strain(dataset)
    index = next(
        idx for idx, plasticity in enumerate(global_vm_plasticity_array) if plasticity > global_plasticity_threshold)
    dataset.load(index)
    canvas_mesh = dataset.mesh.to_pyvista()
    mesh = canvas_mesh
    edges = mesh.extract_feature_edges()
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(canvas_mesh, color="lightblue", opacity=0.1)
    pl.add_mesh(edges, color="black", line_width=3)
    data_Ep = dataset.get_data(field="Statev", data_type="GaussPoint")[2:8]
    data_vm_Ep = np.asarray([sim.Mises_strain(data_Ep[:, i]) for i in range(np.shape(data_Ep)[1])])
    node_data_vm_Ep = dataset.mesh.convert_data(data=data_vm_Ep, convert_from="GaussPoint", convert_to="Node",
                                                n_elm_gp=4)
    mesh.point_data['vm_plastic_strain'] = node_data_vm_Ep
    clipped = mesh.clip_scalar(scalars='vm_plastic_strain', invert=False, value=local_plasticity_threshold / 100.0)
    pl.add_mesh(clipped, cmap="YlOrRd")
    pl.add_axes()
    pl.screenshot(figname)

def predict_vm_equivalent_stress(stress_tensor_dict: dict[str, npt.NDArray[np.float_]], plasticity_array_dict: dict[str, npt.NDArray[np.float_]], plasticity_threshold: float = 0.2) -> float:
    ref_vm_stress = sim.Mises_stress(get_stress_tensor_at_plasticity_threshold(stress_tensor_dict["tension_11"], plasticity_array_dict["tension_11"], plasticity_threshold))
    def mse_vm_stress(sigma_eq):
        mse = 0.0
        for load_case in stress_tensor_dict.keys():
            stress_tensor_at_plasticity_threshold = get_stress_tensor_at_plasticity_threshold(stress_tensor_dict[load_case], plasticity_array_dict[load_case], plasticity_threshold)
            vm_stress = sim.Mises_stress(stress_tensor_at_plasticity_threshold)
            func = lambda sigma_eq: (vm_stress - sigma_eq) ** 2
            mse += func(sigma_eq)
        return mse / len(stress_tensor_dict.keys())

    sigma_eq_ident = minimize(mse_vm_stress, ref_vm_stress, method='SLSQP').x

    return sigma_eq_ident

def predict_hill_parameters(stress_tensor_dict: dict[str, npt.NDArray[np.float_]], plasticity_array_dict: dict[str, npt.NDArray[np.float_]], plasticity_threshold: float = 0.2) -> list[float]:
    sigma_eq_ident = predict_vm_equivalent_stress(stress_tensor_dict, plasticity_array_dict)
    p_guess = np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
    def mse_hill_params(p):
        mse = 0.0
        for load_case in stress_tensor_dict.keys():
            stress_tensor_at_plasticity_threshold = get_stress_tensor_at_plasticity_threshold(stress_tensor_dict[load_case], plasticity_array_dict[load_case], plasticity_threshold)
            func = lambda p: (sim.Hill_stress(stress_tensor_at_plasticity_threshold, p) - sigma_eq_ident) ** 2
            mse += func(p)
        return mse / len(stress_tensor_dict.keys())
    hill_params_ident = minimize(mse_hill_params, p_guess, method='SLSQP')
    return hill_params_ident.x

def predict_ani_parameters(stress_tensor_dict: dict[str, npt.NDArray[np.float_]], plasticity_array_dict: dict[str, npt.NDArray[np.float_]], plasticity_threshold: float = 0.2) -> list[float]:
    sigma_eq_ident = predict_vm_equivalent_stress(stress_tensor_dict, plasticity_array_dict)
    p_guess = np.array([1.0, 1.0, 1.0, -0.5, -0.5, -0.5, 1.5, 1.5, 1.5])
    def mse_ani_params(p):
        mse = 0.0
        for load_case in stress_tensor_dict.keys():
            stress_tensor_at_plasticity_threshold = get_stress_tensor_at_plasticity_threshold(stress_tensor_dict[load_case], plasticity_array_dict[load_case], plasticity_threshold)
            func = lambda p: (sim.Ani_stress(stress_tensor_at_plasticity_threshold, p) - sigma_eq_ident) ** 2
            mse += func(p)
        return mse / len(stress_tensor_dict.keys())
    ani_params_ident = minimize(mse_ani_params, p_guess, method='SLSQP')
    return ani_params_ident.x
#ani_stress init: [1.0, 1.0, 1.0, -0.5, -0.5, -0.5, 1.5, 1.5, 1.5]
# [p11, p22, p33, p12, p13, p23, p44, p55, p66]

def predict_dfa_parameters(stress_tensor_dict: dict[str, npt.NDArray[np.float_]], plasticity_array_dict: dict[str, npt.NDArray[np.float_]], plasticity_threshold: float = 0.2) -> list[float]:
    sigma_eq_ident = predict_vm_equivalent_stress(stress_tensor_dict, plasticity_array_dict)
    p_guess = np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 0.0])
    def mse_dfa_params(p):
        mse = 0.0
        for load_case in stress_tensor_dict.keys():
            stress_tensor_at_plasticity_threshold = get_stress_tensor_at_plasticity_threshold(stress_tensor_dict[load_case], plasticity_array_dict[load_case], plasticity_threshold)
            func = lambda p: (sim.DFA_stress(stress_tensor_at_plasticity_threshold, p) - sigma_eq_ident) ** 2
            mse += func(p)
        return mse / len(stress_tensor_dict.keys())
    dfa_params_ident = minimize(mse_dfa_params, p_guess, method='SLSQP')
    return dfa_params_ident.x



def plot_hill_yield_surface(sigma_eq_vm: float, hill_params: list[float],
                            yield_surface_data_s11: npt.NDArray[np.float_],
                            yield_surface_data_s22: npt.NDArray[np.float_],
                            figname="hill_yield.png") -> None:
    inc = 1001

    theta_array = np.linspace(0.0, 2.0 * np.pi, inc, endpoint=True)
    sigma_11 = np.cos(theta_array)
    sigma_22 = np.sin(theta_array)
    sigma = np.zeros(6)

    result = np.zeros(inc)

    for i in range(0, inc):
        sigma[0] = sigma_11[i]
        sigma[1] = sigma_22[i]
        func = lambda seq: abs(seq * sim.Hill_stress(sigma, hill_params) - sigma_eq_vm)
        res = minimize(func, sigma_eq_vm, method='SLSQP')
        result[i] = res.x[0]

    x = result * np.cos(theta_array)
    y = result * np.sin(theta_array)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel(r"$\sigma_{11}$ (MPa)")
    plt.ylabel(r"$\sigma_{22}$ (MPa)")
    ax.xaxis.set_label_coords(1.05, 0.45)
    ax.yaxis.set_label_coords(0.55, 1.05)
    plt.plot(x, y, "--", label="Hill yield surface")
    plt.plot(yield_surface_data_s11, yield_surface_data_s22, "o", label="Simulation data")
    plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.15))
    plt.savefig(figname)
    plt.close()


def plot_hill_yield_surface_evolution(list_sigma_eq_vm: list[float], list_hill_params: list[list[float]], plasticity_threshold_list=list[float],
                                      figname="hill_yield_surface_evolution.png") -> None:

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel(r"$\sigma_{11}$ (MPa)")
    plt.ylabel(r"$\sigma_{22}$ (MPa)")
    ax.xaxis.set_label_coords(1.05, 0.45)
    ax.yaxis.set_label_coords(0.55, 1.05)

    inc = 1001
    theta_array = np.linspace(0, 2.0 * np.pi, inc, endpoint=True)
    sigma_11 = np.cos(theta_array)
    sigma_22 = np.sin(theta_array)
    sigma = np.zeros(6)
    result = np.zeros(inc)

    for i in range(len(plasticity_threshold_list)):

        try:

            for j in range(0, inc):
                sigma[0] = sigma_11[j]
                sigma[1] = sigma_22[j]
                func = lambda seq: abs(seq * sim.Hill_stress(sigma, list_hill_params[i]) - list_sigma_eq_vm[i])
                res = minimize(func, list_sigma_eq_vm[i], method='SLSQP')
                result[j] = res.x[0]

            x = result * np.cos(theta_array)
            y = result * np.sin(theta_array)

            plt.plot(x, y, "--", label=r"$\epsilon^{p}$ " + str(plasticity_threshold_list[i]) + "%")

        except:
            print("Plasticity threshold ", plasticity_threshold_list[i], " not reached.")

    plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.15))
    plt.savefig(figname)
    plt.close()

def plot_ani_yield_surface(sigma_eq_vm: float, ani_params: list[float],
                           yield_surface_data_s11: npt.NDArray[np.float_],
                           yield_surface_data_s22: npt.NDArray[np.float_],
                           figname="ani_yield.png") -> None:
    inc = 1001

    theta_array = np.linspace(0.0, 2.0 * np.pi, inc, endpoint=True)
    sigma_11 = np.cos(theta_array)
    sigma_22 = np.sin(theta_array)
    sigma = np.zeros(6)

    result = np.zeros(inc)

    for i in range(0, inc):
        sigma[0] = sigma_11[i]
        sigma[1] = sigma_22[i]
        func = lambda seq: abs(seq * sim.Ani_stress(sigma, ani_params) - sigma_eq_vm)
        res = minimize(func, sigma_eq_vm, method='SLSQP')
        result[i] = res.x[0]

    x = result * np.cos(theta_array)
    y = result * np.sin(theta_array)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel(r"$\sigma_{11}$ (MPa)")
    plt.ylabel(r"$\sigma_{22}$ (MPa)")
    ax.xaxis.set_label_coords(1.05, 0.45)
    ax.yaxis.set_label_coords(0.55, 1.05)
    plt.plot(x, y, "--", label="Ani yield surface")
    plt.plot(yield_surface_data_s11, yield_surface_data_s22, "o", label="Simulation data")
    plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.15))
    plt.savefig(figname)
    plt.close()

def plot_ani_shear_yield_surface(sigma_eq_vm: float, ani_params: list[float],
                           yield_surface_data_s11: npt.NDArray[np.float_],
                           yield_surface_data_s12: npt.NDArray[np.float_],
                           figname="ani_yield_shear.png") -> None:
    inc = 1001

    theta_array = np.linspace(0.0, 2.0 * np.pi, inc, endpoint=True)
    sigma_11 = np.cos(theta_array)
    sigma_12 = np.sin(theta_array)
    sigma = np.zeros(6)

    result = np.zeros(inc)

    for i in range(0, inc):
        sigma[0] = sigma_11[i]
        sigma[3] = sigma_12[i]
        func = lambda seq: abs(seq * sim.Ani_stress(sigma, ani_params) - sigma_eq_vm)
        res = minimize(func, sigma_eq_vm, method='SLSQP')
        result[i] = res.x[0]

    x = result * np.cos(theta_array)
    y = result * np.sin(theta_array)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel(r"$\sigma_{11}$ (MPa)")
    plt.ylabel(r"$\sigma_{12}$ (MPa)")
    ax.xaxis.set_label_coords(1.05, 0.45)
    ax.yaxis.set_label_coords(0.55, 1.05)
    plt.plot(x, y, "--", label="Anisotropic shear yield surface")
    plt.plot(yield_surface_data_s11, yield_surface_data_s12, "o", label="Simulation data")
    plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.15))
    plt.savefig(figname)
    plt.close()

def plot_ani_yield_surface_evolution(list_sigma_eq_vm: list[float], list_ani_params: list[list[float]], plasticity_threshold_list=list[float],
                                     figname="anisotropic_yield_surface_evolution.png") -> None:

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel(r"$\sigma_{11}$ (MPa)")
    plt.ylabel(r"$\sigma_{22}$ (MPa)")
    ax.xaxis.set_label_coords(1.05, 0.45)
    ax.yaxis.set_label_coords(0.55, 1.05)

    inc = 1001
    theta_array = np.linspace(0, 2.0 * np.pi, inc, endpoint=True)
    sigma_11 = np.cos(theta_array)
    sigma_22 = np.sin(theta_array)
    sigma = np.zeros(6)
    result = np.zeros(inc)

    for i in range(len(plasticity_threshold_list)):

        try:

            for j in range(0, inc):
                sigma[0] = sigma_11[j]
                sigma[1] = sigma_22[j]
                func = lambda seq: abs(seq * sim.Ani_stress(sigma, list_ani_params[i]) - list_sigma_eq_vm[i])
                res = minimize(func, list_sigma_eq_vm[i], method='SLSQP')
                result[j] = res.x[0]

            x = result * np.cos(theta_array)
            y = result * np.sin(theta_array)

            plt.plot(x, y, "--", label=r"$\epsilon^{p}$ " + str(plasticity_threshold_list[i]) + "%")

        except:
            print("Plasticity threshold ", plasticity_threshold_list[i], " not reached.")

    plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.15))
    plt.savefig(figname)
    plt.close()

def plot_dfa_yield_surface(sigma_eq_vm: float, dfa_params: list[float],
                           yield_surface_data_s11: npt.NDArray[np.float_],
                           yield_surface_data_s22: npt.NDArray[np.float_],
                           figname="dfa_yield.png") -> None:
    inc = 1001

    theta_array = np.linspace(0.0, 2.0 * np.pi, inc, endpoint=True)
    sigma_11 = np.cos(theta_array)
    sigma_22 = np.sin(theta_array)
    sigma = np.zeros(6)

    result = np.zeros(inc)

    for i in range(0, inc):
        sigma[0] = sigma_11[i]
        sigma[1] = sigma_22[i]
        func = lambda seq: abs(seq * sim.DFA_stress(sigma, dfa_params) - sigma_eq_vm)
        res = minimize(func, sigma_eq_vm, method='SLSQP')
        result[i] = res.x[0]

    x = result * np.cos(theta_array)
    y = result * np.sin(theta_array)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel(r"$\sigma_{11}$ (MPa)")
    plt.ylabel(r"$\sigma_{22}$ (MPa)")
    ax.xaxis.set_label_coords(1.05, 0.45)
    ax.yaxis.set_label_coords(0.55, 1.05)
    plt.plot(x, y, "--", label="DFA yield surface")
    plt.plot(yield_surface_data_s11, yield_surface_data_s22, "o", label="Simulation data")
    plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.15))
    plt.savefig(figname)
    plt.close()

def plot_dfa_shear_yield_surface(sigma_eq_vm: float, dfa_params: list[float],
                           yield_surface_data_s11: npt.NDArray[np.float_],
                           yield_surface_data_s12: npt.NDArray[np.float_],
                           figname="dfa_yield_shear.png") -> None:
    inc = 1001

    theta_array = np.linspace(0.0, 2.0 * np.pi, inc, endpoint=True)
    sigma_11 = np.cos(theta_array)
    sigma_12 = np.sin(theta_array)
    sigma = np.zeros(6)

    result = np.zeros(inc)

    for i in range(0, inc):
        sigma[0] = sigma_11[i]
        sigma[3] = sigma_12[i]
        func = lambda seq: abs(seq * sim.DFA_stress(sigma, dfa_params) - sigma_eq_vm)
        res = minimize(func, sigma_eq_vm, method='SLSQP')
        result[i] = res.x[0]

    x = result * np.cos(theta_array)
    y = result * np.sin(theta_array)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel(r"$\sigma_{11}$ (MPa)")
    plt.ylabel(r"$\sigma_{12}$ (MPa)")
    ax.xaxis.set_label_coords(1.05, 0.45)
    ax.yaxis.set_label_coords(0.55, 1.05)
    plt.plot(x, y, "--", label="DFA shear yield surface")
    plt.plot(yield_surface_data_s11, yield_surface_data_s12, "o", label="Simulation data")
    plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.15))
    plt.savefig(figname)
    plt.close()

def plot_dfa_yield_surface_evolution(list_sigma_eq_vm: list[float], list_dfa_params: list[list[float]], plasticity_threshold_list=list[float],
                                     figname="dfa_yield_surface_evolution.png") -> None:

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel(r"$\sigma_{11}$ (MPa)")
    plt.ylabel(r"$\sigma_{22}$ (MPa)")
    ax.xaxis.set_label_coords(1.05, 0.45)
    ax.yaxis.set_label_coords(0.55, 1.05)

    inc = 1001
    theta_array = np.linspace(0, 2.0 * np.pi, inc, endpoint=True)
    sigma_11 = np.cos(theta_array)
    sigma_22 = np.sin(theta_array)
    sigma = np.zeros(6)
    result = np.zeros(inc)

    for i in range(len(plasticity_threshold_list)):

        try:

            for j in range(0, inc):
                sigma[0] = sigma_11[j]
                sigma[1] = sigma_22[j]
                func = lambda seq: abs(seq * sim.DFA_stress(sigma, list_dfa_params[i]) - list_sigma_eq_vm[i])
                res = minimize(func, list_sigma_eq_vm[i], method='SLSQP')
                result[j] = res.x[0]

            x = result * np.cos(theta_array)
            y = result * np.sin(theta_array)

            plt.plot(x, y, "--", label=r"$\epsilon^{p}$ " + str(plasticity_threshold_list[i]) + "%")

        except:
            print("Plasticity threshold ", plasticity_threshold_list[i], " not reached.")

    plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.15))
    plt.savefig(figname)
    plt.close()