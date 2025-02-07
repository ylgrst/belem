import numpy as np
import numpy.typing as npt
import fedoo as fd
import matplotlib.pyplot as plt
import math as m
from simcoon import simmit as sim
from scipy.optimize import minimize, differential_evolution, Bounds
import pyvista as pv
from typing import Optional, Union, List, Tuple, NamedTuple
from tqdm import tqdm

class StrainFromDataset(NamedTuple):
    """
    Class to manage stress data from dataset
    """
    ref_node_id: int
    ref_node_variable: str

def compute_all_arrays_from_data_fields(dataset: fd.MultiFrameDataSet, component: str,
                                        strain_from_dataset: StrainFromDataset) -> dict[str, npt.NDArray[np.float_]]:
    """Returns average stress and strain arrays, von mises stress array, von mises strain array,
    von mises plastic strain array, principal stresses arrays
    :param component: stress array component
    """
    mesh_volume = dataset.mesh.to_pyvista().volume
    rve_volume = dataset.mesh.bounding_box.volume
    density = mesh_volume / rve_volume
    n_iter = dataset.n_iter
    stress_array = np.zeros(n_iter)
    strain_array = np.zeros(n_iter)
    time_array = np.zeros(n_iter)
    stress_component_array_all_iter = np.zeros((n_iter, 6))
    vm_stress_array = np.zeros(n_iter)
    vm_strain_array = np.zeros(n_iter)
    plastic_strain_array = np.zeros(n_iter)
    principal_stresses_array = np.zeros((n_iter, 3))
    output_dict = {}
    for i in tqdm(range(n_iter)):
        dataset.load(i)
        time = dataset.get_data(field="Time")
        time_array[i] = time
        data_stress = dataset.get_data(field="Stress", component=component, data_type="GaussPoint")
        vol_avg_stress = (density / mesh_volume) * dataset.mesh.integrate_field(field=data_stress,
                                                                                type_field="GaussPoint")
        stress_array[i] = vol_avg_stress

        strain = dataset.get_data(field="Disp", component=strain_from_dataset.ref_node_variable[-1],
                                  data_type="Node")[_ref_node_id_to_mesh_node_id(strain_from_dataset.ref_node_id)]
        strain_array[i] = 100 * strain

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
        principal_stresses = _diagonalize_stress_tensor(stress_components_array)
        principal_stresses_array[i] = principal_stresses

    output_dict["time"] = time_array
    output_dict["stress_component"] = stress_array
    output_dict["stress_tensor"] = stress_component_array_all_iter
    output_dict["strain"] = strain_array
    output_dict["vm_stress"] = vm_stress_array
    output_dict["vm_strain"] = vm_strain_array
    output_dict["vm_plastic_strain"] = plastic_strain_array
    output_dict["principal_stresses"] = principal_stresses_array
    return output_dict

def compute_yield_surface_data_from_all_results(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]], plasticity_threshold: float) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    stress_at_plastic_strain_threshold_tension = _get_stress_tensor_at_plasticity_threshold(all_results_dict["tension"]["principal_stresses"], all_results_dict["tension"]["vm_plastic_strain"], plasticity_threshold)
    stress_at_plastic_strain_threshold_compression = _get_stress_tensor_at_plasticity_threshold(
        all_results_dict["compression"]["principal_stresses"], all_results_dict["compression"]["vm_plastic_strain"], plasticity_threshold)
    stress_at_plastic_strain_threshold_biaxial_tension = _get_stress_tensor_at_plasticity_threshold(
        all_results_dict["biaxial_tension"]["principal_stresses"], all_results_dict["biaxial_tension"]["vm_plastic_strain"], plasticity_threshold)
    stress_at_plastic_strain_threshold_biaxial_compression = _get_stress_tensor_at_plasticity_threshold(
        all_results_dict["biaxial_compression"]["principal_stresses"], all_results_dict["biaxial_compression"]["vm_plastic_strain"], plasticity_threshold)
    stress_at_plastic_strain_threshold_tencomp = _get_stress_tensor_at_plasticity_threshold(
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
    stress_at_plastic_strain_threshold_tension = _get_stress_tensor_at_plasticity_threshold(
        all_results_dict["tension"]["stress_component"], all_results_dict["tension"]["vm_plastic_strain"],
        plasticity_threshold)
    stress_at_plastic_strain_threshold_compression = _get_stress_tensor_at_plasticity_threshold(
        all_results_dict["compression"]["stress_component"], all_results_dict["compression"]["vm_plastic_strain"],
        plasticity_threshold)
    stress_at_plastic_strain_threshold_shear = _get_stress_tensor_at_plasticity_threshold(
        all_results_dict["shear"]["stress_component"], all_results_dict["shear"]["vm_plastic_strain"],
        plasticity_threshold)

    plot_data_s11 = [stress_at_plastic_strain_threshold_tension, 0.0, stress_at_plastic_strain_threshold_compression, 0.0]
    plot_data_s12 = [0.0, stress_at_plastic_strain_threshold_shear, 0.0, - stress_at_plastic_strain_threshold_shear]

    return plot_data_s11, plot_data_s12


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

def predict_plasticity_criterion_parameters(criterion: str,
                                            all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]],
                                            plasticity_threshold: float = 0.2) -> list[float]:
    _, stress_tensor_dict, plasticity_array_dict = _build_yield_data_projected_to_all_directions(all_results_dict)
    sigma_eq_ident = _predict_vm_equivalent_stress(all_results_dict)
    if criterion == "hill":
        p_guess = np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
        stress_function = sim.Hill_stress
    elif criterion == "dfa":
        p_guess = np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 0.0])
        stress_function = sim.DFA_stress
    elif criterion == "anisotropic":
        p_guess = np.array([1.0, 1.0, 1.0, -0.5, -0.5, -0.5, 1.5, 1.5, 1.5])
        stress_function = sim.Ani_stress
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Select either hill, dfa or anisotropic")

    def mse_criterion_params(p):
        mse = 0.0
        for load_case in stress_tensor_dict.keys():
            stress_tensor_at_plasticity_threshold = _get_stress_tensor_at_plasticity_threshold(
                stress_tensor_dict[load_case], plasticity_array_dict[load_case], plasticity_threshold)
            func = lambda p: (stress_function(stress_tensor_at_plasticity_threshold, p) - sigma_eq_ident) ** 2
            mse += func(p)
        return mse / len(stress_tensor_dict.keys())
    criterion_params_ident = minimize(mse_criterion_params, p_guess, method="SLSQP")
    return criterion_params_ident.x

def plot_criteria_yield_surface(criterion: str, all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]],
                                criteria_params: list[float],
                                figname: str = "identified_yield_surface") -> None:

    sigma_eq_vm = _predict_vm_equivalent_stress(all_results_dict)
    yield_surface_data_s11, yield_surface_data_s22 = compute_yield_surface_data_from_all_results(all_results_dict, 0.2)

    inc = 1001

    theta_array = np.linspace(0.0, 2.0 * np.pi, inc, endpoint=True)
    sigma_11 = np.cos(theta_array)
    sigma_22 = np.sin(theta_array)
    sigma = np.zeros(6)

    result = np.zeros(inc)

    def identify_yield_surface(stress_function):
        for i in range(0, inc):
            sigma[0] = sigma_11[i]
            sigma[1] = sigma_22[i]
            func = lambda seq: abs(seq * stress_function(sigma, criteria_params) - sigma_eq_vm)
            res = minimize(func, sigma_eq_vm, method='SLSQP')
            result[i] = res.x[0]


    if criterion == "hill":
        if len(criteria_params) != 6:
            raise ValueError("criteria_params must contain 6 parameters for hill criterion")
        identify_yield_surface(sim.Hill_stress)

    elif criterion == "dfa":
        if len(criteria_params) != 7:
            raise ValueError("criteria_params must contain 7 parameters for dfa criterion")
        identify_yield_surface(sim.DFA_stress)

    elif criterion == "anisotropic":
        if len(criteria_params) != 9:
            raise ValueError("criteria_params must contain 9 parameters for anisotropic criterion")
        identify_yield_surface(sim.Ani_stress)

    else:
        raise ValueError(f"Unknown criterion: {criterion}. Select either hill, dfa or anisotropic")

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
    plt.plot(x, y, "--", label=criterion + " yield surface")
    plt.plot(yield_surface_data_s11, yield_surface_data_s22, "o", label="Simulation data")
    plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.15))
    plt.savefig(figname)
    plt.close()

def plot_criteria_shear_yield_surface(criterion: str, all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]],
                                      criteria_params: list[float],
                                      figname: str = "identified_shear_yield_surface") -> None:

    sigma_eq_vm = _predict_vm_equivalent_stress(all_results_dict)
    yield_surface_data_s11, yield_surface_data_s12 = compute_yield_shear_surface_data_from_all_results(all_results_dict, 0.2)

    inc = 1001

    theta_array = np.linspace(0.0, 2.0 * np.pi, inc, endpoint=True)
    sigma_11 = np.cos(theta_array)
    sigma_12 = np.sin(theta_array)
    sigma = np.zeros(6)

    result = np.zeros(inc)

    def identify_yield_surface(stress_function):
        for i in range(0, inc):
            sigma[0] = sigma_11[i]
            sigma[3] = sigma_12[i]
            func = lambda seq: abs(seq * stress_function(sigma, criteria_params) - sigma_eq_vm)
            res = minimize(func, sigma_eq_vm, method='SLSQP')
            result[i] = res.x[0]


    if criterion == "hill":
        if len(criteria_params) != 6:
            raise ValueError("criteria_params must contain 6 parameters for hill criterion")
        identify_yield_surface(sim.Hill_stress)

    elif criterion == "dfa":
        if len(criteria_params) != 7:
            raise ValueError("criteria_params must contain 7 parameters for dfa criterion")
        identify_yield_surface(sim.DFA_stress)

    elif criterion == "anisotropic":
        if len(criteria_params) != 9:
            raise ValueError("criteria_params must contain 9 parameters for anisotropic criterion")
        identify_yield_surface(sim.Ani_stress)

    else:
        raise ValueError(f"Unknown criterion: {criterion}. Select either hill, dfa or anisotropic")

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
    plt.plot(x, y, "--", label=criterion + " shear yield surface")
    plt.plot(yield_surface_data_s11, yield_surface_data_s12, "o", label="Simulation data")
    plt.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.15))
    plt.savefig(figname)
    plt.close()


def compute_average_stress_strain_arrays(dataset: fd.MultiFrameDataSet, component: str,
                                         strain_from_dataset: StrainFromDataset) -> dict[str, npt.NDArray[np.float_]]:
    """Returns average stress and strain arrays"""
    mesh_volume = dataset.mesh.to_pyvista().volume
    rve_volume = dataset.mesh.bounding_box.volume
    density = mesh_volume / rve_volume
    n_iter = dataset.n_iter
    stress_array = np.zeros(n_iter)
    strain_array = np.zeros(n_iter)
    for i in range(n_iter):
        dataset.load(i)
        data_stress = dataset.get_data(field="Stress", component=component, data_type="GaussPoint")
        vol_avg_stress = (density / mesh_volume) * dataset.mesh.integrate_field(field=data_stress,
                                                                                type_field="GaussPoint")
        stress_array[i] = vol_avg_stress

        strain = dataset.get_data(field="Disp", component=strain_from_dataset.ref_node_variable[-1],
                                  data_type="Node")[_ref_node_id_to_mesh_node_id(strain_from_dataset.ref_node_id)]
        strain_array[i] = 100*strain

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
        principal_stresses = _diagonalize_stress_tensor(stress_array)
        principal_stresses_array[i, :] = principal_stresses

    return principal_stresses_array


def create_plastic_strain_and_principal_stress_data(dataset: fd.MultiFrameDataSet) -> tuple[
    npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    p_stresses = compute_principal_stresses(dataset)
    vm_plastic_strain = compute_von_mises_plastic_strain(dataset)

    return vm_plastic_strain, p_stresses



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

def plot_all_vm_stress_vm_strain_from_all_results(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]],
                                                  figname: str = "all_vm_stress_vm_strain.png") -> None:

    sim_to_plot_label = {"tension": "tension", "biaxial_tension": "biaxial tension",
                          "compression": "compression",
                          "biaxial_compression": "biaxial compression",
                          "tencomp": "tension-compression",
                          "shear": "shear"}

    sim_to_linestyle = {"tension": "-", "biaxial_tension": "-",
                          "compression": "--",
                          "biaxial_compression": "--",
                          "tencomp": "-",
                          "shear": "-"}

    plt.figure()
    plt.title("Mises stress vs Mises strain")
    plt.xlabel("Mises strain (%)")
    plt.ylabel("Mises stress (MPa)")
    plt.grid(True)
    for key in sim_to_plot_label.keys():
        strain = all_results_dict[key]["vm_strain"]
        stress = all_results_dict[key]["vm_stress"]
        plt.plot(strain, stress, ls=sim_to_linestyle[key], label=sim_to_plot_label[key])
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

def plot_all_vm_hardening_from_all_results(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]],
                                           figname: str = "all_vm_hardening.png") -> None:

    sim_to_plot_label = {"tension": "tension", "biaxial_tension": "biaxial tension",
                          "compression": "compression",
                          "biaxial_compression": "biaxial compression",
                          "tencomp": "tension-compression",
                          "shear": "shear"}
    sim_to_linestyle = {"tension": "-", "biaxial_tension": "-",
                          "compression": "--",
                          "biaxial_compression": "--",
                          "tencomp": "-",
                          "shear": "-"}

    plt.figure()
    plt.title("Mises stress vs Mises plastic strain")
    plt.xlabel("Mises plastic strain (%)")
    plt.ylabel("Mises stress (MPa)")
    plt.grid(True)
    for key in sim_to_plot_label.keys():
        strain = all_results_dict[key]["vm_plastic_strain"]
        stress = all_results_dict[key]["vm_stress"]
        plt.plot(strain, stress, ls=sim_to_linestyle[key], label=sim_to_plot_label[key])
    plt.legend()
    plt.savefig(figname)
    plt.close()


def compute_all_sim_rp02_from_all_results(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]]) -> dict[str, float]:
    plasticity_threshold = 0.2
    stress_at_plastic_strain_threshold_tension = _get_stress_tensor_at_plasticity_threshold(
        all_results_dict["tension"]["vm_stress"], all_results_dict["tension"]["vm_plastic_strain"],
        plasticity_threshold)
    stress_at_plastic_strain_threshold_compression = _get_stress_tensor_at_plasticity_threshold(
        all_results_dict["compression"]["vm_stress"], all_results_dict["compression"]["vm_plastic_strain"],
        plasticity_threshold)
    stress_at_plastic_strain_threshold_biaxial_tension = _get_stress_tensor_at_plasticity_threshold(
        all_results_dict["biaxial_tension"]["vm_stress"],
        all_results_dict["biaxial_tension"]["vm_plastic_strain"], plasticity_threshold)
    stress_at_plastic_strain_threshold_biaxial_compression = _get_stress_tensor_at_plasticity_threshold(
        all_results_dict["biaxial_compression"]["vm_stress"],
        all_results_dict["biaxial_compression"]["vm_plastic_strain"], plasticity_threshold)
    stress_at_plastic_strain_threshold_tencomp = _get_stress_tensor_at_plasticity_threshold(
        all_results_dict["tencomp"]["vm_stress"], all_results_dict["tencomp"]["vm_plastic_strain"],
        plasticity_threshold)
    stress_at_plastic_strain_threshold_shear = _get_stress_tensor_at_plasticity_threshold(
        all_results_dict["shear"]["vm_stress"], all_results_dict["shear"]["vm_plastic_strain"],
        plasticity_threshold)

    return {"tension": stress_at_plastic_strain_threshold_tension, "biaxial_tension": stress_at_plastic_strain_threshold_biaxial_tension,
            "compression": stress_at_plastic_strain_threshold_compression, "biaxial_compression": stress_at_plastic_strain_threshold_biaxial_compression,
            "tencomp": stress_at_plastic_strain_threshold_tencomp, "shear": stress_at_plastic_strain_threshold_shear}


def analyse_last_frame(dataset: fd.MultiFrameDataSet) -> Tuple[float]:
    """Computes max local plastic strain, ratio between max local and global plastic strains, and max local stress at 5% strain"""
    dataset.load(-1)
    mesh_volume = dataset.mesh.to_pyvista().volume
    data_plastic_Ep = dataset.get_data(field="Statev", data_type="GaussPoint")[2:8]
    data_vm_plastic_Ep = np.asarray(
        [sim.Mises_strain(data_plastic_Ep[:, i]) for i in range(np.shape(data_plastic_Ep)[1])])
    vol_avg_vm_plastic_Ep = dataset.mesh.integrate_field(field=data_vm_plastic_Ep,
                                                         type_field="GaussPoint") / mesh_volume
    data_vm_stress = dataset.get_data(field="Stress", component="vm", data_type="GaussPoint")
    max_local_plastic_Ep = max(data_vm_plastic_Ep)
    max_local_stress = max(data_vm_stress)

    return (max_local_stress, max_local_plastic_Ep, max_local_plastic_Ep/vol_avg_vm_plastic_Ep)


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

def multiplot_clipped_vm_plastic_strain(dataset: fd.MultiFrameDataSet, global_clip_values: Union[List[int], List[float]],
                                        local_plasticity_threshold: float, figname: str) -> None:
    """Plots the Von Mises plastic strain in a mesh, clipped by local plasticity threshold (given in %),
    at simulation increments (give either list of increment indices or global plasticity thresholds (given in %))"""
    global_vm_plasticity_array = compute_von_mises_plastic_strain(dataset)
    for global_clip_value in global_clip_values:
        if type(global_clip_value) == int:
            index = global_clip_value
        elif type(global_clip_value) == float:
            index = next(
                idx for idx, plasticity in enumerate(global_vm_plasticity_array) if plasticity > global_clip_value)
        else:
            raise TypeError("Clip values must be either int or float")
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
        pl.screenshot(figname + "_threshold_" + str(global_clip_value) + ".png")

def _ref_node_id_to_mesh_node_id(ref_node_id: int) -> int:
    if ref_node_id == 0:
        mesh_node_id = -2
    elif ref_node == 1:
        mesh_node_id = -1
    else:
        raise ValueError("Invalid ref_node_id (select either 0 or 1)")

    return mesh_node_id


def _build_yield_data_projected_to_all_directions(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]]) -> list[dict[str, tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]]]:
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


def _diagonalize_stress_tensor(voigt_stress_tensor: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
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

def _get_stress_at_plasticity_threshold(stress_array: npt.NDArray[np.float_],
                                        plasticity_array: npt.NDArray[np.float_], plasticity_threshold: float) -> \
        npt.NDArray[np.float_]:
    """Fetches point in stress-strain curve where set plasticity threshold is reached"""
    try:
        index = next(idx for idx, plasticity in enumerate(plasticity_array) if plasticity > plasticity_threshold)
    except StopIteration:
        index = len(plasticity_array) - 1
    stress = stress_array[:, index]

    return stress

def _get_stress_tensor_at_plasticity_threshold(stress_array: npt.NDArray[np.float_], plasticity_array: npt.NDArray[np.float_], plasticity_threshold: float) -> npt.NDArray[np.float_]:
    """Fetches stress array where set plasticity threshold is reached"""
    try:
        index = next(idx for idx, plasticity in enumerate(plasticity_array) if plasticity > plasticity_threshold)
    except StopIteration:
        index = len(plasticity_array) - 1
    stress = stress_array[index]

    return stress



def _predict_vm_equivalent_stress(all_results_dict: dict[str, dict[str, npt.NDArray[np.float_]]],
                                  plasticity_threshold: float = 0.2) -> float:
    _, stress_tensor_dict, plasticity_array_dict = _build_yield_data_projected_to_all_directions(all_results_dict)
    ref_vm_stress = sim.Mises_stress(_get_stress_tensor_at_plasticity_threshold(stress_tensor_dict["tension_11"], plasticity_array_dict["tension_11"], plasticity_threshold))
    def mse_vm_stress(sigma_eq):
        mse = 0.0
        for load_case in stress_tensor_dict.keys():
            stress_tensor_at_plasticity_threshold = _get_stress_tensor_at_plasticity_threshold(stress_tensor_dict[load_case], plasticity_array_dict[load_case], plasticity_threshold)
            vm_stress = sim.Mises_stress(stress_tensor_at_plasticity_threshold)
            func = lambda sigma_eq: (vm_stress - sigma_eq) ** 2
            mse += func(sigma_eq)
        return mse / len(stress_tensor_dict.keys())

    sigma_eq_ident = minimize(mse_vm_stress, ref_vm_stress, method='SLSQP').x

    return sigma_eq_ident
