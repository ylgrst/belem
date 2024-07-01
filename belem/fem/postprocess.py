import numpy as np
import numpy.typing as npt
import fedoo as fd
import matplotlib.pyplot as plt
import math as m
from simcoon import simmit as sim
from scipy.optimize import fsolve
import pyvista as pv


def compute_average_stress_strain_arrays(dataset: fd.DataSet, component: str, max_strain: np.float_ = 0.05) -> dict[
    str, npt.NDArray[np.float_]]:
    """Returns average stress and strain arrays"""
    mesh_volume = dataset.mesh.to_pyvista().volume
    rve_volume = dataset.mesh.bounding_box.volume
    density = mesh_volume / rve_volume
    n_iter = dataset.n_iter
    stress_array = np.zeros(n_iter)
    strain_array = 100 * np.linspace(0, max_strain, 101)
    for i in range(n_iter):
        dataset.load(i)
        data_stress = dataset.get_data(field="Stress", component=component, data_type="GaussPoint")
        vol_avg_stress = (density / mesh_volume) * dataset.mesh.integrate_field(field=data_stress,
                                                                                type_field="GaussPoint")
        stress_array[i] = vol_avg_stress

    return {"strain": strain_array, "stress": stress_array}

def compute_von_mises_stress(dataset: fd.DataSet) -> npt.NDArray[np.float_]:
    """Returns von mises stress"""
    mesh_volume = dataset.mesh.to_pyvista().volume
    rve_volume = dataset.mesh.bounding_box.volume
    density = mesh_volume/rve_volume
    n_iter = dataset.n_iter
    vm_stress_array = np.zeros(n_iter)
    for i in range(n_iter):
        dataset.load(i)
        data_vm_stress = dataset.get_data(field="Stress", component="vm", data_type="GaussPoint")
        vol_avg_stress = (density/mesh_volume)*dataset.mesh.integrate_field(field=data_vm_stress, type_field="GaussPoint")
        vm_stress_array[i] = vol_avg_stress

    return vm_stress_array

def compute_von_mises_plastic_strain(dataset: fd.DataSet) -> npt.NDArray[np.float_]:
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


def compute_principal_stresses(dataset: fd.DataSet) -> npt.NDArray[np.float_]:
    """Returns principal stresses arrays"""
    component_to_voigt: dict[str, int] = {"XX": 0, "YY": 1, "ZZ": 2, "XY": 3, "XZ": 4, "YZ": 5}
    component_list = ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]
    mesh_volume = dataset.mesh.to_pyvista().volume
    rve_volume = dataset.mesh.bounding_box.volume
    density = mesh_volume / rve_volume
    n_iter = dataset.n_iter
    principal_stresses_array = np.zeros((3, n_iter))
    for i in range(n_iter):
        dataset.load(i)
        stress_array = np.zeros(6)
        for component in component_list:
            data_stress = dataset.get_data(field="Stress", component=component, data_type="GaussPoint")
            vol_avg_stress = (density / mesh_volume) * dataset.mesh.integrate_field(field=data_stress,
                                                                                    type_field="GaussPoint")
            stress_array[component_to_voigt[component]] = vol_avg_stress
        principal_stresses = diagonalize_stress_tensor(stress_array)
        principal_stresses_array[:, i] = principal_stresses

    return principal_stresses_array


def create_plastic_strain_and_principal_stress_data(dataset: fd.DataSet) -> tuple[
    npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    p_stresses = compute_principal_stresses(dataset)
    vm_plastic_strain = compute_von_mises_plastic_strain(dataset)

    return vm_plastic_strain, p_stresses


def get_stress_strain_at_plasticity_threshold(stress_array: npt.NDArray[np.float_],
                                              plasticity_array: npt.NDArray[np.float_], plasticity_threshold: float) -> npt.NDArray[np.float_]:
    """Fetches point in stress-strain curve where set plasticity threshold is reached"""
    index = next(idx for idx, plasticity in enumerate(plasticity_array) if plasticity > plasticity_threshold)
    stress = stress_array[:, index]

    return stress

def get_vm_stress_at_plasticity_threshold(vm_stress_array: npt.NDArray[np.float_], vm_plasticity_array: npt.NDArray[np.float_], plasticity_threshold: float) -> float:
    index = next(idx for idx, plasticity in enumerate(vm_plasticity_array) if plasticity > plasticity_threshold)
    vm_stress = vm_stress_array[index]

    return vm_stress


def compute_average_stress_tensor(dataset: fd.DataSet) -> npt.NDArray[np.float_]:
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


def plot_hardening(stress_array: npt.NDArray[np.float_], plasticity_array: npt.NDArray[np.float_],
                   figname: str = "hardening.png"):
    plt.figure()
    plt.plot(plasticity_array, stress_array, 'o-')
    plt.title("Stress vs plastic strain")
    plt.xlabel("Plastic strain (%)")
    plt.ylabel("Stress (MPa)")
    plt.savefig(figname)
    plt.close()


def compute_yield_surface_data(tension_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                               biaxial_tension_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                               tencomp_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                               compression_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                               biaxial_compression_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                               plasticity_threshold: float) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    stress_at_plastic_strain_threshold_tension = get_stress_strain_at_plasticity_threshold(tension_data[1],
                                                                                           tension_data[0],
                                                                                           plasticity_threshold)
    stress_at_plastic_strain_threshold_compression = get_stress_strain_at_plasticity_threshold(compression_data[1],
                                                                                               compression_data[0],
                                                                                               plasticity_threshold)
    stress_at_plastic_strain_threshold_biaxial_tension = get_stress_strain_at_plasticity_threshold(
        biaxial_tension_data[1], biaxial_tension_data[0], plasticity_threshold)
    stress_at_plastic_strain_threshold_biaxial_compression = get_stress_strain_at_plasticity_threshold(
        biaxial_compression_data[1], biaxial_compression_data[0], plasticity_threshold)
    stress_at_plastic_strain_threshold_tencomp = get_stress_strain_at_plasticity_threshold(tencomp_data[1], tencomp_data[0],
                                                                                         plasticity_threshold)

    plot_data_s11 = [stress_at_plastic_strain_threshold_tension[0],
                     stress_at_plastic_strain_threshold_biaxial_tension[0], 0.0,
                     -stress_at_plastic_strain_threshold_tencomp[0], -stress_at_plastic_strain_threshold_compression[0],
                     -stress_at_plastic_strain_threshold_biaxial_compression[0], 0.0,
                     stress_at_plastic_strain_threshold_tencomp[0], stress_at_plastic_strain_threshold_tension[0]]
    plot_data_s22 = [0.0, stress_at_plastic_strain_threshold_biaxial_tension[1],
                     stress_at_plastic_strain_threshold_tension[0], stress_at_plastic_strain_threshold_tencomp[1], 0.0,
                     -stress_at_plastic_strain_threshold_biaxial_compression[1],
                     -stress_at_plastic_strain_threshold_tension[0], -stress_at_plastic_strain_threshold_tencomp[1], 0.0]

    return plot_data_s11, plot_data_s22


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
    ax.yaxis.set_label_coords(0.45, 1.05)

    plot_data_s11, plot_data_s22 = compute_yield_surface_data(tension_data, biaxial_tension_data, tencomp_data,
                                                              compression_data, biaxial_compression_data,
                                                              plasticity_threshold)

    plt.plot(plot_data_s11, plot_data_s22, "o--")

    plt.savefig(figname)


def plot_yield_surface_evolution(tension_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                                 biaxial_tension_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                                 tencomp_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                                 compression_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                                 biaxial_compression_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                                 plasticity_threshold_list: list[float], figname: str = "yield_surface_evolution.png") -> None:

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
    ax.yaxis.set_label_coords(0.45, 1.05)

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

def plot_clipped_vm_plastic_strain(dataset: fd.DataSet, threshold: float, figname: str) -> None:
    """Plots the Von Mises plastic strain in a mesh, clipped by threshold, at the end of a simulation"""
    dataset.load(-1)
    canvas_mesh = dataset.mesh.to_pyvista()
    mesh = canvas_mesh
    edges = mesh.extract_feature_edges()
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(canvas_mesh, color="lightblue", opacity=0.1)
    pl.add_mesh(edges, color="black", line_width=3)
    data_Ep = dataset.get_data(field="Statev", data_type="GaussPoint")[2:8]
    data_vm_Ep = np.asarray([sim.Mises_strain(data_Ep[:, i]) for i in range(np.shape(data_Ep)[1])])
    node_data_vm_Ep = dataset.mesh.convert_data(data=data_vm_Ep, convert_from="GaussPoint", convert_to="Node", n_elm_gp=4)
    mesh.point_data['vm_plastic_strain'] = node_data_vm_Ep
    clipped = mesh.clip_scalar(scalars='vm_plastic_strain', invert=False, value=threshold)
    pl.add_mesh(clipped, cmap="YlOrRd")
    pl.add_axes()
    pl.screenshot(figname)

def compute_quadratic_hill_params(tension_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                                 biaxial_tension_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]],
                                 shear_data: tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
    """Computes F, G, H, L, M, N parameters from Hill yield criterion"""

    plasticity_threshold = 0.2
    sigma_yield_11 = get_stress_strain_at_plasticity_threshold(tension_data[1], tension_data[0], plasticity_threshold)[0]
    sigma_yield_22 = get_stress_strain_at_plasticity_threshold(biaxial_tension_data[1], biaxial_tension_data[0], plasticity_threshold)[1]
    sigma_yield_33 = sigma_yield_11
    tau_yield_12 = get_stress_strain_at_plasticity_threshold(shear_data[1], shear_data[0], plasticity_threshold)[0]
    tau_yield_13 = tau_yield_12
    tau_yield_23 = tau_yield_12

    F = 0.5 * ((1.0/(sigma_yield_22)**2) + (1.0/(sigma_yield_33)**2) - (1.0/(sigma_yield_11)**2))
    G = 0.5 * ((1.0/(sigma_yield_33)**2) + (1.0/(sigma_yield_11)**2) - (1.0/(sigma_yield_22)**2))
    H = 0.5 * ((1.0/(sigma_yield_11)**2) + (1.0/(sigma_yield_22)**2) - (1.0/(sigma_yield_33)**2))
    L = 0.5 * (1/(tau_yield_23)**2)
    M = 0.5 * (1 / (tau_yield_13) ** 2)
    N = 0.5 * (1 / (tau_yield_12) ** 2)

    return np.array([F, G, H, L, M, N])

def compute_hill_equivalent_stress(hill_params: npt.NDArray, stress_vector: npt.NDArray) -> float:
    F, G, H, L, M, N = hill_params
    mat = np.array([[H+G, -H, -G, 0.0, 0.0, 0.0],
                    [-H, F+H, -F, 0.0, 0.0, 0.0],
                    [-G, -F, F+G, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, N, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, M, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, L]
                    ])
    sigma_eq_squared = (3/2) * stress_vector @ mat @ stress_vector

    return m.sqrt(sigma_eq_squared)

def compute_hill_yield_surface_data(hill_params: npt.NDArray[np.float_], tension_dataset: fd.DataSet) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    theta_array = np.linspace(0, 2*np.pi, 1000)
    r_list = []
    F, G, H, L, M, N = hill_params
    hill_mat = np.array([[H+G, -H, -G, 0.0, 0.0, 0.0],
                    [-H, F+H, -F, 0.0, 0.0, 0.0],
                    [-G, -F, F+G, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, N, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, M, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, L]
                    ])
    avg_sigma = get_stress_strain_at_plasticity_threshold(compute_average_stress_tensor(tension_dataset), compute_von_mises_plastic_strain(tension_dataset), plasticity_threshold=0.2)
    sigma_eq = compute_hill_equivalent_stress(hill_params, avg_sigma)
    sigma_y = get_vm_stress_at_plasticity_threshold(compute_von_mises_stress(tension_dataset), compute_von_mises_plastic_strain(tension_dataset), 0.2)
    for theta in theta_array:
        mat = np.array([[np.cos(theta), np.sin(theta), 0.0],
                        [-np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 0.0]])
        stress_vector_theta = np.array([mat[0,0], mat[1,1], mat[2,2], mat[0,1], mat[0,2], mat[1,2]])
        func = lambda r: F*(r*mat[1,1] - r*mat[2,2])**2 + G*(r*mat[2,2] - r*mat[0,0])**2 + H*(r*mat[0,0] - r*mat[1,1])**2 + 2*L*(r*mat[1,2])**2 + 2*M*(r*mat[2,0])**2 + 2*N*(r*mat[0,1])**2 - 1.0
        #func = lambda r: (3/2)*(r*stress_vector_theta) @ hill_mat @ (r*stress_vector_theta) - sigma_y**2
        result = fsolve(func, sigma_eq)
        r_list.append(result)

    return np.asarray(r_list)[:,0], theta_array

def plot_hill_yield_surface(r_array: npt.NDArray[np.float_], theta_array: npt.NDArray[np.float_], figname="hill_yield.png") -> None:
    x = r_array * np.cos(theta_array)
    y = r_array * np.sin(theta_array)

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
    ax.yaxis.set_label_coords(0.45, 1.05)
    plt.plot(x, y, "--")
    plt.savefig(figname)

