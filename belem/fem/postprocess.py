import numpy as np
import numpy.typing as npt
import fedoo as fd
import matplotlib.pyplot as plt

def compute_stress_strain_plastic_arrays(dataset : fd.DataSet, stress_component: str, plastic_strain_component: str, max_strain: np.float_ = 0.05):
    """Returns stress, strain and plastic strain arrays"""
    component_to_voigt: dict[str, int] = {"XX": 0, "YY": 1, "ZZ": 2, "XY": 3, "XZ": 4, "YZ": 5}
    volume = dataset.mesh.to_pyvista().volume
    n_iter = dataset.n_iter
    stress_array = np.zeros(n_iter)
    plastic_strain_array = np.zeros(n_iter)
    strain_array = 100 * np.linspace(0, max_strain, 101)
    for i in range(n_iter):
        dataset.load(i)
        data_stress = dataset.get_data(field = "Stress", component = stress_component, data_type = "GaussPoint")
        vol_avg_stress = dataset.mesh.integrate_field(field=data_stress, type_field="GaussPoint")/volume
        stress_array[i] = vol_avg_stress
        data_Ep = dataset.get_data(field="Statev", data_type="GaussPoint")[2:8][
            component_to_voigt[plastic_strain_component]]
        vol_avg_Ep = dataset.mesh.integrate_field(field=data_Ep, type_field="GaussPoint") / volume
        plastic_strain_array[i] = vol_avg_Ep

    return strain_array, stress_array, 100*plastic_strain_array

def get_stress_strain_at_plasticity_threshold(stress_array: npt.NDArray[np.float_], plasticity_array: npt.NDArray[np.float_], plasticity_threshold: np.float_, max_strain: np.float_ = 0.05):
    """Fetches point in stress-strain curve where set plasticity threshold is reached"""
    strain_array = 100 * np.linspace(0, max_strain, 101)
    index = next(idx for idx, plasticity in enumerate(plasticity_array) if plasticity > plasticity_threshold)
    strain = strain_array[index]
    stress = stress_array[index]

    return strain, stress

def plot_stress_strain(stress_array: npt.NDArray[np.float_], strain_array: npt.NDArray[np.float_]):
    plt.figure()
    plt.plot(strain_array, stress_array, 'o-')
    plt.title("Stress vs strain")
    plt.xlabel("Strain (%)")
    plt.ylabel("Stress (MPa)")
    plt.savefig("stress_strain.png")
    plt.close()

def plot_hardening(stress_array: npt.NDArray[np.float_], plasticity_array: npt.NDArray[np.float_]):
    plt.figure()
    plt.plot(plasticity_array, stress_array, 'o-')
    plt.title("Stress vs plastic strain")
    plt.xlabel("Plastic strain (%)")
    plt.ylabel("Stress (MPa)")
    plt.savefig("hardening.png")
    plt.close()

def plot_2d_yield_surface(tension_data: fd.DataSet, biaxial_tension_data: fd.DataSet, shear_data: fd.DataSet, compression_data: fd.DataSet, biaxial_compression_data: fd.DataSet, plasticity_threshold: np.float_, figname: str = "stress_at_Ep.png"):
    tension_strain, tension_stress, tension_plastic_strain = compute_stress_strain_arrays(tension_data, "XX", "XX")
    compression_strain, compression_stress, compression_plastic_strain = compute_stress_strain_arrays(compression_data, "XX", "XX")
    biaxial_tension_strain, biaxial_tension_stress, biaxial_tension_plastic_strain = compute_stress_strain_arrays(biaxial_tension_data, "XY", "XY")
    biaxial_compression_strain, biaxial_compression_stress, biaxial_compression_plastic_strain = compute_stress_strain_arrays(biaxial_compression_data, "XY", "XY")
    shear_strain, shear_stress, shear_plastic_strain = compute_stress_strain_arrays(shear_data, "XY", "XY")

    stress_at_plastic_strain_threshold_tension = get_stress_strain_at_plasticity_threshold(tension_stress, tension_plastic_strain, plasticity_threshold)
    stress_at_plastic_strain_threshold_compression = get_stress_strain_at_plasticity_threshold(compression_stress, compression_plastic_strain, plasticity_threshold)
    stress_at_plastic_strain_threshold_biaxial_tension = get_stress_strain_at_plasticity_threshold(biaxial_tension_stress, biaxial_tension_plastic_strain, plasticity_threshold)
    stress_at_plastic_strain_threshold_biaxial_compression = get_stress_strain_at_plasticity_threshold(biaxial_compression_stress, biaxial_compression_plastic_strain, plasticity_threshold)
    stress_at_plastic_strain_threshold_shear = get_stress_strain_at_plasticity_threshold(shear_stress, shear_plastic_strain, plasticity_threshold)

    plt.figure()
    plot_data = [stress_at_plastic_strain_threshold_tension, stress_at_plastic_strain_threshold_biaxial_tension, stress_at_plastic_strain_threshold_tension, stress_at_plastic_strain_threshold_shear, -stress_at_plastic_strain_threshold_compression, -stress_at_plastic_strain_threshold_biaxial_compression, -stress_at_plastic_strain_threshold_compression, stress_at_plastic_strain_threshold_shear, stress_at_plastic_strain_threshold_tension]
    theta = np.deg2rad(np.arange(0, 361, 45))
    rcos = [plot_data[i] * np.cos(theta[i]) for i in range(len(theta))]
    rsin = [plot_data[i] * np.sin(theta[i]) for i in range(len(theta))]
    plt.plot(rcos, rsin, 'o--')
    plt.savefig(figname)