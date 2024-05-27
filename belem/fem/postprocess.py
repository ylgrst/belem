import numpy as np
import fedoo as fd
import math as m
import matplotlib.pyplot as plt

def plot_stress_strain_curve_uniaxial_tension(dataset : fd.DataSet, stress_component: "str", cell_size: float):
    volume = dataset.mesh.to_pyvista().volume
    n_iter = dataset.n_iter
    y_array = np.zeros(n_iter)
    disp_array = np.linspace(0, 0.2, 101)
    x_array = [100 * disp / cell_size for disp in disp_array]
    for i in range(n_iter):
        dataset.load(i)
        data_y = dataset.get_data(field = "Stress", component = stress_component, data_type = "GaussPoint")
        vol_avg_field_y = dataset.mesh.integrate_field(field=data_y, type_field="GaussPoint")/volume
        y_array[i] = vol_avg_field_y
    plt.figure()
    plt.title("Stress vs strain")
    plt.xlabel('Strain (%)')
    plt.ylabel('Stress (MPa)')
    plt.plot(x_array, y_array, 'o-')
    plt.savefig("stress_strain.png")