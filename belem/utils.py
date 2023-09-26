import json
import numpy as np

def compute_density_offset_polyfit_from_dataset_json_file(json_filename):
    with open(json_filename) as json_file:
        dataset = json.load(json_file)
    models = {}
    for key, value in dataset.items():
        data_points = np.asarray(value)
        offsets, densities = data_points[:, 0], data_points[:, 1]
        degree = 9
        p = np.polyfit(densities, offsets, degree)
        model = np.poly1d(p)
        models[key] = model

    return models