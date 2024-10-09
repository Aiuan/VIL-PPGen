import os
import yaml
import json
import csv
import numpy as np


def check_and_create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    #     print('Create {}'.format(path))
    # else:
    #     print('Already create {}'.format(path))


def read_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(stream=f, Loader=yaml.FullLoader)
    return config


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_dict_as_json(json_path, dict_data):
    # check nparray, transfer to list
    for k, v in dict_data.items():
        if isinstance(v, np.ndarray):
            dict_data[k] = v.tolist()

    data = json.dumps(dict_data, sort_keys=True, indent=4)
    with open(json_path, 'w', newline='\n') as f:
        f.write(data)


def load_pcd(path):
    with open(path, "r") as f:
        data = f.readlines()
        keys = data[1].replace('\n', '').split(' ')[1:]
        values = np.genfromtxt(data[10:], delimiter=' ')
        pcd = dict()
        for i, key in enumerate(keys):
            pcd[key] = values[:, i]
    return pcd


def save_dict_as_pcd(pcd_path, pcd_dict):
    assert 'x' in pcd_dict.keys() and 'y' in pcd_dict.keys() and 'z' in pcd_dict.keys()

    pcd_dict = pcd_dict.copy()

    # sort keys: x, y, z, ...
    pcd_dict_sorted = {'x': pcd_dict.pop('x'), 'y': pcd_dict.pop('y'), 'z': pcd_dict.pop('z')}
    pcd_dict_sorted.update(pcd_dict)

    keys = list(pcd_dict_sorted.keys())
    n_keys = len(keys)
    n_points = len(pcd_dict_sorted[keys[0]])

    with open(pcd_path, 'w') as file:
        file.write('VERSION .7\n')
        file.write('FIELDS')
        for key in keys:
            file.write(' {}'.format(key))
        file.write('\n')
        file.write('SIZE' + ' 4' * n_keys + '\n')
        file.write('TYPE' + ' F' * n_keys + '\n')
        file.write('COUNT' + ' 1' * n_keys + '\n')
        file.write('WIDTH {}\n'.format(n_points))
        file.write('HEIGHT 1\n')
        file.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        file.write('POINTS {}\n'.format(n_points))
        file.write('DATA ascii\n')

        writer = csv.writer(file, delimiter=' ')
        rows = zip(*pcd_dict_sorted.values())
        writer.writerows(rows)
