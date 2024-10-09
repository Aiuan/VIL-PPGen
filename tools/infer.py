import os
import re
import argparse
import yaml
import glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d

from dataset.CustomDataset import CustomDataset
from network.pseudo_point_generation import VILPPGen
from metric.CustomEvaluator import CustomEvaluator
from utils.load_data import load_batch_dict_to_gpu
from utils import read_config, check_and_create_path, save_dict_as_pcd

matplotlib.use('TkAgg')


def build_network(network_cfg, devices, checkpoint_path):
    print('=' * 100)
    print('Build Model')

    model = VILPPGen(config=network_cfg, devices=devices)

    assert os.path.exists(checkpoint_path), 'Do not exist checkpoint: {}'.format(checkpoint_path)
    print(f"Load model's weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=devices[0])
    model.load_state_dict(checkpoint['model'])

    return model


def load_input(root, depth_min, depth_max, point_cloud_range):
    assert os.path.exists(root), 'Do not exist input: {}'.format(root)

    vis_image = cv2.imread(os.path.join(root, 'visible.png'))
    # convert BGR to RGB
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    # convert HWC to CHW
    vis_image = np.transpose(vis_image, (2, 0, 1))
    # convert to float32
    vis_image = vis_image.astype(np.float32)
    # normalize to [0.0, 1.0]
    vis_image = vis_image / 255.0

    vis_sparse_depth = np.load(os.path.join(root, 'sparse_depth_visible.npy'))
    # convert HW to CHW
    vis_sparse_depth = np.expand_dims(vis_sparse_depth, axis=0)
    # convert to float32
    vis_sparse_depth = vis_sparse_depth.astype(np.float32)
    # limit depth
    vis_sparse_depth = np.clip(vis_sparse_depth, depth_min, depth_max)

    inf_image = cv2.imread(os.path.join(root, 'infrared.png'), cv2.IMREAD_GRAYSCALE)
    # convert HW to CHW
    inf_image = np.expand_dims(inf_image, axis=0)
    # convert to float32
    inf_image = inf_image.astype(np.float32)
    # normalize to [0.0, 1.0]
    inf_image = inf_image / 255.0

    inf_sparse_depth = np.load(os.path.join(root, 'sparse_depth_infrared.npy'))
    # convert HW to CHW
    inf_sparse_depth = np.expand_dims(inf_sparse_depth, axis=0)
    # convert to float32
    inf_sparse_depth = inf_sparse_depth.astype(np.float32)
    # limit depth
    inf_sparse_depth = np.clip(inf_sparse_depth, depth_min, depth_max)

    calib = CustomDataset.load_json(os.path.join(root, 'calib.json'))
    # convert to float32
    for key, value in calib.items():
        calib[key] = np.array(value, dtype=np.float32)
    vis_intrinsic = calib['intrinsic_visible']
    inf_intrinsic = calib['intrinsic_infrared']
    vis_extrinsic = calib['extrinsic_lidar2visible']
    inf_extrinsic = calib['extrinsic_lidar2infrared']

    lid_points = CustomDataset.load_pcd(os.path.join(root, 'sparse_lidar.pcd'))
    # convert to float32
    for key, value in lid_points.items():
        lid_points[key] = value.astype(np.float32)
    # normalize intensity to [0.0, 1.0]
    lid_points['intensity'] = lid_points['intensity'] / 255.0
    # format to [x, y, z, intensity]
    lid_points = np.stack((lid_points['x'], lid_points['y'], lid_points['z'], lid_points['intensity']), axis=1)
    # remove points outside point_cloud_range
    mask = (
            (lid_points[:, 0] >= point_cloud_range[0]) & (lid_points[:, 0] <= point_cloud_range[3])
            & (lid_points[:, 1] >= point_cloud_range[1]) & (lid_points[:, 1] <= point_cloud_range[4])
            & (lid_points[:, 2] >= point_cloud_range[2]) & (lid_points[:, 2] <= point_cloud_range[5])
    )
    lid_points = lid_points[mask, :]

    data_dict = {
        'vis_image': vis_image,
        'vis_sparse_depth': vis_sparse_depth,
        'inf_image': inf_image,
        'inf_sparse_depth': inf_sparse_depth,
        'vis_intrinsic': vis_intrinsic,
        'inf_intrinsic': inf_intrinsic,
        'vis_extrinsic': vis_extrinsic,
        'inf_extrinsic': inf_extrinsic,
        'lid_points': lid_points
    }

    enable_evaluator = False
    if os.path.exists(os.path.join(root, 'depth_visible.npy')) and os.path.exists(
            os.path.join(root, 'depth_infrared.npy')):
        enable_evaluator = True

        vis_gt_depth = np.load(os.path.join(root, 'depth_visible.npy'))
        # convert HW to CHW
        vis_gt_depth = np.expand_dims(vis_gt_depth, axis=0)
        # convert to float32
        vis_gt_depth = vis_gt_depth.astype(np.float32)
        # limit depth
        vis_gt_depth = np.clip(vis_gt_depth, depth_min, depth_max)
        data_dict['vis_gt_depth'] = vis_gt_depth

        inf_gt_depth = np.load(os.path.join(root, 'depth_infrared.npy'))
        # convert HW to CHW
        inf_gt_depth = np.expand_dims(inf_gt_depth, axis=0)
        # convert to float32
        inf_gt_depth = inf_gt_depth.astype(np.float32)
        # limit depth
        inf_gt_depth = np.clip(inf_gt_depth, depth_min, depth_max)
        data_dict['inf_gt_depth'] = inf_gt_depth

    data_dict_list = [data_dict]
    batch_dict = {}
    keys = data_dict_list[0].keys()
    for key in keys:
        value_list = [data_dict[key] for data_dict in data_dict_list]
        if key == 'lid_points':
            output = []
            for i, value in enumerate(value_list):
                value_pad = np.pad(value, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                output.append(value_pad)
            batch_dict[key] = np.concatenate(output, axis=0)
        else:
            batch_dict[key] = np.stack(value_list, axis=0)

    return batch_dict, enable_evaluator


def build_evaluator(evaluator_cfg):
    print('=' * 100)
    print('Build Indicator')

    evaluator = CustomEvaluator(config=evaluator_cfg)

    return evaluator


def filter_points(points, ratio=0.1):
    y = points[:, 1]
    z = points[:, 2]
    mask = z <= y * ratio
    points = points[mask, :]
    return points


def show_pcd(xyz, window_name='vis', colors=None, save_path=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=2048, height=1024)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    vis.get_render_option().background_color = np.asarray([1, 1, 1])
    vis.get_render_option().point_size = 4
    vis.get_render_option().show_coordinate_frame = False

    # set view control
    intrinsic = np.array([
        886.81001347526524,
        0.0,
        0.0,
        0.0,
        886.81001347526524,
        0.0,
        1023.5,
        511.5,
        1.0
    ]).reshape((3, 3)).T
    extrinsic = np.array([
        0.971712755542819,
        -0.037362678114438955,
        0.23319166151365392,
        0.0,
        -0.22239314244350805,
        -0.47703119773140834,
        0.8502837917925048,
        0.0,
        0.079470817974167496,
        -0.87809183271173574,
        -0.47184650302345349,
        0.0,
        9.0805358875596927,
        13.024511845827075,
        1.680353306382071,
        1.0
    ]).reshape((4, 4)).T
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
    cam.intrinsic.intrinsic_matrix = intrinsic
    cam.extrinsic = extrinsic
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

    if save_path is None:
        vis.run()
    else:
        buffer = vis.capture_screen_float_buffer(do_render=True)
        pcd_image = cv2.cvtColor((np.asarray(buffer) * 255.0).astype('uint8'), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, pcd_image)
    vis.destroy_window()


def save_output(root, batch_dict, depth_min, depth_max):
    check_and_create_path(root)

    i = 0  # batch size is 1

    vis_coarse_depth = batch_dict['vis_coarse_depth'][i, 0].cpu().numpy()
    plt.imsave(os.path.join(root, 'vis_coarse_depth.png'), vis_coarse_depth, cmap='jet', vmin=depth_min, vmax=depth_max)

    inf_coarse_depth = batch_dict['inf_coarse_depth'][i, 0].cpu().numpy()
    plt.imsave(os.path.join(root, 'inf_coarse_depth.png'), inf_coarse_depth, cmap='jet', vmin=depth_min, vmax=depth_max)

    vis_geopoints = batch_dict['vis_geopoints']
    vis_geopoints = vis_geopoints[vis_geopoints[:, 0] == i, 1:]
    vis_geopoints = vis_geopoints.cpu().numpy()
    save_dict_as_pcd(
        os.path.join(root, 'vis_geopoints.pcd'),
        {
            'x': vis_geopoints[:, 0],
            'y': vis_geopoints[:, 1],
            'z': vis_geopoints[:, 2],
        }
    )
    vis_geopoints_filtered = filter_points(vis_geopoints)
    show_pcd(
        xyz=vis_geopoints_filtered[:, :3],
        window_name='vis_geopoints',
        colors=np.array([68, 114, 196]).reshape(1, -1).repeat(vis_geopoints_filtered.shape[0], axis=0) / 255.0,
        save_path=os.path.join(root, 'vis_geopoints.png')
    )

    inf_geopoints = batch_dict['inf_geopoints']
    inf_geopoints = inf_geopoints[inf_geopoints[:, 0] == i, 1:]
    inf_geopoints = inf_geopoints.cpu().numpy()
    save_dict_as_pcd(
        os.path.join(root, 'inf_geopoints.pcd'),
        {
            'x': inf_geopoints[:, 0],
            'y': inf_geopoints[:, 1],
            'z': inf_geopoints[:, 2],
        }
    )
    inf_geopoints_filtered = filter_points(inf_geopoints)
    show_pcd(
        xyz=inf_geopoints_filtered[:, :3],
        window_name='inf_geopoints',
        colors=np.array([112, 173, 71]).reshape(1, -1).repeat(inf_geopoints_filtered.shape[0], axis=0) / 255.0,
        save_path=os.path.join(root, 'inf_geopoints.png')
    )

    lid_geopoints = batch_dict['lid_geopoints']
    lid_geopoints = lid_geopoints[lid_geopoints[:, 0] == i, 1:]
    lid_geopoints = lid_geopoints.cpu().numpy()
    save_dict_as_pcd(
        os.path.join(root, 'lid_geopoints.pcd'),
        {
            'x': lid_geopoints[:, 0],
            'y': lid_geopoints[:, 1],
            'z': lid_geopoints[:, 2],
        }
    )
    lid_geopoints_filtered = filter_points(lid_geopoints)
    show_pcd(
        xyz=lid_geopoints_filtered[:, :3],
        window_name='lid_geopoints',
        colors=np.array([237, 125, 49]).reshape(1, -1).repeat(lid_geopoints_filtered.shape[0], axis=0) / 255.0,
        save_path=os.path.join(root, 'lid_geopoints.png')
    )

    vis_inf_conf = batch_dict['vis_inf_conf'][i, 0].cpu().numpy()
    plt.imsave(os.path.join(root, 'vis_inf_conf.png'), vis_inf_conf, cmap='jet', vmin=0.0, vmax=1.0)
    vis_inf_off = batch_dict['vis_inf_off'][i, 0].cpu().numpy()
    plt.imsave(os.path.join(root, 'vis_inf_off.png'), vis_inf_off, cmap='jet', vmin=-5, vmax=5)

    vis_lid_conf = batch_dict['vis_lid_conf'][i, 0].cpu().numpy()
    plt.imsave(os.path.join(root, 'vis_lid_conf.png'), vis_lid_conf, cmap='jet', vmin=0.0, vmax=1.0)
    vis_lid_off = batch_dict['vis_lid_off'][i, 0].cpu().numpy()
    plt.imsave(os.path.join(root, 'vis_lid_off.png'), vis_lid_off, cmap='jet', vmin=-5, vmax=5)

    inf_lid_conf = batch_dict['inf_lid_conf'][i, 0].cpu().numpy()
    plt.imsave(os.path.join(root, 'inf_lid_conf.png'), inf_lid_conf, cmap='jet', vmin=0.0, vmax=1.0)
    inf_lid_off = batch_dict['inf_lid_off'][i, 0].cpu().numpy()
    plt.imsave(os.path.join(root, 'inf_lid_off.png'), inf_lid_off, cmap='jet', vmin=-5, vmax=5)

    inf_vis_conf = batch_dict['inf_vis_conf'][i, 0].cpu().numpy()
    plt.imsave(os.path.join(root, 'inf_vis_conf.png'), inf_vis_conf, cmap='jet', vmin=0.0, vmax=1.0)
    inf_vis_off = batch_dict['inf_vis_off'][i, 0].cpu().numpy()
    plt.imsave(os.path.join(root, 'inf_vis_off.png'), inf_vis_off, cmap='jet', vmin=-5, vmax=5)

    vis_refined_depth = batch_dict['vis_refined_depth'][i, 0].cpu().numpy()
    plt.imsave(os.path.join(root, 'vis_refined_depth.png'), vis_refined_depth, cmap='jet', vmin=depth_min,
               vmax=depth_max)

    inf_refined_depth = batch_dict['inf_refined_depth'][i, 0].cpu().numpy()
    plt.imsave(os.path.join(root, 'inf_refined_depth.png'), inf_refined_depth, cmap='jet', vmin=depth_min,
               vmax=depth_max)

    vis_points = batch_dict['vis_points']
    vis_points = vis_points[vis_points[:, 0] == i, 1:]
    vis_points = vis_points.cpu().numpy()
    save_dict_as_pcd(
        os.path.join(root, 'vis_points.pcd'),
        {
            'x': vis_points[:, 0],
            'y': vis_points[:, 1],
            'z': vis_points[:, 2],
            'r': vis_points[:, 3] * 255,
            'g': vis_points[:, 4] * 255,
            'b': vis_points[:, 5] * 255
        }
    )
    vis_points_filtered = filter_points(vis_points)
    show_pcd(
        xyz=vis_points_filtered[:, :3],
        window_name='vis_points',
        colors=vis_points_filtered[:, 3:6],
        save_path=os.path.join(root, 'vis_points.png')
    )

    inf_points = batch_dict['inf_points']
    inf_points = inf_points[inf_points[:, 0] == i, 1:]
    inf_points = inf_points.cpu().numpy()
    save_dict_as_pcd(
        os.path.join(root, 'inf_points.pcd'),
        {
            'x': inf_points[:, 0],
            'y': inf_points[:, 1],
            'z': inf_points[:, 2],
            'infrared': inf_points[:, 3] * 255
        }
    )
    inf_points_filtered = filter_points(inf_points)
    show_pcd(
        xyz=inf_points_filtered[:, :3],
        window_name='inf_points',
        colors=inf_points_filtered[:, 3].reshape(-1, 1).repeat(3, axis=1),
        save_path=os.path.join(root, 'inf_points.png')
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='../config/VIL-PPGen.yaml', help='path of config')
    parser.add_argument('--checkpoint_path', type=str, default='../VIL-PPGen.pth', help='path of checkpoint')
    parser.add_argument('--gpu', type=int, default=None, help='gpu id')
    parser.add_argument('--input_root', type=str, default='../input', help='output root')
    parser.add_argument('--output_root', type=str, default='../output', help='output root')
    return parser.parse_args()


def main():
    args = get_args()

    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    gpu = args.gpu
    input_root = args.input_root
    output_root = args.output_root

    # read config
    config = read_config(config_path)

    # print config
    print('=' * 100)
    print('Config\n' + yaml.dump(config))

    # select devices
    if gpu is None:
        devices_list = config['gpus']
    else:
        devices_list = [gpu]
    devices = []
    print('=' * 100)
    print('Select Devices')
    for i, idx_gpu in enumerate(devices_list):
        devices.append(torch.device('cuda:{}'.format(idx_gpu)))
        print('({}/{}) Select cuda:{}'.format(i + 1, len(devices_list), idx_gpu))

    # build network
    model = build_network(config['network_cfg'], devices, checkpoint_path)

    # load input
    batch_dict, enable_evaluator = load_input(
        input_root, config['depth_min'], config['depth_max'], config['point_cloud_range'])

    if enable_evaluator:
        # build evaluator
        evaluator = build_evaluator(evaluator_cfg=config['evaluator_cfg'])
    else:
        evaluator = None

    print('=' * 100)
    print('Start To Infer')

    with torch.no_grad():
        model.eval()

        batch_dict = load_batch_dict_to_gpu(batch_dict, devices[0])

        batch_dict = model(batch_dict, training=False, need_loss=False)

        # save output
        save_output(output_root, batch_dict, config['depth_min'], config['depth_max'])

        if evaluator is not None:
            evaluator.record(batch_dict)
            metric_dict = evaluator.evaluate()
            for key, value in metric_dict.items():
                print(f'{key}: {value.item():.3f}')

    print('Inference Completed')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('CUDA devices is not avaliable')
        exit()

    main()
