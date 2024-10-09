import os
import re
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import openpyxl

from dataset.CustomDataset import CustomDataset
from network.pseudo_point_generation import VILPPGen
from metric.CustomEvaluator import CustomEvaluator
from utils.eval_tools import init_path_for_evaluating, EvaluatingParams
from utils.load_data import load_batch_dict_to_gpu
from utils import check_and_create_path, save_dict_as_pcd, save_dict_as_json


def parse_work_dir(work_dir):
    if os.path.exists(work_dir):
        print(f'{work_dir} is exist.')
        return work_dir

    # try to supplement work_dir with latest timestamp
    dirname = os.path.dirname(work_dir)
    basename = os.path.basename(work_dir)
    timestamp_pattern = r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}'
    pattern = re.compile(rf'^{basename}_{timestamp_pattern}$')
    items = []
    for item in os.listdir(dirname):
        match = re.search(pattern, item)
        if match is not None:
            items.append(item)
    items.sort()

    if len(items) == 0:
        print(f'{work_dir} is not exist.')
        exit()
    else:
        work_dir_complete = os.path.join(dirname, items[-1])
        print(f'{work_dir} is uncomplete, use {work_dir_complete} instead.')
        return work_dir_complete


def build_dataset_and_dataloader(dataset_cfg, dataloader_cfg):
    print('=' * 100)
    print('Build Dataset And Dataloader')

    dataset = CustomDataset(config=dataset_cfg)
    print('The length of dataset: {}'.format(len(dataset)))

    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_cfg['batch_size'],
        collate_fn=dataset.collate_batch,
        shuffle=False,
        num_workers=dataloader_cfg['num_workers'],
        drop_last=False,
        pin_memory=False
    )
    print('The length of dataloader: {}'.format(len(dataloader)))

    return dataset, dataloader


def build_network(network_cfg, devices, checkpoint_path):
    print('=' * 100)
    print('Build Network')

    model = VILPPGen(config=network_cfg, devices=devices)

    print(f"Load model's weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=devices[0])
    model.load_state_dict(checkpoint['model'])

    return model


def build_evaluator(evaluator_cfg):
    print('=' * 100)
    print('Build Indicator')

    evaluator = CustomEvaluator(config=evaluator_cfg)

    return evaluator


def dump_image(output_root, batch_dict, depth_max, depth_min):
    image_dir = os.path.join(output_root, 'image')
    check_and_create_path(image_dir)

    batch_size = batch_dict['name'].shape[0]
    for i in range(batch_size):
        vis_image = batch_dict['vis_image'][i].permute(1, 2, 0).cpu().numpy()
        vis_image = vis_image[:, :, [2, 1, 0]] * 255
        vis_image = vis_image.astype('uint8')

        vis_sparse_depth = batch_dict['vis_sparse_depth'][i].permute(1, 2, 0).cpu().numpy()
        vis_sparse_depth = (vis_sparse_depth - depth_min) / (depth_max - depth_min) * 255
        vis_sparse_depth = vis_sparse_depth.astype('uint8')
        vis_sparse_depth = cv2.applyColorMap(vis_sparse_depth, cv2.COLORMAP_JET)

        inf_image = batch_dict['inf_image'][i].tile((3, 1, 1)).permute(1, 2, 0).cpu().numpy()
        inf_image = inf_image * 255
        inf_image = inf_image.astype('uint8')

        inf_sparse_depth = batch_dict['inf_sparse_depth'][i].permute(1, 2, 0).cpu().numpy()
        inf_sparse_depth = (inf_sparse_depth - depth_min) / (depth_max - depth_min) * 255
        inf_sparse_depth = inf_sparse_depth.astype('uint8')
        inf_sparse_depth = cv2.applyColorMap(inf_sparse_depth, cv2.COLORMAP_JET)

        vis_refined_depth = batch_dict['vis_refined_depth'][i].permute(1, 2, 0).cpu().numpy()
        vis_refined_depth = (vis_refined_depth - depth_min) / (depth_max - depth_min) * 255
        vis_refined_depth = vis_refined_depth.astype('uint8')
        vis_refined_depth = cv2.applyColorMap(vis_refined_depth, cv2.COLORMAP_JET)

        vis_gt_depth = batch_dict['vis_gt_depth'][i].permute(1, 2, 0).cpu().numpy()
        vis_gt_depth = (vis_gt_depth - depth_min) / (depth_max - depth_min) * 255
        vis_gt_depth = vis_gt_depth.astype('uint8')
        vis_gt_depth = cv2.applyColorMap(vis_gt_depth, cv2.COLORMAP_JET)

        inf_refined_depth = batch_dict['inf_refined_depth'][i].permute(1, 2, 0).cpu().numpy()
        inf_refined_depth = (inf_refined_depth - depth_min) / (depth_max - depth_min) * 255
        inf_refined_depth = inf_refined_depth.astype('uint8')
        inf_refined_depth = cv2.applyColorMap(inf_refined_depth, cv2.COLORMAP_JET)

        inf_gt_depth = batch_dict['inf_gt_depth'][i].permute(1, 2, 0).cpu().numpy()
        inf_gt_depth = (inf_gt_depth - depth_min) / (depth_max - depth_min) * 255
        inf_gt_depth = inf_gt_depth.astype('uint8')
        inf_gt_depth = cv2.applyColorMap(inf_gt_depth, cv2.COLORMAP_JET)

        image = cv2.vconcat([vis_image, vis_sparse_depth,
                             inf_image, inf_sparse_depth,
                             vis_refined_depth, vis_gt_depth,
                             inf_refined_depth, inf_gt_depth])
        image_name = batch_dict['name'][i]
        image_path = os.path.join(image_dir, f'{image_name}.png')
        cv2.imwrite(image_path, image)


def dump_depth(output_root, batch_dict):
    depth_dir = os.path.join(output_root, 'depth')
    check_and_create_path(depth_dir)

    vis_depth_dir = os.path.join(depth_dir, 'depth_visible')
    check_and_create_path(vis_depth_dir)

    inf_depth_dir = os.path.join(depth_dir, 'depth_infrared')
    check_and_create_path(inf_depth_dir)

    batch_size = batch_dict['name'].shape[0]
    for i in range(batch_size):
        name = batch_dict['name'][i]

        vis_depth = batch_dict['vis_refined_depth'][i].cpu().numpy()
        vis_depth = vis_depth.squeeze(axis=0).astype('float32')
        np.save(os.path.join(vis_depth_dir, f'{name}.npy'), vis_depth)

        inf_depth = batch_dict['inf_refined_depth'][i].cpu().numpy()
        inf_depth = inf_depth.squeeze(axis=0).astype('float32')
        np.save(os.path.join(inf_depth_dir, f'{name}.npy'), inf_depth)


def dump_point(output_root, batch_dict):
    point_dir = os.path.join(output_root, 'point')
    check_and_create_path(point_dir)

    vis_point_dir = os.path.join(point_dir, 'points_visible')
    check_and_create_path(vis_point_dir)

    inf_point_dir = os.path.join(point_dir, 'points_infrared')
    check_and_create_path(inf_point_dir)

    batch_size = batch_dict['name'].shape[0]
    points_visible = batch_dict['vis_points'].cpu().numpy()
    points_infrared = batch_dict['inf_points'].cpu().numpy()
    for i in range(batch_size):
        name = batch_dict['name'][i]

        pts_visible = points_visible[points_visible[:, 0] == i][:, 1:]
        dict_visible = {
            'x': pts_visible[:, 0],
            'y': pts_visible[:, 1],
            'z': pts_visible[:, 2],
            'r': (pts_visible[:, 3] * 255).round(),
            'g': (pts_visible[:, 4] * 255).round(),
            'b': (pts_visible[:, 5] * 255).round()
        }
        save_dict_as_pcd(os.path.join(vis_point_dir, f'{name}.pcd'), dict_visible)

        pts_infrared = points_infrared[points_infrared[:, 0] == i][:, 1:]
        dict_infrared = {
            'x': pts_infrared[:, 0],
            'y': pts_infrared[:, 1],
            'z': pts_infrared[:, 2],
            'infrared': (pts_infrared[:, 3] * 255).round()
        }
        save_dict_as_pcd(os.path.join(inf_point_dir, f'{name}.pcd'), dict_infrared)


def save_metric(output_root, metric_dict):
    save_dict_as_json(os.path.join(output_root, 'metric.json'), metric_dict)

    metrics = {}
    for key in metric_dict.keys():
        items = key.replace('metric_', '').split('_')
        row_name = items[0]
        col_name = '_'.join(items[1:]) if len(items) > 1 else ''

        if row_name not in metrics.keys():
            metrics[row_name] = {}

        metrics[row_name][col_name] = metric_dict[key]
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_excel(os.path.join(output_root, 'metric.xlsx'), index=True, header=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--work_dir', type=str,
        default='../work_dir/VIL-PPGen', help='path of work directory'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default='latest.pth', help='name of checkpoint'
    )
    parser.add_argument(
        '--note', type=str,
        default=None, help='note of eval directory'
    )
    parser.add_argument(
        '--gpu', type=int,
        default=None, help='gpu id'
    )
    parser.add_argument(
        '--dataset_split', type=str,
        default='test', help='dataset split'
    )
    parser.add_argument('--save_image', action='store_true', help='save image')
    parser.add_argument('--save_depth', action='store_true', help='save depth')
    parser.add_argument('--save_point', action='store_true', help='save point')

    return parser.parse_args()


def main():
    args = get_args()

    work_dir = parse_work_dir(args.work_dir)
    checkpoint = args.checkpoint
    note = args.note
    gpu = args.gpu
    dataset_split = args.dataset_split
    save_image = args.save_image
    save_depth = args.save_depth
    save_point = args.save_point

    config = init_path_for_evaluating(work_dir, checkpoint, note, dataset_split)

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

    print('=' * 100)
    print('Init EvaluatingParams')
    eval_params = EvaluatingParams(file_cfg=config['file_cfg'])

    dataset_cfg = config['dataset_val_cfg']
    dataset_cfg['split'] = dataset_split
    dataloader_cfg = config['dataloader_val_cfg']
    dataset, dataloader = build_dataset_and_dataloader(dataset_cfg=dataset_cfg, dataloader_cfg=dataloader_cfg)

    model = build_network(network_cfg=config['network_cfg'],
                          devices=devices, checkpoint_path=eval_params.checkpoint_path)

    evaluator = build_evaluator(evaluator_cfg=config['evaluator_cfg'])

    print('=' * 100)
    print('Start To Validate')

    with torch.no_grad():
        model.eval()

        for step, batch_dict in tqdm(enumerate(dataloader), total=len(dataloader)):
            eval_params.ready_for_next_step(step)

            batch_dict = load_batch_dict_to_gpu(batch_dict, devices[0])

            batch_dict = model(batch_dict, training=False, need_loss=False)

            if save_image:
                dump_image(
                    output_root=config['file_cfg']['evals_dir'],
                    batch_dict=batch_dict,
                    depth_max=dataset_cfg['depth_max'],
                    depth_min=dataset_cfg['depth_min']
                )

            if save_depth:
                dump_depth(
                    output_root=config['file_cfg']['evals_dir'],
                    batch_dict=batch_dict
                )

            if save_point:
                dump_point(
                    output_root=config['file_cfg']['evals_dir'],
                    batch_dict=batch_dict
                )

            evaluator.record(batch_dict)

        metric_dict = evaluator.evaluate()

        eval_params.dump_epoch(metric_dict)
        print(eval_params.format_res())

        # save metrics as csv
        save_metric(
            output_root=config['file_cfg']['evals_dir'],
            metric_dict=eval_params.res_epoch
        )

    print('Evaluating Completed')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('CUDA devices is not avaliable')
        exit()

    main()
