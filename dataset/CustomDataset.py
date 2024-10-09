import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, config=None, root=None, split='all'):
        assert config is not None or root is not None, 'config or root should be provided'

        self.config = {} if config is None else config

        self.root = self.config.get('root', root)

        self.split = self.config.get('split', split)

        self.items = self.load_items_in_split()

        self.length = len(self.items)

        self.depth_min = self.config.get('depth_min', 0.0)
        self.depth_max = self.config.get('depth_max', 100.0)

        self.point_cloud_range = self.config.get('point_cloud_range', [-51.2, 0.0, -4.8, 51.2, 102.4, 4.8])

    def load_items_from_file(self, file_name):
        file_path = os.path.join(self.root, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                items = f.readlines()
        else:
            # print(f'{file_path} not found.')
            items = []

        return items

    def load_items_in_split(self):
        if self.split == 'trainval':
            items_train = self.load_items_from_file('train.txt')
            items_val = self.load_items_from_file('val.txt')
            items = set(items_train) | set(items_val)
        elif self.split == 'all' or self.split == 'trainvaltest':
            items_train = self.load_items_from_file('train.txt')
            items_val = self.load_items_from_file('val.txt')
            items_test = self.load_items_from_file('test.txt')
            items = set(items_train) | set(items_val) | set(items_test)
        else:
            items = self.load_items_from_file(f'{self.split}.txt')

        items = [item.strip() for item in items]
        items.sort()

        return items

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name = self.items[idx]

        vis_image = cv2.imread(os.path.join(self.root, 'visible', f'{name}.png'))
        # convert BGR to RGB
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        # convert HWC to CHW
        vis_image = np.transpose(vis_image, (2, 0, 1))
        # convert to float32
        vis_image = vis_image.astype(np.float32)
        # normalize to [0.0, 1.0]
        vis_image = vis_image / 255.0

        vis_sparse_depth = np.load(os.path.join(self.root, 'sparse_depth_visible', f'{name}.npy'))
        # convert HW to CHW
        vis_sparse_depth = np.expand_dims(vis_sparse_depth, axis=0)
        # convert to float32
        vis_sparse_depth = vis_sparse_depth.astype(np.float32)
        # limit depth
        vis_sparse_depth = np.clip(vis_sparse_depth, self.depth_min, self.depth_max)

        inf_image = cv2.imread(os.path.join(self.root, 'infrared', f'{name}.png'), cv2.IMREAD_GRAYSCALE)
        # convert HW to CHW
        inf_image = np.expand_dims(inf_image, axis=0)
        # convert to float32
        inf_image = inf_image.astype(np.float32)
        # normalize to [0.0, 1.0]
        inf_image = inf_image / 255.0

        inf_sparse_depth = np.load(os.path.join(self.root, 'sparse_depth_infrared', f'{name}.npy'))
        # convert HW to CHW
        inf_sparse_depth = np.expand_dims(inf_sparse_depth, axis=0)
        # convert to float32
        inf_sparse_depth = inf_sparse_depth.astype(np.float32)
        # limit depth
        inf_sparse_depth = np.clip(inf_sparse_depth, self.depth_min, self.depth_max)

        calib = self.load_json(os.path.join(self.root, 'calib', f'{name}.json'))
        # convert to float32
        for key, value in calib.items():
            calib[key] = np.array(value, dtype=np.float32)
        vis_intrinsic = calib['intrinsic_visible']
        inf_intrinsic = calib['intrinsic_infrared']
        vis_extrinsic = calib['extrinsic_lidar2visible']
        inf_extrinsic = calib['extrinsic_lidar2infrared']

        lid_points = self.load_pcd(os.path.join(self.root, 'sparse_lidar', f'{name}.pcd'))
        # convert to float32
        for key, value in lid_points.items():
            lid_points[key] = value.astype(np.float32)
        # normalize intensity to [0.0, 1.0]
        lid_points['intensity'] = lid_points['intensity'] / 255.0
        # format to [x, y, z, intensity]
        lid_points = np.stack((lid_points['x'], lid_points['y'], lid_points['z'], lid_points['intensity']), axis=1)
        # remove points outside point_cloud_range
        mask = (
                (lid_points[:, 0] >= self.point_cloud_range[0]) & (lid_points[:, 0] <= self.point_cloud_range[3])
                & (lid_points[:, 1] >= self.point_cloud_range[1]) & (lid_points[:, 1] <= self.point_cloud_range[4])
                & (lid_points[:, 2] >= self.point_cloud_range[2]) & (lid_points[:, 2] <= self.point_cloud_range[5])
        )
        lid_points = lid_points[mask, :]

        vis_gt_depth = np.load(os.path.join(self.root, 'depth_visible', f'{name}.npy'))
        # convert HW to CHW
        vis_gt_depth = np.expand_dims(vis_gt_depth, axis=0)
        # convert to float32
        vis_gt_depth = vis_gt_depth.astype(np.float32)
        # limit depth
        vis_gt_depth = np.clip(vis_gt_depth, self.depth_min, self.depth_max)

        inf_gt_depth = np.load(os.path.join(self.root, 'depth_infrared', f'{name}.npy'))
        # convert HW to CHW
        inf_gt_depth = np.expand_dims(inf_gt_depth, axis=0)
        # convert to float32
        inf_gt_depth = inf_gt_depth.astype(np.float32)
        # limit depth
        inf_gt_depth = np.clip(inf_gt_depth, self.depth_min, self.depth_max)

        data_dict = {
            'idx': idx,
            'name': name,
            'vis_image': vis_image,
            'vis_sparse_depth': vis_sparse_depth,
            'inf_image': inf_image,
            'inf_sparse_depth': inf_sparse_depth,
            'vis_intrinsic': vis_intrinsic,
            'inf_intrinsic': inf_intrinsic,
            'vis_extrinsic': vis_extrinsic,
            'inf_extrinsic': inf_extrinsic,
            'lid_points': lid_points,
            'vis_gt_depth': vis_gt_depth,
            'inf_gt_depth': inf_gt_depth
        }

        return data_dict

    def collate_batch(self, data_dict_list):
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
        return batch_dict

    @staticmethod
    def load_json(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data

    @staticmethod
    def load_pcd(path):
        with open(path, "r") as f:
            data = f.readlines()
            keys = data[1].replace('\n', '').split(' ')[1:]
            values = np.genfromtxt(data[10:], delimiter=' ')
            pcd = dict()
            for i, key in enumerate(keys):
                pcd[key] = values[:, i]
        return pcd


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import open3d as o3d
    from torch.utils.data import DataLoader

    matplotlib.use('TkAgg')

    dataset = CustomDataset(root='../data/CustomDataset_ZJU')

    data_dict = dataset[0]

    for key, value in data_dict.items():
        try:
            print(key, value.dtype, value.shape)
        except:
            print(key, type(value), value)

    vmin, vmax = dataset.depth_min, dataset.depth_max
    fig, axs = plt.subplots(nrows=2, ncols=3)
    axs[0, 0].imshow(np.transpose(data_dict['vis_image'], (1, 2, 0)))
    axs[0, 1].imshow(np.transpose(data_dict['vis_sparse_depth'], (1, 2, 0)), cmap='jet', vmin=vmin, vmax=vmax)
    axs[0, 2].imshow(np.transpose(data_dict['vis_gt_depth'], (1, 2, 0)), cmap='jet', vmin=vmin, vmax=vmax)
    axs[1, 0].imshow(np.transpose(data_dict['inf_image'], (1, 2, 0)), cmap='gray')
    axs[1, 1].imshow(np.transpose(data_dict['inf_sparse_depth'], (1, 2, 0)), cmap='jet', vmin=vmin, vmax=vmax)
    axs[1, 2].imshow(np.transpose(data_dict['inf_gt_depth'], (1, 2, 0)), cmap='jet', vmin=vmin, vmax=vmax)
    plt.show()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='lid_points', width=1920, height=1080)
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(data_dict['lid_points'][:, :3])
    cmap_jet = matplotlib.colormaps.get_cmap('jet')
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    colors = cmap_jet(norm(data_dict['lid_points'][:, 3]))
    pointcloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    vis.add_geometry(pointcloud)
    vis.get_render_option().background_color = np.asarray([1, 1, 1])
    vis.get_render_option().point_size = 2
    vis.get_render_option().show_coordinate_frame = False
    vis.run()
    vis.destroy_window()

    # test dataloader with collate_batch
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=dataset.collate_batch
    )
    batch_dict = next(iter(dataloader))

    print(f'{__file__} done.')
