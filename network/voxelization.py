import torch
from torch import nn
from spconv.pytorch.utils import PointToVoxel


class Voxelizer(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.device = device

        self.voxel_size = config['voxel_size']
        self.point_cloud_range = config['point_cloud_range']
        self.num_point_features = config['num_point_features']
        self.max_num_voxels = config['max_num_voxels']
        self.max_num_points_per_voxel = config['max_num_points_per_voxel']

        self.voxel_generator = PointToVoxel(
            vsize_xyz=self.voxel_size,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=self.num_point_features,
            max_num_voxels=self.max_num_voxels,
            max_num_points_per_voxel=self.max_num_points_per_voxel,
            device=self.device
        )

    def forward(self, points, return_point2voxel=False):
        batch_size = points[:, 0].max().int().item() + 1

        voxels = []
        voxel_coordinates = []
        num_points_per_voxel = []
        pc_voxel_ids = []

        for i in range(batch_size):
            this_points = points[points[:, 0] == i, 1:]

            if return_point2voxel:
                this_voxels, this_voxel_coordinates, this_num_points_per_voxel, this_pc_voxel_ids = \
                    self.voxel_generator.generate_voxel_with_id(this_points)

                pc_voxel_ids.append(this_pc_voxel_ids + sum([len(item) for item in voxels]))

            else:
                this_voxels, this_voxel_coordinates, this_num_points_per_voxel = self.voxel_generator(this_points)

            this_voxel_coordinates = torch.cat(
                (
                    i * torch.ones((this_voxel_coordinates.shape[0], 1),
                                   dtype=this_voxel_coordinates.dtype, device=this_voxel_coordinates.device),
                    this_voxel_coordinates
                ),
                dim=-1
            )

            voxels.append(this_voxels)
            voxel_coordinates.append(this_voxel_coordinates)
            num_points_per_voxel.append(this_num_points_per_voxel)

        voxels = torch.cat(voxels, dim=0)
        voxel_coordinates = torch.cat(voxel_coordinates, dim=0)
        num_points_per_voxel = torch.cat(num_points_per_voxel, dim=0)

        if return_point2voxel:
            pc_voxel_ids = torch.cat(pc_voxel_ids, dim=0)
            return voxels, voxel_coordinates, num_points_per_voxel, pc_voxel_ids
        else:
            return voxels, voxel_coordinates, num_points_per_voxel
