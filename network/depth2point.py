import torch
from torch import nn


class D2PConverter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, depth, intrinsic, extrinsic, feature=None, return_point2pixel=False):
        '''
        :param depth: (b, 1, h, w)
        :param intrinsic: (b, 3, 3)
        :param extrinsic: (b, 4, 4)
        :param feature: None or (b, c, h, w)
        :param return_point2pixel: bool
        :return:
        '''
        b, _, h, w = depth.shape
        device = depth.device

        v, u = torch.meshgrid(
            [torch.arange(h, device=device), torch.arange(w, device=device)],
            indexing='ij'
        )
        u = u.float().unsqueeze(0).expand(b, -1, -1).unsqueeze(1)
        v = v.float().unsqueeze(0).expand(b, -1, -1).unsqueeze(1)

        coors_uvd = torch.cat((
            u * depth,
            v * depth,
            depth,
            torch.ones_like(depth, device=device)
        ), dim=1)

        project_matrix = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(dim=0).tile((b, 1, 1))
        project_matrix[:, :3, :3] = intrinsic
        project_matrix = torch.matmul(project_matrix, extrinsic)
        project_matrix_inverse = torch.inverse(project_matrix)

        coors_xyz = torch.matmul(
            project_matrix_inverse,
            coors_uvd.reshape((b, 4, -1))
        ).reshape(b, 4, h, w)

        batch_idx = torch.arange(b, device=device).reshape(b, 1, 1, 1).tile(1, 1, h, w)
        points = torch.cat([batch_idx, coors_xyz[:, :3, :, :]], dim=1)
        if feature is not None:
            points = torch.cat([points, feature], dim=1)

        mask = depth > 0
        points_valid = points.permute(0, 2, 3, 1)[mask.squeeze(1), :]

        if return_point2pixel:
            points2pixel = torch.cat([v, u], dim=1).long()
            points2pixel_valid = points2pixel.permute(0, 2, 3, 1)[mask.squeeze(1), :]
            return points_valid, points2pixel_valid
        else:
            return points_valid
