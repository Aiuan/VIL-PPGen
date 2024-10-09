import torch
from torch import nn


class PointShuffler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points, others=None):
        batch_size = points[:, 0].max().int().item() + 1
        device = points.device

        points_shuffled = []
        others_shuffled = []
        for i in range(batch_size):
            this_mask = points[:, 0] == i
            shuffle_idx = torch.randperm(this_mask.sum(), device=device)

            this_points = points[this_mask, :]
            points_shuffled.append(this_points[shuffle_idx, :])

            if others is not None:
                this_other = others[this_mask, :]
                others_shuffled.append(this_other[shuffle_idx, :])

        if others is not None:
            return torch.cat(points_shuffled, dim=0), torch.cat(others_shuffled, dim=0)
        else:
            return torch.cat(points_shuffled, dim=0)
