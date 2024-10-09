import torch
from torch import nn
import torch.nn.functional as F


class LossAggregator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def forward(self, loss_dict):
        loss_aggregated_dict = {}
        loss = None
        for loss_name, weight in self.config['weight'].items():
            if loss_name not in loss_dict.keys():
                continue

            loss_aggregated_dict[loss_name] = loss_dict[loss_name].mean()
            if loss is None:
                loss = loss_aggregated_dict[loss_name] * weight
            else:
                loss += loss_aggregated_dict[loss_name] * weight
        loss_aggregated_dict['loss'] = loss

        return loss, loss_aggregated_dict
