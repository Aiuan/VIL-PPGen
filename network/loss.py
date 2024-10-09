import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothL1DepthLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, preds, targets):
        '''
        :param preds: (B, 1, H, W)
        :param targets: (B, 1, H, W)
        :return: loss: (B)
        '''
        weights = (targets > 0).float()
        normalizer = weights.sum(dim=[1, 2, 3], keepdim=True)
        weights = weights / normalizer.clamp(min=1.0)

        losses = F.smooth_l1_loss(
            input=preds,
            target=targets,
            reduction='none',
            beta=self.beta
        )
        losses = losses * weights
        losses = losses.sum(dim=[1, 2, 3])

        return losses


class LossAggregator(nn.Module):
    def __init__(self, weight):
        super().__init__()

        self.weight = weight

    def forward(self, loss_dict):
        loss_aggregated_dict = {}
        loss = None
        for loss_name, weight in self.weight.items():
            if loss_name not in loss_dict.keys():
                continue

            loss_aggregated_dict[loss_name] = loss_dict[loss_name].mean()
            if loss is None:
                loss = loss_aggregated_dict[loss_name] * weight
            else:
                loss += loss_aggregated_dict[loss_name] * weight
        loss_aggregated_dict['loss'] = loss

        return loss, loss_aggregated_dict
