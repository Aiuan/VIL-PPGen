import torch


def calculate_root_mean_squared_error(preds, targets):
    '''
    :param preds: (N)
    :param targets: (N)
    :return: RMSE
    '''
    return torch.sqrt(torch.mean(torch.pow(preds - targets, 2)))


def calculate_mean_absolute_error(preds, targets):
    '''
    :param preds: (N)
    :param targets: (N)
    :return: MAE
    '''
    return torch.mean(torch.abs(preds - targets))


def calculate_mean_absolute_relative_error(preds, targets):
    '''
    :param preds: (N)
    :param targets: (N)
    :return: REL
    '''
    return torch.mean(torch.abs(preds - targets) / targets)


class CustomEvaluator(object):
    def __init__(self, config):
        self.config = config

        self.gt_depth_key = config.get('gt_depth_key', 'gt_depth')
        self.pred_depth_key = config.get('pred_depth_key', 'pred_depth')

        self.depth_grids = config.get('depth_grids', None)

        self.data = None
        self.ready_for_next_evaluate()

    def ready_for_next_evaluate(self):
        self.data = {}

    @torch.no_grad()
    def record(self, batch_dict):
        assert self.gt_depth_key in batch_dict.keys() and self.pred_depth_key in batch_dict.keys()

        if self.gt_depth_key not in self.data.keys():
            self.data['gt_depth'] = []

        if self.pred_depth_key not in self.data.keys():
            self.data['pred_depth'] = []

        gt_depth = batch_dict[self.gt_depth_key]
        pred_depth = batch_dict[self.pred_depth_key]
        mask = gt_depth > 0
        self.data['gt_depth'].append(gt_depth[mask])
        self.data['pred_depth'].append(pred_depth[mask])

    @torch.no_grad()
    def evaluate(self, metric_dict=None):
        if metric_dict is None:
            metric_dict = {}

        pred_depth = torch.cat(self.data['pred_depth'], dim=0)
        gt_depth = torch.cat(self.data['gt_depth'], dim=0)

        metric_dict['metric_RMSE'] = calculate_root_mean_squared_error(preds=pred_depth, targets=gt_depth)
        metric_dict['metric_MAE'] = calculate_mean_absolute_error(preds=pred_depth, targets=gt_depth)
        metric_dict['metric_REL'] = calculate_mean_absolute_relative_error(preds=pred_depth, targets=gt_depth)

        if self.depth_grids is not None:
            for name, (dmin, dmax) in self.depth_grids.items():
                mask = torch.logical_and(gt_depth >= dmin, gt_depth < dmax)
                metric_dict['metric_RMSE_' + name] = calculate_root_mean_squared_error(
                    preds=pred_depth[mask], targets=gt_depth[mask])
                metric_dict['metric_MAE_' + name] = calculate_mean_absolute_error(
                    preds=pred_depth[mask], targets=gt_depth[mask])
                metric_dict['metric_REL_' + name] = calculate_mean_absolute_relative_error(
                    preds=pred_depth[mask], targets=gt_depth[mask])

        self.ready_for_next_evaluate()

        return metric_dict


if __name__ == '__main__':
    device = 'cuda:0'
    batch_size = 4
    height, width = 256, 512
    depth_min, depth_max = 0.0, 100.0
    n_steps = 1000
    evaluator_cfg = {
        'gt_depth_key': 'gt_depth',
        'pred_depth_key': 'pred_depth',
        'depth_grids': {
            'near': [0.0, 30.0],
            'mid': [30.0, 60.0],
            'far': [60.0, 100.0]
        }
    }


    def simulate_data():
        target = torch.rand((batch_size, 1, height, width), dtype=torch.float32, device=device) * \
                 (depth_max - depth_min) + depth_min
        pred = torch.clamp(
            target + torch.rand((batch_size, 1, height, width), dtype=torch.float32, device=device),
            min=depth_min, max=depth_max
        )
        return pred, target


    evaluator = CustomEvaluator(config=evaluator_cfg)

    for i in range(n_steps):
        batch_dict = {}
        batch_dict['pred_depth'], batch_dict['gt_depth'] = simulate_data()

        evaluator.record(batch_dict)

    metric_dict = evaluator.evaluate()

    print(f'{__file__} done.')
