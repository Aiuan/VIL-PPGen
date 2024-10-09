import os
import glob
import datetime

from utils import read_config, check_and_create_path


def init_path_for_evaluating(work_dir, checkpoint, note, dataset_split):
    checkpoint_path = os.path.join(work_dir, 'checkpoints', checkpoint)
    assert os.path.exists(checkpoint_path), 'Do not exist checkpoint: {}'.format(checkpoint_path)

    ts_str = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    evals_root = os.path.join(work_dir, 'evals')
    check_and_create_path(evals_root)

    if note is None:
        note = os.path.basename(os.path.realpath(checkpoint_path)).split('.')[0] + '_' + dataset_split
    evals_dir = os.path.join(evals_root, '{}_{}'.format(note, ts_str))
    check_and_create_path(evals_dir)

    config_path = glob.glob(os.path.join(work_dir, '*.yaml'))[0]
    config = read_config(config_path)
    config['file_cfg']['checkpoint_path'] = checkpoint_path
    config['file_cfg']['evals_dir'] = evals_dir

    return config


class EvaluatingParams(object):
    def __init__(self, file_cfg):
        self.checkpoint_path = file_cfg['checkpoint_path']

        self.step = None
        self.cnt_step_this_epoch = 0
        self.res_epoch = dict()
        self.res_step = None

    def ready_for_next_step(self, step):
        self.step = step
        self.cnt_step_this_epoch += 1
        self.res_step = dict()

    def dump_step(self, tensor_dict):
        for key, value in tensor_dict.items():
            self.res_step[key] = value.item()
            if not key in self.res_epoch.keys():
                self.res_epoch[key] = 0
            self.res_epoch[key] += value.item()

    def dump_epoch(self, tensor_dict=None):
        for key, value in self.res_epoch.items():
            self.res_epoch[key] /= self.cnt_step_this_epoch

        if tensor_dict is not None:
            for key, value in tensor_dict.items():
                self.res_epoch[key] = value.item()

    def format_res(self):
        info_res = 'Results:\n'
        for key, value in self.res_epoch.items():
            info_res = info_res + "{} = {:.6f}\n".format(key, value)
        info_res = info_res[:-1]
        return info_res
