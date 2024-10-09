import os
import datetime
import shutil
import torch

from utils import check_and_create_path


def init_path_for_training(file_cfg, config_path=None):
    ts_str = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    resume_dir = file_cfg['resume_dir']
    if resume_dir is not None:
        if os.path.exists(resume_dir):
            exps_root = os.path.dirname(resume_dir)

            exp_dir = resume_dir

            checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
            if not os.path.exists(checkpoints_dir):
                print('Do not exist checkpoints_dir: {}'.format(checkpoints_dir))
                exit()

            logs_root = os.path.join(exp_dir, 'logs')
            if not os.path.exists(logs_root):
                print('Do not exist logs_root: {}'.format(logs_root))
                exit()

            runs_root = os.path.join(exp_dir, 'runs')
            if not os.path.exists(logs_root):
                print('Do not exist logs_root: {}'.format(logs_root))
                exit()

        else:
            print('Do not exist resume_dir: {}'.format(resume_dir))
            exit()
    else:
        exps_root = file_cfg['exps_root']
        if exps_root is not None:
            check_and_create_path(exps_root)

            note = file_cfg['note']
            exp_dir = os.path.join(exps_root, '{}_{}'.format(note, ts_str))
            check_and_create_path(exp_dir)

            if config_path is not None:
                # backup config.yaml
                shutil.copy(config_path, exp_dir)

            checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
            check_and_create_path(checkpoints_dir)

            logs_root = os.path.join(exp_dir, 'logs')
            check_and_create_path(logs_root)

            runs_root = os.path.join(exp_dir, 'runs')
            check_and_create_path(runs_root)
        else:
            print('resume_dir / exps_root is None')
            exit()

    logs_dir = os.path.join(logs_root, ts_str)
    check_and_create_path(logs_dir)

    # update config
    file_cfg['resume_dir'] = resume_dir
    file_cfg['exps_root'] = exps_root
    file_cfg['exp_dir'] = exp_dir
    file_cfg['checkpoints_dir'] = checkpoints_dir
    file_cfg['logs_root'] = logs_root
    file_cfg['logs_dir'] = logs_dir
    file_cfg['runs_root'] = runs_root

    return file_cfg


class TrainingParams(object):
    def __init__(self, file_cfg, epoch_max):

        self.resume_dir = file_cfg['resume_dir']
        if self.resume_dir is not None:
            self.resume_checkpoint = 'latest.pth' \
                if file_cfg['resume_checkpoint'] is None else file_cfg['resume_checkpoint']
            self.checkpoint_path = os.path.join(self.resume_dir, 'checkpoints', self.resume_checkpoint)
            self.checkpoint = torch.load(self.checkpoint_path)
            self.epoch_previous = self.checkpoint['epoch']

            self.info_init = 'Recover training from {}'.format(self.checkpoint_path)
        else:
            self.resume_checkpoint = None
            self.checkpoint_path = None
            self.checkpoint = None
            self.epoch_previous = -1

            self.info_init = 'Training from scratch'

        self.epoch_tostart = self.epoch_previous + 1
        self.epoch_max = epoch_max

        self.epoch = None
        self.step = None
        self.cnt_step_this_epoch = None
        self.res_epoch = None
        self.res_step = None

        self.epoch_val = None
        self.step_val = None
        self.cnt_step_this_epoch_val = None
        self.res_epoch_val = None
        self.res_step_val = None

    def ready_for_next_epoch(self, training, epoch):
        if training:
            self.epoch = epoch
            self.res_epoch = dict()
            self.cnt_step_this_epoch = 0
        else:
            self.epoch_val = epoch
            self.res_epoch_val = dict()
            self.cnt_step_this_epoch_val = 0

    def ready_for_next_step(self, training, step):
        if training:
            self.step = step
            self.cnt_step_this_epoch += 1
            self.res_step = dict()
        else:
            self.step_val = step
            self.cnt_step_this_epoch_val += 1
            self.res_step_val = dict()

    def dump_step(self, training, tensor_dict):
        for key, value in tensor_dict.items():
            if training:
                self.res_step[key] = value.item()
                if not key in self.res_epoch.keys():
                    self.res_epoch[key] = 0
                self.res_epoch[key] += value.item()
            else:
                self.res_step_val[key] = value.item()
                if not key in self.res_epoch_val.keys():
                    self.res_epoch_val[key] = 0
                self.res_epoch_val[key] += value.item()

    def dump_epoch(self, training, tensor_dict=None):
        if training:
            for key, value in self.res_epoch.items():
                self.res_epoch[key] /= self.cnt_step_this_epoch
        else:
            for key, value in self.res_epoch_val.items():
                self.res_epoch_val[key] /= self.cnt_step_this_epoch_val

        if tensor_dict is not None:
            for key, value in tensor_dict.items():
                if training:
                    self.res_epoch[key] = value.item()
                else:
                    self.res_epoch_val[key] = value.item()

    def format_res(self, training):
        info_res = 'Results:\n'
        if training:
            res = self.res_epoch
        else:
            res = self.res_epoch_val
        for key, value in res.items():
            info_res = info_res + "{} = {:.6f}\n".format(key, value)
        info_res = info_res[:-1]
        return info_res

    def is_epoch(self, period_epoch, force_first=True, force_last=True):
        if period_epoch > 0:
            if (self.epoch + 1) % period_epoch == 0:
                return True
            elif force_first and self.epoch == self.epoch_tostart:
                return True
            elif force_last and self.epoch == self.epoch_max - 1:
                return True
            else:
                return False
        else:
            return False
