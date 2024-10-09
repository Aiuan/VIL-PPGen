import os
from torch.utils.tensorboard import SummaryWriter

class TensorBoard_Logger(object):
    def __init__(self, logs_dir, mode):
        if mode == 'train':
            os.mkdir(os.path.join(logs_dir, 'train'))
            self.writer = SummaryWriter(os.path.join(logs_dir, 'train'))
        elif mode == 'val':
            os.mkdir(os.path.join(logs_dir, 'val'))
            self.writer = SummaryWriter(os.path.join(logs_dir, 'val'))
        else:
            print('Unknown mode={}'.format(mode))
            exit()

    def __del__(self):
        self.writer.close()

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def log_image(self, tag, image, step):
        self.writer.add_image(tag, image, step)
        self.writer.flush()

