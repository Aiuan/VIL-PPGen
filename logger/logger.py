from logger.commandline_logger import CMD_Printer
from logger.tensorboard_logger import TensorBoard_Logger
from logger.txt_logger import TXT_Logger


class Logger(object):
    def __init__(self, logs_dir):
        self.logs_dir = logs_dir
        self.cmd_printer = CMD_Printer(timer_on=False)
        self.txt_logger = TXT_Logger(self.logs_dir)
        self.tb_logger_train = TensorBoard_Logger(self.logs_dir, 'train')
        self.tb_logger_val = TensorBoard_Logger(self.logs_dir, 'val')

    def log_text(self, text, cmd_print=True, color=None, txt_log=True):
        if cmd_print:
            self.cmd_printer.print(text, color)
        if txt_log:
            self.txt_logger.log(text)

    def log_scalar(self, mode, tag, value, step):
        if mode == 'train':
            self.tb_logger_train.log_scalar(tag, value, step)
        if mode == 'val':
            self.tb_logger_val.log_scalar(tag, value, step)

    def log_scalars_by_dict(self, data_dict, mode, step):
        for key, value in data_dict.items():
            self.log_scalar(mode=mode, tag=key.replace('_', '/'), value=value, step=step)

    def log_image(self, mode, tag, image, step):
        if mode == 'train':
            self.tb_logger_train.log_image(tag, image, step)
        if mode == 'val':
            self.tb_logger_val.log_image(tag, image, step)
