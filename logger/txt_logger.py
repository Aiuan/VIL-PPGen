import logging
import datetime

class TXT_Logger(object):
    def __init__(self, logs_dir):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        filename = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        filepath = '{}/{}.txt'.format(logs_dir, filename)
        file_handler = logging.FileHandler(filepath)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(self, text):
        self.logger.info(text)

