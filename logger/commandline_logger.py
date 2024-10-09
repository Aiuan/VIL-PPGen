import datetime

def log(text):
    print(text)


def log_BLUE(text):
    print('\033[0;34;40m{}\033[0m'.format(text))


def log_YELLOW(text):
    print('\033[0;33;40m{}\033[0m'.format(text))


def log_GREEN(text):
    print('\033[0;32;40m{}\033[0m'.format(text))


def log_RED(text):
    print('\033[0;31;40m{}\033[0m'.format(text))


class CMD_Printer(object):
    def __init__(self, timer_on=False):
        self.timer_on = timer_on

    def change_timer_on(self, timer_on):
        self.timer_on = timer_on

    def print(self, text, color=None):
        if self.timer_on:
            ts_str = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            text = '[{}] {}'.format(ts_str, text)

        if color == 'r' or color == 'red':
            log_RED(text)
        elif color == 'g' or color == 'green':
            log_GREEN(text)
        elif color == 'b' or color == 'blue':
            log_BLUE(text)
        elif color == 'y' or color == 'yellow':
            log_YELLOW(text)
        else:
            log(text)

