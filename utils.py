import os
import random
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, fn, ask=True):
        if not os.path.exists("./results/"):
            os.mkdir("./results/")

        logdir = self._make_dir(fn)
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        if len(os.listdir(logdir)) != 0 and ask:
            ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                        "Will you proceed [y/N]? ")
            if ans in ['y', 'Y']:
                shutil.rmtree(logdir)
            else:
                exit(1)

        self.set_dir(logdir)

    def _make_dir(self, fn):
        # today = datetime.today().strftime("%y%m%d")
        logdir = f'./results/{fn}/'
        return logdir

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.writer = SummaryWriter(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        self.writer.add_image(tag, images, step)

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, step, bins='auto')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_name(args):
    fn = f'{args.exp}_{args.model}_{args.id}_{args.data}'

    if args.exp in ['metaprune']:
        fn += f'_{args.pruner}_{args.amount}'

    fn += f'_{args.seed}'
    return fn


def psnr(mse):
    """
    Computes PSNR from MSE.
    """
    return -10.0 * mse.log10()
