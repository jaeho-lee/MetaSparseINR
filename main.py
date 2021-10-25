import os
import argparse

import torch

from exps.meta import exp as metaexp
from exps.metaprune import exp as metapruneexp
from utils import Logger, set_random_seed, file_name


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, help='experiment name to run')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--id', type=str, default='main', help='experiment identifier')

""" Args about Data """
parser.add_argument('--data', type=str, default='celeba')
parser.add_argument('--res', type=int, default=178)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--ds', type=int, default=4)

""" Args about Model """
parser.add_argument('--model', type=str, default='siren')
parser.add_argument('--width', type=int, default=256)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--dim_in', type=int, default=2)
parser.add_argument('--dim_out', type=int, default=3)
parser.add_argument('--w0', type=float, default=200.)
parser.add_argument('--load_path', type=str, default=None, help='path of the model')

""" Args about MAML """
parser.add_argument('--alpha', type=float, default=0.01, help='learning rate for inner loop.')
parser.add_argument('--beta', type=float, default=1e-5, help='learning rate for outer loop.')
parser.add_argument('--num_steps_inner', type=int, default=2, help='Number of inner loops to run')
parser.add_argument('--epochs', type=int, default=150000, help='Number of outer loops to run (per iteration).')

""" Args about Pruning """
parser.add_argument('--pruner', type=str, default='MP', help='pruning method (Supported: MP, RP)')
parser.add_argument('--amount', type=float, default=0.2, help='pruning amount (e.g., per iteration)')
parser.add_argument('--prune_num', type=int, default=20, help='pruning iteration number')

""" Additional args ends here. """
args = parser.parse_args()

""" FIX THE RANDOMNESS """
set_random_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

""" ROOT DIRECTORY """
fn = file_name(args)
logger = Logger(fn)
logger.log(args)
logger.log(f'Log path: {logger.logdir}')

""" RUN THE EXP """
if args.exp == 'metaprune':
    args.load_path = f'./results/meta_baseline_{args.model}_main_{args.data}_{args.seed}/net_meta.pth'
    if not os.path.exists(args.load_path):
        raise ValueError("Train baseline model first")
    metapruneexp(logger.logdir, args, logger)
elif args.exp == 'meta_baseline':
    metaexp(logger.logdir, args, logger)
else:
    raise ValueError("Unknown experiment.")
