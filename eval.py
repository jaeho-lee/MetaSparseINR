import argparse

import torch

from evals.eval import test, eval_imp, eval_oneshot
from tools.dataloader import get_loaders
from utils import set_random_seed


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, help='experiment name to run')
parser.add_argument('--opt_type', type=str, help='experiment name to run')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--id', type=str, default='trial', help='experiment identifier')

""" Args about Data """
parser.add_argument('--data', type=str, default='celeba')
parser.add_argument('--res', type=int, default=178)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--ds', type=int, default=4)
parser.add_argument('--use_train_set', action='store_true',
                    help='set true for evaluating meta inr with train set')

""" Args about Model """
parser.add_argument('--model', type=str, default='siren')
parser.add_argument('--width', type=int, default=256)  # number of hidden layer
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--dim_in', type=int, default=2)
parser.add_argument('--dim_out', type=int, default=3)
parser.add_argument('--w0', type=float, default=200.)
parser.add_argument('--amount', type=float, default=0.2, help='pruning amount (e.g., per iteration)')
parser.add_argument('--pruner', type=str, default='MP', help='pruning method (Supported: MP, RP)')
parser.add_argument('--prune_num', type=int, default=20, help='pruning iteration number')

""" Args about INR fitting """
parser.add_argument('--alpha', type=float, default=0.01, help='learning rate for inner loop.')
parser.add_argument('--num_steps_inner', type=int, default=2, help='Number of inner loops to run')

""" Args about Load """
parser.add_argument('--load_path', type=str, default=None, help='path of the model')
parser.add_argument('--no_init', default=False, help='train inr w.o meta initialization')

parser.add_argument('--one_class_idx', help='None: multi-class, Not None: one-class', default=None, type=int)
parser.add_argument('--num_test', help='number of test set to evaluate', default=None, type=int)

""" Additional args ends here. """
args = parser.parse_args()

""" FIX THE RANDOMNESS """
set_random_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

""" RUN THE EXP """
if args.data in ['celeba', 'imagenette']:
    args.num_test = 100
if args.data == 'sdf':
    args.num_test = 250

# Default type
if args.opt_type == 'two_step_sgd':
    args.optim = 'sgd'
    args.num_steps_inner = 2
    args.alpha = 1e-2
elif args.opt_type == 'default':
    args.optim = 'adam'
    args.num_steps_inner = 100
    args.alpha = 1e-4
else:
    raise NotImplementedError()


if args.exp == 'base':
    args.load_path = f'./results/meta_baseline_{args.model}_main_{args.data}_{args.seed}/net_meta.pth'
    test(args)
    result_list = [[], []]

elif args.exp == 'prune':
    result_list = [[], []]
    args.load_path = f'./results/meta_baseline_{args.model}_main_{args.data}_{args.seed}/net_meta.pth'
    mse, psnr = test(args)
    result_list[0].append(mse)
    result_list[1].append(psnr)

    base_load_path = f'./results/metaprune_{args.model}_main_{args.data}_{args.pruner}_{args.amount}'
    base_load_path += f'_{args.seed}/'

    for i in range(args.prune_num):
        args.load_path = f'{base_load_path}net_{i}.pth'
        mse, psnr = test(args, prune=True)
        result_list[0].append(mse)
        result_list[1].append(psnr)

elif args.exp == 'dense_narrow':
    base_load_path = f'./results/meta_baseline_{args.model}_width_'

    width_list = [230, 206, 184, 164, 148, 132, 118, 106,
                  94, 84, 76, 68, 60, 54, 48, 44, 38, 34, 32, 28]

    result_list = [[], []]
    for i in range(len(width_list)):
        suffix = f"_{args.data}_{args.seed}/net_meta.pth"
        args.load_path = f'{base_load_path}{width_list[i]}{suffix}'
        args.width = width_list[i]

        mse, psnr = test(args)
        result_list[0].append(mse)
        result_list[1].append(psnr)

elif args.exp == 'one_shot':
    assert args.opt_type == 'default'

    _, test_loader = get_loaders(args.data, args.res, args.batch_size, use_train_set=args.use_train_set)

    result_list = [[], []]
    args.load_path = f'./results/meta_baseline_{args.model}_main_{args.data}_{args.seed}/net_meta.pth'
    mse, psnr = test(args)
    result_list[0].append(mse)
    result_list[1].append(psnr)

    for i in range(args.prune_num):
        mse, psnr = eval_oneshot(args, test_loader, iter=i)
        result_list[0].append(mse)
        result_list[1].append(psnr)

elif args.exp == 'imp':
    assert args.opt_type == 'default'

    _, test_loader = get_loaders(args.data, args.res, args.batch_size, use_train_set=args.use_train_set)

    result_list = [[], []]
    args.load_path = f'./results/meta_baseline_{args.model}_main_{args.data}_{args.seed}/net_meta.pth'
    mse, psnr = test(args)
    result_list[0].append(mse)
    result_list[1].append(psnr)

    for i in range(args.prune_num):
        mse, psnr = eval_imp(args, test_loader, iter=i)
        result_list[0].append(mse)
        result_list[1].append(psnr)

elif args.exp == 'scratch':
    args.no_init = True

    width_list = [256, 230, 206, 184, 164, 148, 132, 118, 106,
                  94, 84, 76, 68, 60, 54, 48, 44, 38, 34, 32, 28]

    result_list = [[], []]
    for i in range(len(width_list)):
        args.width = width_list[i]
        mse, psnr = test(args)
        result_list[0].append(mse)
        result_list[1].append(psnr)

else:
    raise NotImplementedError()

""" Printing results """
mse = map('{:.6f}'.format, result_list[0])
psnr = map('{:.6f}'.format, result_list[1])
print('MSE result')
print('\t'.join(mse))
print('PSNR result')
print('\t'.join(psnr))
