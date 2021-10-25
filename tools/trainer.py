import time
import tqdm

import torch

from torchmeta.utils.gradient_based import gradient_update_parameters as GUP
from utils import AverageMeter


def maml(it, wrapper, loader, opt, num_steps_outer, num_steps_inner=2, alpha=1e-2, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    losses = dict()
    losses['in'] = AverageMeter()
    losses['out'] = AverageMeter()
    check = time.time()

    for step_outer in tqdm.tqdm(range(num_steps_outer)):

        data, _ = next(iter(loader))
        batch_size = data.size(0)
        data = data.cuda()

        # reference: https://github.com/tristandeleu/pytorch-meta
        outer_loss = torch.tensor(0.).cuda()
        for i in range(batch_size):  # act as number of tasks
            params = None
            task_data = data[i].unsqueeze(dim=0)

            for step_inner in range(num_steps_inner):
                loss = wrapper(task_data, params=params)

                wrapper.net.zero_grad()
                params = GUP(wrapper.net, loss, params=params, step_size=alpha)

                ### Log losses ###
                losses['in'].update(loss.item(), 1)

            outer_loss += wrapper(task_data, params=params)

        opt.zero_grad()
        outer_loss.div_(batch_size)
        outer_loss.backward()
        opt.step()

        ### Log losses ###
        losses['out'].update(outer_loss.item(), 1)

        if step_outer % 100 == 0:
            step = step_outer + it * num_steps_outer
            if logger is not None:
                logger.scalar_summary('train/loss_in', losses['in'].average, step)
                logger.scalar_summary('train/loss_out', losses['out'].average, step)

    log_('[Time %.3f] [LossIn %f] [LossOut %f]' %
         (time.time() - check, losses['in'].average, losses['out'].average))
