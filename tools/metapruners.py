import torch
from torch.nn.utils import prune
from functools import partial


def get_pruner(prunerstr):
    if prunerstr == 'MP':
        return partial(MP)
    elif prunerstr == 'RP':
        return partial(RP)
    else:
        raise ValueError("No such pruner")


"""
Main funtions
"""


def MP(model, amount):
    wtuples = _weight_tuples(model)
    de_facto_amount = _compute_new_amount(model, amount)
    prune.global_unstructured(wtuples, pruning_method=prune.L1Unstructured, amount=de_facto_amount)


def RP(model, amount):
    wtuples = _weight_tuples(model)
    de_facto_amount = _compute_new_amount(model, amount)
    prune.global_unstructured(wtuples, pruning_method=prune.RandomUnstructured, amount=de_facto_amount)


def prune_reparam(model):
    module_list = _get_modules(model)
    for idx, m in enumerate(module_list):
        prune.identity(m, name="weight")


def get_sparsities(model):
    module_list = _get_modules(model)
    unmasked_list = []
    for m in module_list:
        unmasked_list.append(1.0 * m.weight_mask.sum() / m.weight.numel())
    return torch.FloatTensor(unmasked_list)


def _count_unmasked_weights(model):
    """
    Compute the number of unmasked weight in a model.
    """
    module_list = _get_modules(model)
    unmasked_list = []
    for m in module_list:
        if hasattr(m, 'weight_mask'):
            unmasked_list.append(m.weight_mask.sum())
        else:
            unmasked_list.append(m.weight.numel())
    return torch.FloatTensor(unmasked_list)


def _compute_new_amount(model, amount):
    """
    If we prevent some layers from being pruned, compute new sparsity for the remainder.
    """
    unmaskeds = _count_unmasked_weights(model)
    total_unmaskeds = unmaskeds.sum()
    to_kill = int(total_unmaskeds * amount)
    return float(to_kill / unmaskeds.sum())


def _weight_tuples(model):
    module_list = _get_modules(model)
    layer_list = [(m, 'weight') for m in module_list]
    return tuple(layer_list)


def _get_modules(model):
    module_list = []
    for layer in model.layers:
        if hasattr(layer.linear, 'weight'):
            module_list.append(layer.linear)
    return module_list


def _get_weights(model):
    weight_list = []
    for layer in model.layers:
        if hasattr(layer.linear, 'weight'):
            weight_list.append(layer.linear.weight.clone())
    return weight_list
