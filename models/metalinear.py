import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torchmeta.modules.module import MetaModule


class MetaCustomLinear(nn.Linear, MetaModule):
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.linear(input, params['weight'], bias) if 'weight_orig' not in params.keys() \
            else F.linear(input, params['weight_orig']*self.weight_mask, bias)
