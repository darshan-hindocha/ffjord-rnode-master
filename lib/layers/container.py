import torch.nn as nn


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, reg_states=tuple(), reverse=False, inds=None,jacfree=False):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        for i in inds:
            if jacfree and self.chain[i].cnflayer==True:
                self.chain[i].jacfree=True
                x, logpx, reg_states = self.chain[i](x, logpx, reg_states, reverse=reverse)
            else: 
                x, logpx, reg_states = self.chain[i](x, logpx, reg_states, reverse=reverse)
        return x, logpx, reg_states
