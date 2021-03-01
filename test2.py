import torch
import learn2learn as l2l
from copy import deepcopy as dcp


torch.manual_seed(123)

def print_params(params):
    for p in params:
        print(p)


class HypergradTransform(torch.nn.Module):
    """Hypergradient-style per-parameter learning rates"""

    def __init__(self, param, lr=0.01):
        super(HypergradTransform, self).__init__()
        self.lr = lr * torch.ones_like(param, requires_grad=True)
        self.lr = torch.nn.Parameter(self.lr)

    def forward(self, grad):
        print_params(self.parameters())
        return self.lr * grad


model = torch.nn.Linear(2, 1)

metaopt = l2l.optim.LearnableOptimizer(
    model=model,  # We pass the model, not its parameters
    transform=HypergradTransform,  # Any transform could work
    lr=0.1)

opt = torch.optim.Adam(metaopt.parameters(), lr=3e-1)
loss = torch.nn.MSELoss()

data = torch.tensor([3.0, 4.0]), torch.tensor([7.0])
x, y = data


for _ in range(3):
    metaopt.zero_grad(); opt.zero_grad()

    err = loss(model(x), y); print(err, loss, '\n')

    err.backward()

    # metaopt.step()  # Update model parameters

    opt.step()  # Update metaopt parameters

    metaopt.step()  # Update model parameters


