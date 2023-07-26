import random
import syft as sy

import torch
import torch.nn.functional as F
from torch.autograd import Variable as Var
import json

# https://github.com/OpenMined/PySyft/issues/2901

hook = sy.TorchHook(torch, verbose=True)

me = hook.local_worker
me.is_client_worker = False

bob = sy.VirtualWorker(id="bob", hook=hook, is_client_worker=False)
alice = sy.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
james = sy.VirtualWorker(id="james", hook=hook, is_client_worker=False)

x = torch.LongTensor([[3,-2],[-3,-4]])
x = x.share(bob, alice)

y = torch.LongTensor([[5,6],[7,8]])
y = y.share(bob, alice)

print(x.mm(y).get())
