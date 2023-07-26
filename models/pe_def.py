import torch
import pe_layers

class GCNet(pe_layers.PaillierEncryptedGCModel):
    def __init__(self, D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super().__init__()
        self.linear1 = pe_layers.PaillierLinearCut(D_in, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        y_pred = self.linear1(x)
        return y_pred

class TwoLayerNet(pe_layers.PaillierEncryptedSubModel):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(H, D_out)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h1 = self.relu1(self.linear1(x))
        return self.relu2(self.linear2(h1))