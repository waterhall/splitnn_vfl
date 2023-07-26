import torch
import net_layers
import torch.nn.functional as F

class CCGCNet(net_layers.NetworkedGCModel):
    def __init__(self, D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CCGCNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        y_pred = torch.sigmoid(self.linear1(x))
        return y_pred


class FourLayerNet(net_layers.NetworkedSubModel):
    def __init__(self, D_in, H_1, H_2, H_3, H_4):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FourLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H_1) # 10, 5
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(H_1, H_2) # 5, 6
        self.linear3 = torch.nn.Linear(H_2, H_3) # 6, 7
        self.linear4 = torch.nn.Linear(H_3, H_4) # 7, 8

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = F.dropout(x, p=0.25)
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))

        return x
