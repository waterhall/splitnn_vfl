import torch
import logging
import syft as sy
import torch as th

# https://stackoverflow.com/questions/54421029/python-websockets-how-to-setup-connect-timeout
hook = sy.TorchHook(th)

class NetworkedSubModel(torch.nn.Module):
    # Do any initalization here
    def setup(self, config_file, gc_conn, tp_conn):
        logging.info("Default Networked Model")
    # Handling the seialization of your custom object between the models.
    def serialize(self, tensor):
        return tensor

class NetworkedGCModel(torch.nn.Module):
    def setup(self, config_file, gc_conn, tp_conn):
        logging.info("Default Networked Model")
    def deserialize(self, ser_tensor):
        return ser_tensor
    def cat(self, tensor_lists):
        return torch.cat(tensor_lists, 1).detach().clone()
    def grad(self, input_tensor):
        return input_tensor.grad.detach().clone()

class UnsecureTP(object):
    def __init__(self, **kwargs):
        pass
    def setup(self, config_file, gc_conn, tp_conn):
        logging.info(f"No Setup")
    def distribute_keys(self, config_file, target_conn):
        logging.info(f"No need to distribute keys")
    def decrypt(self, config_file, target_conn):
        logging.info(f"No need to decrypt")