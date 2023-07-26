# We need to serialize PaillierTensor and Torch Tensors

# Code in file nn/two_layer_net_module.py
from syft.frameworks.torch.tensors.interpreters.paillier import PaillierTensor
import numpy as np
import torch
import syft as sy
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
from pe_layers import serialize_paillier, deserialize_paillier, serialize_pub, deserialize_pub
import pickle 
import pytest
from tpe_layers import ThresholdPaillier
import phe 
from numpy import ndarray
from phe.paillier import EncryptedNumber

def decrypt(encrypted, priv_keys):
    shares = []
    for pk in priv_keys:
        shares.append(pk.partial_decrypt(encrypted))
    encrypted.shares = shares
    return priv_keys[0].decrypt(encrypted)

def tpv_decrypt(element, priv_key):
    # Case 1: tensor recursion
    if isinstance(element, torch.Tensor):
        paillier = element.child
        if isinstance(paillier, PaillierTensor):
            return torch.Tensor([tpv_decrypt(subchild, priv_key) for subchild in paillier.child])
        else:
            raise TypeError(type(paillier))
    elif isinstance(element, ndarray):
        return [tpv_decrypt(subelement, priv_key) for subelement in element]
    elif isinstance(element, EncryptedNumber):
        return decrypt(element, priv_key)
    else:
        raise TypeError(type(element))

# hook PyTorch to add extra functionalities like the ability to encrypt torch tensors
hook = sy.TorchHook(th)
pub, pri = sy.keygen(n_length=256)

wrapped_pub = deserialize_pub(serialize_pub(pub))
#ser_paillier = serialize_paillier(x) 
torch.manual_seed(0)

SIZE = 10
N_TESTS = 100
@pytest.mark.parametrize(
    "x_tensor",
    [ torch.rand(SIZE) for i in range(N_TESTS) ],
)
def test_mm(x_tensor):
    x_encrypted = x_tensor.encrypt(protocol="paillier", public_key=pub)
    mm_x = x_encrypted * (torch.eye(1))
    mm_x.child.pubkey = x_encrypted.child.pubkey
    wrapped_paillier = deserialize_paillier(serialize_paillier(mm_x), wrapped_pub)
    _x = wrapped_paillier.decrypt(private_key=pri)
    assert(np.allclose(_x, x_tensor, atol=1e-4))

@pytest.mark.parametrize(
    "x_tensor",
    [ torch.rand(SIZE) for i in range(N_TESTS) ],
)
def test_pickle(x_tensor):
    
    x_encrypted = x_tensor.encrypt(protocol="paillier", public_key=pub)

    _x_tensor = x_encrypted.decrypt(private_key=pri)
    assert(np.allclose(_x_tensor, x_tensor, atol=1e-4))

    _x_tensor = pickle.loads(pickle.dumps(x_tensor))
    assert(np.allclose(_x_tensor, x_tensor, atol=1e-4))

    _x_tensor = deserialize_paillier(pickle.loads(pickle.dumps(serialize_paillier(x_encrypted)))).decrypt(private_key=pri)
    assert(np.allclose(_x_tensor, x_tensor, atol=1e-4))

@pytest.mark.parametrize(
    "a,b,fn",
    [(th.tensor([[float(a)]], dtype=torch.float32),th.tensor([[float(b)]], dtype=torch.float32),fn) for a in np.random.uniform(-100, 100, size=10) for b in np.random.uniform(-100, 100, size=10) for fn in [lambda a,b:a+b, lambda a,b:a-b]],
)
def test_threshold_paillier(a, b, fn):
    tp = ThresholdPaillier(128, 5, 5)

    priv_keys = tp.priv_keys
    pub_key = tp.pub_key
    c = fn(a,b)

    # Serialize pub key
    pub_key = deserialize_pub(serialize_pub(pub_key))
    #pub_key = pickle.loads(pickle.dumps(pub_key))
    priv_keys = pickle.loads(pickle.dumps(priv_keys))
    
    _a = a.encrypt(protocol="paillier", public_key=pub_key)
    _b = b.encrypt(protocol="paillier", public_key=pub_key)
    _c = fn(_a,_b)
    _c.child.pubkey = _a.child.pubkey 
    # Go through ser deser
    _c = deserialize_paillier(pickle.loads(pickle.dumps(serialize_paillier(_c))), pub_key)
    
    decrypted_c = tpv_decrypt(_c, priv_keys)
    
    assert(decrypted_c == c[0][0])
