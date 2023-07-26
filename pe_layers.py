import torch
import logging
import syft as sy
import torch as th
import threading
import asyncio
import websockets
from pickle import dumps, loads
import time
import numpy as np
import net_layers
import torch.nn as nn

from phe.paillier import PaillierPublicKey
from syft.frameworks.torch.tensors.interpreters.paillier import PaillierTensor
from syft.serde.serde import deserialize
from phe.paillier import EncryptedNumber
from phe.paillier import PaillierPublicKey
from numpy import ndarray
from numpy import array

# Transfer 10MiB
MAX_SIZE = 1*1024**3
RW_BUFFER = 2**24
MAX_Q = 1024
PING_TIMEOUT = 240
CLOSE_TIMEOUT = 200

def serialize_paillier(element):
    # Case 1: tensor recursion
    if isinstance(element, torch.Tensor):
        paillier = element.child
        if isinstance(paillier, PaillierTensor):
            child = [serialize_paillier(subchild) for subchild in paillier.child]
            return {'n': paillier.pubkey.n, 'values': child} # in PaillierPublicKey g = n + 1
        else:
            raise TypeError(type(paillier))

    # Case 2: ndarray recursion
    elif isinstance(element, ndarray):
        return [serialize_paillier(subelement) for subelement in element]

    # Case 3: EncryptedNumber serialization
    elif isinstance(element, EncryptedNumber):
        return (str(element.ciphertext()), str(element.exponent))

    # Case 4: Unknown type
    else:
        raise TypeError(type(element))

def deserialize_paillier(struct, pub=None):
    # Case 1: dict recursion
    if isinstance(struct, dict):

        if pub:
            assert(int(struct['n']) == pub.n)
        pub = PaillierPublicKey(n=int(struct['n']))
        child = [deserialize_paillier(substruct, pub) for substruct in struct['values']]
        # Building Paillier Tensor
        tensor = PaillierTensor()
        tensor.child = array(child)
        tensor.pubkey = pub
        return tensor.wrap()

    # Case 2: list recursion
    elif isinstance(struct, list):
        return [deserialize_paillier(substruct, pub) for substruct in struct]

    # Case 3: Tuple deserialization
    elif isinstance(struct, tuple):
        return EncryptedNumber(pub, int(struct[0]), int(struct[1]))

    # Case 4: Unknown type
    else:
        raise TypeError(type(struct))
def serialize_pub(pub):
    return {'n': pub.n}
def deserialize_pub(struct):
    return PaillierPublicKey(n=int(struct['n']))

# https://stackoverflow.com/questions/54421029/python-websockets-how-to-setup-connect-timeout
hook = sy.TorchHook(th)

class PaillierEncryptedSubModel(net_layers.NetworkedSubModel):
    def setup(self, config_file, gc_conn, tp_conn):
        super().setup(config_file, gc_conn, tp_conn)

        logging.info("Exchanging keys")
        # This will go into TP

        time.sleep(10)

        ssl_context, hostname, port = tp_conn

        async def get_pub():
            uri = f"wss://{hostname}:{port}"
            async with websockets.connect(uri, ssl=ssl_context, max_size=MAX_SIZE, 
                read_limit=RW_BUFFER, write_limit=RW_BUFFER, max_queue=MAX_Q, ping_timeout=PING_TIMEOUT) as websocket:
                
                pub_key = await websocket.recv()
                pub_key = deserialize_pub(loads(pub_key))

            return pub_key

        self.pub = asyncio.get_event_loop().run_until_complete(get_pub())
    def __call__(self, x):
        output = super().__call__(x)
        return output
    def serialize(self, tensor):
        return serialize_paillier(tensor.encrypt("paillier", public_key=self.pub))

class PaillierEncryptedGCModel(net_layers.NetworkedGCModel):
    def setup(self, config_file, gc_conn, tp_conn):
        super().setup(config_file, gc_conn, tp_conn)
        PaillierLinearFunction.tp_conn = tp_conn
        # Generate & Distribute keys
    def __call__(self, output_enc):
        return super().__call__(output_enc)
    def deserialize(self, ser_tensor):
        return deserialize_paillier(ser_tensor)
    def cat(self, tensor_lists):
        concated_tensor = PaillierTensor()
        concated_tensor.child = np.concatenate(np.array([ ea_t.child for ea_t in tensor_lists]), axis=1)
        concated_tensor.pubkey = tensor_lists[0].child.pubkey
        concated_tensor = concated_tensor.wrap()

        # Test serialize_paillier

        return concated_tensor
    def grad(self, input_tensor):
        return PaillierLinearFunction.grad_input.clone()
        
class PaillierLinearFunction(th.autograd.Function):
    tp_conn = None
    grad_input = None
    @staticmethod
    async def decrypt_using_tp(enc_paillier_tensor):

        ssl_context, hostname, port = PaillierLinearFunction.tp_conn
        assert(port == 8800)

        logging.info(f"Requesting decryption from {PaillierLinearFunction.tp_conn}")
        
        uri = f"wss://{hostname}:{port}"
        async with websockets.connect(
        uri, ssl=ssl_context, max_size=MAX_SIZE, 
        read_limit=RW_BUFFER, write_limit=RW_BUFFER, max_queue=MAX_Q,
        close_timeout=CLOSE_TIMEOUT, ping_timeout=PING_TIMEOUT
        ) as websocket:
            await websocket.send(dumps(serialize_paillier(enc_paillier_tensor)))
            decrypted_paillier_tensor = await websocket.recv()
            decrypted_paillier_tensor = loads(decrypted_paillier_tensor)

        return decrypted_paillier_tensor

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(weight, bias)
        ctx._input = input

        # Performed in parallel in SA
        # SA also sends out the output to the individual nodes
        output = input.mm(weight.t())
        output.child.pubkey = input.child.pubkey

        # GC asks nodes for decryption from TP
        output = asyncio.get_event_loop().run_until_complete(PaillierLinearFunction.decrypt_using_tp(output))
        #input = asyncio.get_event_loop().run_until_complete(PaillierLinearFunction.decrypt_using_tp(input))
        #output = input.mm(weight.t())

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output.clone()

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        weight, bias = ctx.saved_tensors
        input = ctx._input
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
            PaillierLinearFunction.grad_input = grad_input
            grad_input = None
        if ctx.needs_input_grad[1]:
            # Performed in parallel in SA
            # SA also sends out the output to the individual nodes
            grad_weight = grad_output.t().mm(input)
            grad_weight.child.pubkey = input.child.pubkey

            # GC asks nodes for decryption from TP
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            grad_weight = asyncio.get_event_loop().run_until_complete(PaillierLinearFunction.decrypt_using_tp(grad_weight))
            #input = asyncio.get_event_loop().run_until_complete(PaillierLinearFunction.decrypt_using_tp(input))
            #grad_weight = grad_output.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class PaillierLinearCut(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        out = PaillierLinearFunction.apply(input, self.weight, self.bias)
        # See the autograd section for explanation of what happens here.
        return out
    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

class PaillierTP(object):
    def __init__(self, n_length, **kwargs):
        self.n_length = n_length
    def setup(self, config_file, gc_conn, tp_conn):
        self.pub, self.pri = sy.keygen(n_length=self.n_length)
    def distribute_keys(self, config_file, target_conn):
        logging.info(f"Distributing keys, listening at {target_conn}")
        # Ignore hostname as we are serving
        ssl_context, _, port = target_conn

        def serve_submodels(ssl_context, port):
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            stop = loop.create_future()

            async def serve_key(websocket, path):
                logging.info(f"[Key] Subm connected to {port}")
                await websocket.send(dumps(serialize_pub(self.pub)))
                logging.info(f"send key {port} > {self.pub}")
                stop.set_result(None)

            async def comms_server(stop):
                async with websockets.serve(serve_key, '0.0.0.0', port, ssl=ssl_context, max_size=MAX_SIZE, 
                read_limit=RW_BUFFER, write_limit=RW_BUFFER, max_queue=MAX_Q, ping_timeout=PING_TIMEOUT):
                    await stop

            a = loop.run_until_complete(comms_server(stop))

        serve_submodels(ssl_context, port)
    def decrypt(self, config_file, target_conn):
        logging.info(f"Started Decryption Service, listening at {target_conn}")

        # Ignore hostname as we are serving
        ssl_context, _, port = target_conn

        def serve_submodels(ssl_context, port):
            training_iters = config_file['training_iters']
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # twice for every train and once for predict
            limit = 3*training_iters
            
            stop = loop.create_future()
            # TODO Hack, fix for production
            stop.count = 0

            async def decrypt_one(websocket, path):
                enc_mm_out = await websocket.recv()
                enc_mm_out = deserialize_paillier(loads(enc_mm_out))
                mm_out = enc_mm_out.decrypt(private_key=self.pri)
                await websocket.send(dumps(mm_out))
                logging.info(f"Sent decrypted tensor #[{stop.count}].")

                stop.count += 1
                if stop.count == limit:
                    stop.set_result(None)

            async def decrypt_server(stop):
                async with websockets.serve(decrypt_one, '0.0.0.0', port, ssl=ssl_context, max_size=MAX_SIZE, 
                read_limit=RW_BUFFER, write_limit=RW_BUFFER, max_queue=MAX_Q, ping_timeout=PING_TIMEOUT):
                    await stop

            a = loop.run_until_complete(decrypt_server(stop))

        serve_submodels(ssl_context, port)
