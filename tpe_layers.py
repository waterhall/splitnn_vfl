# Credits: https://coderzcolumn.com/tutorials/python/threshold-paillier

from phe import paillier
import numpy as np
import random
import sympy
import math
import logging 
import asyncio
import websockets
from pickle import dumps, loads
from pe_layers import serialize_paillier, deserialize_paillier, serialize_pub, deserialize_pub
import torch
import torch as th
import time
from syft.frameworks.torch.tensors.interpreters.paillier import PaillierTensor
from numpy import ndarray
from phe.paillier import EncryptedNumber
import net_layers
import torch.nn as nn

# Transfer 10MiB
MAX_SIZE = 1*1024**3
RW_BUFFER = 2**24
MAX_Q = 1024
PING_TIMEOUT = 240
CLOSE_TIMEOUT = 200

# https://www.brics.dk/RS/00/45/BRICS-RS-00-45.pdf
class ThresholdPaillier(object):
    def __init__(self, size_of_n, n_shares=5, threshold=2):
        #size_of_n = 1024
        pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
        self.p1 = priv.p
        self.q1 = priv.q

        while sympy.primetest.isprime(2*self.p1 +1)!= True:
            pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
            self.p1 = priv.p
        while sympy.primetest.isprime(2*self.q1 +1)!= True:
            pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
            self.q1 = priv.q

        self.p = (2*self.p1) + 1
        self.q = (2*self.q1) + 1
        
        #print(sympy.primetest.isprime(self.p), sympy.primetest.isprime(self.q), sympy.primetest.isprime(self.p1), sympy.primetest.isprime(self.q1))
        
        self.n = self.p * self.q
        self.s = 1
        self.ns = pow(self.n, self.s)
        self.nSPlusOne = pow(self.n,self.s+1)
        self.nPlusOne = self.n + 1
        self.nSquare = self.n*self.n

        self.m = self.p1 * self.q1
        self.nm = self.n*self.m
        self.l = n_shares # Number of shares of private key
        self.w = threshold # The minimum of decryption servers needed to make a correct decryption.
        self.delta = self.factorial(self.l)
        self.rnd = random.randint(1,1e50)
        self.combineSharesConstant = sympy.mod_inverse((4*self.delta*self.delta)%self.n, self.n)
        self.d = self.m * sympy.mod_inverse(self.m, self.n)

        self.r = random.randint(1,self. p) ## Need to change upper limit from p to one in paper
        while math.gcd(self.r,self.n) != 1:
            self.r = random.randint(0, self.p)
        self.v = (self.r*self.r) % self.nSquare

        # Public key
        self.pub_key = ThresholdPaillierPublicKey(self.n, self.nSPlusOne, self.r, self.ns, self.w,\
                                                 self.delta, self.combineSharesConstant)
        # self.pub_key = paillier.PaillierPublicKey(self.n)

        # We fake the initalization over the network.
        self.priv_keys = []
        for i in range(self.l):
            self.priv_keys.append(ThresholdPaillierPrivateKey(self.pub_key, self.n, self.l, self.combineSharesConstant, self.w, \
                                            self.v, i, self.r, self.delta, self.nSPlusOne, self.nm, self.d))

        # multi-party computation
        for priv_key in self.priv_keys:
            priv_key.init_priv_key(self.priv_keys)

    def factorial(self, n):
        fact = 1
        for i in range(1,n+1):
            fact *= i
        return fact

    def computeGCD(self, x, y):
       while(y):
           x, y = y, x % y
       return x

class PartialShare(object):
    def __init__(self, share, server_id):
        self.share = share
        self.server_id =server_id

# This is the parital key
class ThresholdPaillierPrivateKey(paillier.PaillierPrivateKey):
    def __init__(self, public_key, n, l,combineSharesConstant, w, v, i, r, delta, nSPlusOne, nm, d):
        self.public_key = public_key
        self.n = n
        self.l = l
        self.combineSharesConstant = combineSharesConstant
        self.w = w
        self.v = v
        self.si = 0
        self.i = i
        self.server_id = i+1
        self.r = r
        self.delta = delta
        self.nSPlusOne = nSPlusOne
        self.nm = nm
        self.d = d
        if self.i == 0:
            self.ais = self.d
        else:
            self.ais = random.randint(0,self.nm-1)
    def partial_decrypt(self, c):
        if not hasattr(c, 'shares'):
            c.shares = []
        ciphertext = c.ciphertext(be_secure=False)
        share = PartialShare(pow(ciphertext, self.si*2*self.delta, self.nSPlusOne), self.server_id)
        c.shares.append(share)
        return share
    def compute_ais(self, target_server_id):
        return self.ais * pow(target_server_id, self.i)
    def init_priv_key(self, priv_keys):
        for j in range(self.w):
            self.si += priv_keys[j].compute_ais(self.server_id)
        self.si = self.si % self.nm
    def decrypt_encoded(self, encrypted_number, Encoding=None):
        if not isinstance(encrypted_number, paillier.EncryptedNumber):
            raise TypeError('Expected encrypted_number to be an EncryptedNumber'
                            ' not: %s' % type(encrypted_number))

        if self.public_key != encrypted_number.public_key:
            raise ValueError('encrypted_number was encrypted against a '
                             'different key!')

        if Encoding is None:
            Encoding = paillier.EncodedNumber

        encoded = self.raw_decrypt(encrypted_number)
        return Encoding(self.public_key, encoded,
                             encrypted_number.exponent)

    def raw_decrypt(self, ciphertext):
        return self.combineShares(ciphertext.shares)
    def combineShares(self, shrs):
        pub_key = self.public_key
        w, delta, combineSharesConstant, nSPlusOne, n, ns = pub_key.w, pub_key.delta, pub_key.combineSharesConstant, pub_key.nSPlusOne, pub_key.n, pub_key.ns
        cprime = 1
        for i in range(w):
            ld = delta
            for iprime in range(w):
                if i != iprime:
                    if shrs[i].server_id != shrs[iprime].server_id:
                        ld = (ld * -shrs[iprime].server_id) // (shrs[i].server_id - shrs[iprime].server_id)
            #print(ld)
            shr = sympy.mod_inverse(shrs[i].share, nSPlusOne) if ld < 0 else shrs[i].share
            ld = -1*ld if ld <1 else ld
            temp = pow(shr, 2 * ld, nSPlusOne)
            cprime = (cprime * temp) % nSPlusOne
        L = (cprime - 1) // n
        result = (L * combineSharesConstant) % n
        return result - ns if result > (ns // 2) else result

class ThresholdPaillierPublicKey(paillier.PaillierPublicKey):
    def __init__(self,n, nSPlusOne, r, ns, w, delta, combineSharesConstant):
        super().__init__(n)
        self.n = n
        self.nSPlusOne = nSPlusOne
        self.r = r
        self.ns =ns
        self.w = w
        self.delta = delta
        self.combineSharesConstant = combineSharesConstant
    def raw_encrypt(self, msg, r_value=None):
        msg = msg % self.nSPlusOne if msg < 0 else msg
        c = (pow(self.n+1, msg, self.nSPlusOne) * pow(self.r, self.ns, self.nSPlusOne)) % self.nSPlusOne
        return c

class ThresholdPaillierTP(object):
    def __init__(self, n_length, **kwargs):
        self.n_length = n_length
    def setup(self, config_file, gc_conn, tp_conn):
        
        self.n_subm = len(config_file['models'])
        self.tp = ThresholdPaillier(self.n_length, self.n_subm, self.n_subm)

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
                await websocket.send(dumps(serialize_pub(self.tp.pub_key)))
                logging.info(f"send key {port} > {self.tp.pub_key}")
                stop.set_result(None)

            async def comms_server(stop):
                async with websockets.serve(serve_key, '0.0.0.0', port, ssl=ssl_context, max_size=MAX_SIZE, 
                read_limit=RW_BUFFER, write_limit=RW_BUFFER, max_queue=MAX_Q, ping_timeout=PING_TIMEOUT):
                    await stop

            a = loop.run_until_complete(comms_server(stop))

        serve_submodels(ssl_context, port)
    
    def tp_decrypt(self, encrypted):
        shares = []
        for pk in self.tp.priv_keys:
            shares.append(pk.partial_decrypt(encrypted))
        encrypted.shares = shares

        return self.tp.priv_keys[0].decrypt(encrypted)

    def tpv_decrypt(self, element):
        # Case 1: tensor recursion
        if isinstance(element, torch.Tensor):
            paillier = element.child
            if isinstance(paillier, PaillierTensor):
                return torch.Tensor([self.tpv_decrypt(subchild) for subchild in paillier.child])
            else:
                raise TypeError(type(paillier))
        elif isinstance(element, ndarray):
            return [self.tpv_decrypt(subelement) for subelement in element]
        elif isinstance(element, EncryptedNumber):
            return self.tp_decrypt(element)
        else:
            raise TypeError(type(element))

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
                enc_mm_out = deserialize_paillier(loads(enc_mm_out), self.tp.pub_key)
                mm_out = self.tpv_decrypt(enc_mm_out)

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

class ThresholdPaillierEncryptedSubModel(net_layers.NetworkedSubModel):
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

class ThresholdPaillierEncryptedGCModel(net_layers.NetworkedGCModel):
    def setup(self, config_file, gc_conn, tp_conn):
        super().setup(config_file, gc_conn, tp_conn)
        ThresholdPaillierLinearFunction.tp_conn = tp_conn
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
        return ThresholdPaillierLinearFunction.grad_input.clone()

class ThresholdPaillierLinearFunction(th.autograd.Function):
    tp_conn = None
    grad_input = None
    @staticmethod
    async def decrypt_using_tp(enc_paillier_tensor):

        ssl_context, hostname, port = ThresholdPaillierLinearFunction.tp_conn
        assert(port == 8800)

        logging.info(f"Requesting decryption from {ThresholdPaillierLinearFunction.tp_conn}")
        
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
        output = asyncio.get_event_loop().run_until_complete(ThresholdPaillierLinearFunction.decrypt_using_tp(output))
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
            ThresholdPaillierLinearFunction.grad_input = grad_input
            grad_input = None
        if ctx.needs_input_grad[1]:
            # Performed in parallel in SA
            # SA also sends out the output to the individual nodes
            grad_weight = grad_output.t().mm(input)
            grad_weight.child.pubkey = input.child.pubkey

            # GC asks nodes for decryption from TP
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            grad_weight = asyncio.get_event_loop().run_until_complete(ThresholdPaillierLinearFunction.decrypt_using_tp(grad_weight))
            #input = asyncio.get_event_loop().run_until_complete(PaillierLinearFunction.decrypt_using_tp(input))
            #grad_weight = grad_output.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class ThresholdPaillierLinearCut(nn.Module):
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
        out = ThresholdPaillierLinearFunction.apply(input, self.weight, self.bias)
        # See the autograd section for explanation of what happens here.
        return out
    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )
