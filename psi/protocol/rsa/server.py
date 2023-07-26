from Crypto.PublicKey import RSA
from psi.protocol import rsa
from psi.protocol.rsa import utils
import concurrent.futures
from multiprocessing import Process, Queue
import multiprocessing
import numpy as np

def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]

class Server:
    """Server for RSA-PSI protocol"""

    def __init__(self, private_key=None, key_size=2048, e=0x10001):
        """
        Args:
            private_key: RSA private key, a key will be generated using
                e and key_size if it's not provided.
            key_size: size in bits of the key.
            e: RSA public exponent.
        """

        if private_key is None:
            self.public_key, self.private_key = rsa.keygen(key_size, e)
        else:
            self.public_key, self.private_key = rsa.from_private(
                self.private_key)

    @property
    def keys(self):
        return self.public_key, self.private_key

    def sign(self, x):
        """Sign a single element using the RSA private key.

        Args:
            x: integer in the range [0, n), where n is the RSA modulus.

        Returns:
            The signature of x.
        """
        #print(x)
        assert 0 <= x < self.private_key.n, "x should be in range [0, {})".format(
            self.private_key.n)

        return utils.sign(self.private_key, x)

    def _sign_set(self, X):
        """Sign a set of elements using the RSA private key.

        Args:
            X: list of integers in the range [0, n), where n is the RSA modulus.

        Returns:
            A list of integers representing the signatures of the set X.
        """
        signatures = []
        for x in X:
            s = self.sign(x)
            signatures.append(s)
        return signatures
    def sign_set(self, X):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        N_WORKERS = multiprocessing.cpu_count()
        #return self._sign_set(X)
        batched = chunks(X, N_WORKERS)

        jobs = []
        for i, ea_batch in enumerate(batched):
            p = Process(target=self._parallel_sign_set, args=(ea_batch, i, return_dict))
            p.start()
            jobs.append(p)
        
        for j in jobs:
            j.join()

        return_signed = []
        for i in range(N_WORKERS):
            return_signed += return_dict[i]
        return return_signed

    # Wrapper to ensure that the data is returned properly
    def _parallel_sign_set(self, X, procnum, return_dict):
        sigs = self._sign_set(X)
        return_dict[procnum] = sigs