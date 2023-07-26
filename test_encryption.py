from tpe_layers import ThresholdPaillier
from itertools import combinations
import numpy as np

# content of test_expectation.py
import pytest

def decrypt(encrypted, priv_keys):
    for pk in priv_keys:
        pk.partial_decrypt(encrypted)
    return priv_keys[0].decrypt(encrypted)

@pytest.mark.parametrize(
    "n_shares,threshold,target",
    [(n_shares, i+1, target) for n_shares in range(10) for i in range(n_shares) for target in [-100, -1.1, 0, 1.1, 100]],
)
def test_encrypt_decrypt(n_shares, threshold, target):
    tp = ThresholdPaillier(128, n_shares, threshold)

    priv_keys = tp.priv_keys
    pub_key = tp.pub_key

    c = pub_key.encrypt(target)

    for comb in combinations(range(n_shares), threshold):
        pub_key = tp.pub_key
        shares = []
        for i in comb:
            shares.append(priv_keys[i].partial_decrypt(c))
        c.shares = shares
        assert(priv_keys[0].decrypt(c) == target)

@pytest.mark.parametrize(
    "a,b,fn",
    [(int(a),int(b),fn) for a in np.random.randint(-100, 100, size=10) for b in np.random.randint(-100, 100, size=10) for fn in [lambda a,b:a+b, lambda a,b:a-b]]
    + [(float(a),float(b),fn) for a in np.random.uniform(-100, 100, size=10) for b in np.random.uniform(-100, 100, size=10) for fn in [lambda a,b:a+b, lambda a,b:a-b]],
)
def test_compute_symmertric(a, b, fn):
    tp = ThresholdPaillier(128, 5, 5)

    priv_keys = tp.priv_keys
    pub_key = tp.pub_key
    c = fn(a,b)
    
    _a = pub_key.encrypt(a)
    _b = pub_key.encrypt(b)
    _c = fn(_a,_b)
    
    assert(decrypt(_c, priv_keys) == c)

@pytest.mark.parametrize(
    "a,b,fn",
    [(int(a),int(b),fn) for a in np.random.randint(-100, 100, size=10) for b in np.random.randint(-100, 100, size=10) for fn in [lambda a,b:a*b]]
    + [(float(a),float(b),fn) for a in np.random.uniform(-100, 100, size=10) for b in np.random.uniform(-100, 100, size=10) for fn in [lambda a,b:a*b]],
)
def test_compute_unsymmertric(a, b, fn):
    tp = ThresholdPaillier(128, 5, 5)

    priv_keys = tp.priv_keys
    pub_key = tp.pub_key
    c = fn(a,b)
    
    _a = pub_key.encrypt(a)
    _b = b
    _c = fn(_a,_b)
    
    assert(decrypt(_c, priv_keys) == c)

@pytest.mark.parametrize(
    "a,b,c,fn",
    [(int(a),float(b),int(c),fn) for a in np.random.randint(-100, 100, size=10) for b in np.random.uniform(-100, 100, size=10) for c in np.random.randint(-100, 100, size=10) for fn in [lambda a,b,c:(a+b)*c]],
)
def test_compute_multi(a, b, c, fn):
    tp = ThresholdPaillier(128, 5, 5)

    priv_keys = tp.priv_keys
    pub_key = tp.pub_key
    d = fn(a, b, c)
    
    _a = pub_key.encrypt(a)
    _b = pub_key.encrypt(b)
    _d = fn(_a,_b, c)
    
    assert(np.isclose(decrypt(_d, priv_keys), d))