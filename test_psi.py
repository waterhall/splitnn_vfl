from psi.protocol import rsa
from psi.datastructure import bloom_filter
import pytest
import random

def run_protocol(client_set, server_set):
    ## BASE
    server = rsa.Server()
    public_key = server.public_key
    client = rsa.Client(public_key)
    print("RNG")
    random_factors = client.random_factors(len(client_set))
    print("Setup")
    ## SETUP
    signed_server_set = server.sign_set(server_set)
    print("Encode")
    # must encode to bytes
    signed_server_set = [str(sss).encode() for sss in signed_server_set]
    print("BF")
    bf = bloom_filter.build_from(signed_server_set)
    print("A")
    ## ONLINE
    A = client.blind_set(client_set, random_factors)
    print("B")
    B = server.sign_set(A)
    print("unblind")
    unblinded_client_set = client.unblind_set(B, random_factors)
    print("unblind-bytes")
    # must encode to bytes
    unblinded_client_set = [str(ucs).encode() for ucs in unblinded_client_set]
    
    intr = client.intersect(client_set, unblinded_client_set, bf)
    return intr


N_ELEMS = 10000
population = range(1000, N_ELEMS)
random.seed(0)
sample = random.sample(population, 10)

@pytest.mark.parametrize(
    "n_elems",
    [ ea_item for ea_item in sample],
)
def test_ua_protocol(n_elems):
    A = list(range(n_elems))
    B = list(range(n_elems+10))

    C = run_protocol(A, B)
    print(len(C))
    assert(len(C) == n_elems)