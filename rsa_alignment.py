import ssl_utils
from psi.protocol import rsa
from psi.datastructure import bloom_filter
from pickle import dumps, loads
from cryptography.hazmat.primitives import serialization
import logging

import asyncio
import pathlib
import ssl
import websockets
import signal
from Crypto.PublicKey import RSA
from psi.protocol import rsa
import time
from invoke.exceptions import UnexpectedExit
import sys
from stopwatch import Stopwatch

total_time_stopwatch = Stopwatch()
send_time_stopwatch = Stopwatch()
recv_time_stopwatch = Stopwatch()

# Transfer 10MiB
MAX_SIZE = 1*1024**3
RW_BUFFER = 2**24
MAX_Q = 1024
CLOSE_TIMEOUT = 100

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

BATCH_SIZE = 1000
async def send_data_batch(websocket, data):
    #print("send_data_batch")
    data_len  = len(data)
    send_time_stopwatch.bytes += sys.getsizeof(data_len)
    send_time_stopwatch.start()
    await websocket.send(dumps(data_len))
    send_time_stopwatch.stop()

    for ea_batch in batch(data, BATCH_SIZE):
        #print("send batch")
        send_time_stopwatch.bytes += sys.getsizeof(ea_batch)
        send_time_stopwatch.start()
        await websocket.send(dumps(ea_batch))
        send_time_stopwatch.stop()

async def recv_data_batch(websocket):
    #print("recv_data_batch")
    recv_time_stopwatch.start()
    data_len = await websocket.recv()
    recv_time_stopwatch.stop()
    data_len = loads(data_len)
    recv_time_stopwatch.bytes += sys.getsizeof(data_len)

    data = []
    while len(data) < data_len:
        #print("recv batch")
        recv_time_stopwatch.start()
        ea_batch = await websocket.recv()
        recv_time_stopwatch.stop()
        ea_batch = loads(ea_batch)
        data += ea_batch
        recv_time_stopwatch.bytes += sys.getsizeof(ea_batch)

    return data

def align_from(my_set, ssl_context, hostname="0.0.0.0", port=8500):

    # This RSA will only be for the UA purposes
    server = rsa.Server()
    ua_public_key = server.public_key
    loop = asyncio.get_event_loop()
    stop = loop.create_future()

    async def ua_get(websocket, path):
        ## SETUP
        #print("signing")
        signed_server_set = server.sign_set(my_set)
        #print("signed")
        # must encode to bytes
        signed_server_set = [str(sss).encode() for sss in signed_server_set]
        
        pub_key_pem = ua_public_key.exportKey()
        send_time_stopwatch.bytes += sys.getsizeof(pub_key_pem)
        send_time_stopwatch.start()
        await websocket.send(dumps(pub_key_pem))
        send_time_stopwatch.stop()
        await send_data_batch(websocket, signed_server_set)

        # Sign a set
        A = await recv_data_batch(websocket)
        B = server.sign_set(A)
        await send_data_batch(websocket, B)

        stop.set_result(None)

    async def ua_get_server(stop):
        async with websockets.serve(ua_get, hostname, port, ssl=ssl_context, max_size=MAX_SIZE, 
            read_limit=RW_BUFFER, write_limit=RW_BUFFER, max_queue=MAX_Q, close_timeout=CLOSE_TIMEOUT):
            await stop

    loop.run_until_complete(ua_get_server(stop))
    
async def align_to(ssl_context, hostname, client_set, port=8500):
    n_retries = 0
    while n_retries < 10:
        try:
            uri = f"wss://{hostname}:{port}"
            async with websockets.connect(
            uri, ssl=ssl_context, max_size=MAX_SIZE, 
            read_limit=RW_BUFFER, write_limit=RW_BUFFER, max_queue=MAX_Q,
            close_timeout=CLOSE_TIMEOUT
            ) as websocket:

                recv_time_stopwatch.start()
                pub_key_pem = await websocket.recv()
                recv_time_stopwatch.stop()
                pub_key_pem = loads(pub_key_pem)
                signed_server_set = await recv_data_batch(websocket)
                bf = bloom_filter.build_from(signed_server_set)
                public_key = RSA.importKey(pub_key_pem)

                # Got the server public key, now exchanging info
                client = rsa.Client(public_key)
                random_factors = client.random_factors(len(client_set))

                ## ONLINE
                A = client.blind_set(client_set, random_factors)
                await send_data_batch(websocket, A)
                B = await recv_data_batch(websocket)

                unblinded_client_set = client.unblind_set(B, random_factors)
                # must encode to bytes
                unblinded_client_set = [str(ucs).encode() for ucs in unblinded_client_set]

                intr = client.intersect(client_set, unblinded_client_set, bf)

            return intr
        except Exception as e: 
            logging.warning(f"Retrying for the {n_retries} times")
            import traceback
            traceback.print_exc()
            logging.exception("Expected timeout error")

            time.sleep(1)

            n_retries += 1
            

def user_align(node_id, my_address, to_address, my_set, node_size):
    
    total_time_stopwatch.restart()
    send_time_stopwatch.reset()
    recv_time_stopwatch.reset()
    
    send_time_stopwatch.bytes = 0
    recv_time_stopwatch.bytes = 0

    # Listen on 0.0.0.0 instead of the hostname to listen on all interfaces
    _, my_ssl, _, my_port = my_address
    _, to_ssl, to_hostname, to_port = to_address

    my_ssl_context = my_ssl.ssl_context()
    to_ssl_context = to_ssl.ssl_context()

    for i in range(node_size-1):
        try:
            if node_id == 0:
                my_set = asyncio.get_event_loop().run_until_complete(align_to(to_ssl_context, to_hostname, my_set, to_port ))
                align_from(my_set, my_ssl_context, "0.0.0.0", my_port)
                time.sleep(1)
            else:
                align_from(my_set, my_ssl_context, "0.0.0.0", my_port)
                time.sleep(1)
                my_set = asyncio.get_event_loop().run_until_complete(align_to(to_ssl_context, to_hostname, my_set, to_port ))
        except Exception as e: 
            import traceback
            traceback.print_exc()
            logging.exception("Unexpected Error")

    total_time_stopwatch.stop()
    return my_set
