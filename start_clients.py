from fabric import Connection
from collections import deque
import yaml
import json
import base64
from ssl_utils import SSHKeyPairFile
from remote import LocalFileManager, RemoteFileManager, ConfigFile
from collections import defaultdict
import sys
import logging_utils
import logging
from threading import Thread


# ssh -i ssh_keys/ba.key darthvader_aka_eric@34.70.251.74
# https://stackoverflow.com/questions/54612609/paramiko-not-a-valid-rtp-private-key-file
# ssh-keygen -t rtp -f ssh_keys/gcloud.key -C darthvader_aka_eric
# ssh-keygen -p -m PEM -f ./ssh_keys/gcloud.key

# Read from yaml file, temp
config_file_path = sys.argv[1]
config_file = ConfigFile(config_file_path)
config_file_params = config_file.to_dict()

random_seed = int(sys.argv[2])

host_configs = config_file_params['hosts'].items()

allocations = config_file_params["allocations"]
inv_allocations = defaultdict(set)
for k, v in allocations.items():
    inv_allocations[v].add(k)

tp_ssl = None
gc_ssl = None

ssl_files = []
for host_config in host_configs:
    host_ref, params = host_config

    hostname = params['conn_param']['host']
    ssl_files.append(SSHKeyPairFile(host_ref))

    if 'trusted_party' in inv_allocations[host_ref]:
        tp_ssl = ssl_files[-1]

    if 'gc_model' in inv_allocations[host_ref]:
        gc_ssl = ssl_files[-1]

l_fm = LocalFileManager('distrb_keys', ssl_files + [ config_file ])
l_fm.create_all()

# Shifted the hostname by 1
# So, 1>2 2>3 ... N-1>N N>1
# 1's next is 2
# So 1 will send message to 2
ssl_files_shifted = deque(ssl_files)
ssl_files_shifted.rotate(-1)

threads = []
for host_config, ssl_file, next_ssl in reversed(list(zip(host_configs, ssl_files, ssl_files_shifted))):
    
    host_ref, params = host_config

    conn_param = params['conn_param']

    c = Connection(**conn_param)
    r_fm = RemoteFileManager(c, host_ref=host_ref)

    # Transfer the following keys
    # localhost keys, next key and 
    transfer_file_params = [ (ssl_file, dict(key_ref='localhost')),
                        (next_ssl, dict(key_ref='next')),
                        (tp_ssl, dict(key_ref='tp')),
                        (gc_ssl, dict(key_ref='gc')),
                        (config_file, dict(config_ref='param')) ]

    for file, param in transfer_file_params:
        r_fm.transfer_from(l_fm, file, **param)

    r_fm.end_transfer(l_fm)

    run_param = {"id" : host_ref,
                "random_seed": random_seed}
    dynamic_params = {'encoded_config': base64.urlsafe_b64encode(json.dumps(run_param).encode()).decode('UTF-8')}

    mlflow_params = config_file_params['mlflow']

    cmd = mlflow_params['preamble'] + mlflow_params['mlflow_run']
    cmd = cmd.format(**dynamic_params)

    def threaded_function(cmd, c, host_ref):
        logging.info(f"Starting remote node at {host_ref} -- {c.host}")
        result = c.run(cmd, pty=False, hide=False)
        
        cmd_output = result.stdout
        print(f"=========={host_ref} -- {c.host}==========")
        print(cmd_output)
        print(f"=========={host_ref} -- {c.host}==========")

    thread = Thread(target = threaded_function, args = (cmd, c, host_ref))
    threads.append(thread)
    
# Start together, to reduce sync issues
for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

l_fm.exp_cleanup()