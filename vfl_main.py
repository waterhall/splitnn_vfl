import sys
import yaml
import json
import base64
import os
from remote import LocalFileManager, ConfigFile
import importlib
#import rsa_alignment as alignment
alignment = importlib.import_module('rsa_alignment')

from ssl_utils import SSHKeyPairFile
from remote import LocalFileManager
import asyncio
import sys
import time
import logging_utils
import logging
import mlflow
from collections import defaultdict
from vfl import GCModel, SubModel, TrustedParty
import numpy as np
from multiprocessing import Process

def main(argv, arc):
    encoded_config = argv[1]
    config = json.loads(base64.urlsafe_b64decode(encoded_config.encode()).decode())
    
    host_id = config['id']

    logging.info(f"Parameters: {config}")
    l_fm = LocalFileManager(config['id'])
    
    config_file = None
    # Find the ConfigFile
    for f in l_fm.files:
        if type(f) == ConfigFile:
            config_file = f

    config_file = config_file.to_dict()
    config_file['random_seed'] = config['random_seed']
    
    host_configs = config_file['hosts']
    allocations = config_file['allocations']
    vfl_params = config_file['models']

    inv_allocations = defaultdict(set)
    for k, v in allocations.items():
        inv_allocations[v].add(k)

    gc_host = allocations['gc_model']
    node_size = len(host_configs)

    host_list = [ (host_ref, SSHKeyPairFile(host_ref), host_config['conn_param']['host'], host_config['ports']['alignment']) for host_ref, host_config in host_configs.items() ]
    # GC must be at position index=1, so that it ends first
    gc_index = next(i for i, (host_ref, _, _, _) in enumerate(host_list) if host_ref == gc_host)
    shift_gc = len(host_list)-gc_index+1
    host_list = (host_list[-shift_gc:] + host_list[:-shift_gc]) 

    # Init the ssh keys
    ssh_key_files = [ ssh_f for _,ssh_f,_,_ in host_list ]
    l_fm = LocalFileManager(host_id, ssh_key_files)
    l_fm.create_all()

    node_id = 0
    my_host_name = None
    host_map = { host_ref : (ssl_key, host_name) for host_ref, ssl_key, host_name, a in host_list }
    for host_ref, _, host_name, a in host_list:
        if host_ref == host_id:
          my_host_name = host_name
          break
        node_id += 1

    mlflow.log_params(config)

    tags = {
        "mlflow.runName"    : f'{config_file["name"]}-{host_id}-networked',
        "host_id"         : host_id,
        "host_name"     : my_host_name
    }
    mlflow.set_tags(tags)

    # Prepare connections for everyone
    ssl_key, hostname = host_map[allocations['gc_model']]
    gc_conn = ssl_key.ssl_context(), hostname

    ssl_key, hostname = host_map[allocations['trusted_party']]
    tp_conn = ssl_key.ssl_context(), hostname

    model_allocations = inv_allocations[host_id]

    processes = []
    my_models = []
    for ea_model_ref in model_allocations:

      # Append the target port that matters to the model
      gc_port = vfl_params['gc_model']['ports'][ea_model_ref] if ea_model_ref in vfl_params['gc_model']['ports'] else None
      tp_port = config_file['trusted_party']['ports'][ea_model_ref] if ea_model_ref in config_file['trusted_party']['ports'] else None

      my_gc_conn = gc_conn + (gc_port,)
      my_tp_conn = tp_conn + (tp_port,)

      if 'gc_model' == ea_model_ref :
        my_models.append(GCModel(**vfl_params['gc_model'], model_ref=ea_model_ref, gc_conn=my_gc_conn, tp_conn=my_tp_conn))
      elif 'trusted_party' == ea_model_ref :
        tp = TrustedParty(**config_file['trusted_party'], model_ref=ea_model_ref, gc_conn=my_gc_conn, tp_conn=my_tp_conn)
        # Start the TP early since there is no need for alignment here
        process = Process(target=tp.train_networked, args=(config_file,))
        process.start()
        processes.append(process)
      else:
        my_models.append(SubModel(**vfl_params['sub_models'][ea_model_ref], model_ref=ea_model_ref, gc_conn=my_gc_conn, tp_conn=my_tp_conn))

    assert(len(my_models)>0)
    
    # This is safe as its a garuntee that if one of the models is aligned all are.
    train_ids, test_ids = my_models[0].data.get_ids()

    # TODO True or 
    # Perform alignment if the database is not aligned
    if not my_models[0].data.is_aligned():
      logging.info("Performing alignment.")

      
      N_train = config_file['validation']['N_train']
      N_test = config_file['validation']['N_test']
      train_ids = {i for i in range(N_train)}
      test_ids = {i for i in range(N_test)}
      for model in my_models:
        model.data.align(train_ids, test_ids)
      '''
      for sub_model in my_models[1:]:
        train_ids_part, test_ids_part = sub_model.data.get_ids()
        train_ids &= train_ids_part
        test_ids &= test_ids_part    
      logging.info(f"Aligning: Train - {len(train_ids)}, Test - {len(test_ids)}")

      alignment_datas = {
        'test': list(test_ids),
        'train': list(train_ids)
      }

      my_address = host_list[node_id]
      to_address = host_list[(node_id+1)%node_size]

      for d_t in alignment_datas.keys():

        ids = alignment_datas[d_t]

        aligned_ids = alignment.user_align(node_id, my_address, to_address, ids, node_size)
        
        metrics = {
          f'{d_t}_align_total_time': alignment.total_time_stopwatch.duration,
          f'{d_t}_align_send_time' : alignment.send_time_stopwatch.duration,
          f'{d_t}_align_recv_time': alignment.recv_time_stopwatch.duration,
          f'{d_t}_align_send_bytes': alignment.send_time_stopwatch.bytes,
          f'{d_t}_align_recv_bytes': alignment.recv_time_stopwatch.bytes
        }
        mlflow.log_metrics(metrics) # , step=limit

        #assert(aligned_db == {i for i in range(60000)})
        logging.info(f"{host_id} Aligned {d_t} dataset - {len(aligned_ids)}")
        alignment_datas[d_t] = aligned_ids

      # Aligned, apply ids to dataset
      train_ids, test_ids = alignment_datas['train'], alignment_datas['test']
      for model in my_models:
        model.data.align(train_ids, test_ids)
      
      logging.info("Exit after successful align.")
      import sys
      sys.exit(0)
      '''
    else:
      logging.info("Skipping alignment, using checkedpointed user ids.")

    for model in my_models:
      process = Process(target=model.train_networked, args=(config_file, ))
      process.start()
      processes.append(process)

    for process in processes:
      process.join()

    l_fm.exp_cleanup()

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))