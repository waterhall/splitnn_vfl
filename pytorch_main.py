import yaml
import sys
from vfl import *
import logging_utils
import torch
import numpy as np

def main(argv, arc):
    
    with open(argv[1], 'r') as stream:
        try:
            vfl_params = yaml.safe_load(stream)
            # print(vfl_params)
            vfl_model = VFLModel(**vfl_params['models'])
        except yaml.YAMLError as exc:
            print(exc)

    tags = {
        "mlflow.runName"    : f'{vfl_params["name"]}-pytorch',
        "host_id"         : 'localhost',
        "host_name"     : 'localhost'
    }
    mlflow.set_tags(tags)
    vfl_params['random_seed'] = int(argv[2])
    
    # This is where we initalize
    batch_size = vfl_params['batch_size']
    N_train = vfl_params['validation']['N_train']
    N_test = vfl_params['validation']['N_test']

    train_ids, test_ids = vfl_model.gc_model.data.get_ids()

    for sub_model in vfl_model.sub_models_dict.values():
      train_ids_part, test_ids_part = sub_model.data.get_ids()
      train_ids &= train_ids_part
      test_ids &= test_ids_part

    is_data_aligned = vfl_params['is_data_aligned']
    if not is_data_aligned:
        vfl_model.align(train_ids, test_ids) # align datasets

    pytorch_model = vfl_model.to_pytorch()
    loss_fn = vfl_model.gc_model.loss_fn
    print(pytorch_model.parameters())
    optimizer = vfl_model.gc_model.loadOptimizer(pytorch_model.parameters())
    X_datas = [ subm.data for subm in vfl_model.sub_models_dict.values() ]
    X_test = [ torch.from_numpy(subm.data.test.to_numpy(dtype=np.float32)) for subm in vfl_model.sub_models_dict.values() ]
    
    Y_data = vfl_model.gc_model.data

    batch_i = 0
    indices = np.arange(N_train)
    np.random.seed(vfl_params['random_seed'])
    np.random.shuffle(indices)

    for t in range(vfl_params['training_iters']):
        print(f"===========Iteration {t}===========")
        prev_batch_i = batch_i
        batch_i = (t * batch_size) % N_train
        # check for reset!
        if prev_batch_i > batch_i:
            print("Protocol - Shuffle together")
            # Shuffle together
            np.random.shuffle(indices)

        batch_ids = indices[batch_i : batch_i + batch_size]

        optimizer.zero_grad()
        X = [ torch.from_numpy(X_data.getData(batch_ids, train=True, dtype=np.float32)) for X_data in X_datas ]

        if type(loss_fn) == torch.nn.modules.loss.BCELoss:  # if BCE
            y = torch.from_numpy(Y_data.getData(batch_ids, train=True, dtype=np.float32))
        else:  # if crossentropyloss
            y = torch.from_numpy(Y_data.getData(batch_ids, train=True, dtype=np.int64).squeeze())

        
        outputs = pytorch_model(X)

        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        loss = loss.item()

        with torch.no_grad():
            metric = {}
            outputs = pytorch_model(X_test)
            
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == torch.from_numpy(vfl_model.gc_model.data.test.to_numpy(dtype=np.int64).squeeze())).sum().item()
            accuracy = correct / len(vfl_model.gc_model.data.test.to_numpy(dtype=np.int64).squeeze())
            metric['accuracy'] = accuracy
            logging.debug(f"Test Accuracy: {accuracy}")
            if type(vfl_model.gc_model.loss_fn) == torch.nn.modules.loss.BCELoss:
                auc = roc_auc_score(vfl_model.gc_model.data.test.to_numpy(dtype=np.int64), outputs)
                metric['roc_auc_score'] = auc
                logging.debug(f"roc_auc_score: {auc}")
            
        #y_pred, metric = vfl_model.predict() # if never pass in anything, means predict over entire test set
        
        metric['loss'] = loss
        mlflow.log_metrics(metric, step=t)

        print(loss, metric)

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))