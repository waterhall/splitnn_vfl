import yaml
import sys
from vfl import *
import logging_utils

def main(argv, arc):
    
    with open(argv[1], 'r') as stream:
        try:
            vfl_params = yaml.safe_load(stream)
            # print(vfl_params)
            vfl_model = VFLModel(**vfl_params['models'])
        except yaml.YAMLError as exc:
            print(exc)

    tags = {
        "mlflow.runName"    : f'{vfl_params["name"]}-framework',
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

        loss = vfl_model.train(batch_ids)
        y_pred, metric = vfl_model.predict() # if never pass in anything, means predict over entire test set

        metric['loss'] = loss
        mlflow.log_metrics(metric, step=t)

        print(loss, metric)

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))