import torch
import sys
from torchvision import datasets, transforms
import numpy as np
import math
import yaml
from models.cc_gcmodel_def import CCGCNet
from models.cc_submodel_def import FourLayerNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pandas as pd

FILES_DIR = '~'

# set seed so the thing is determinsitic
np.random.seed(seed=0)

args_dict = {}

def prepareDatasets(): # prepare dataset for each model
    df = pd.read_csv('files/ccfraud/creditcard.csv')

    X = df.iloc[:, :-1].values  # extracting features
    y = df.iloc[:, -1].values  # extracting labels

    sc = StandardScaler()
    X = sc.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    train_images = []
    test_images = []

    num_features = math.floor(D_in / num_models)
    start = 0
    end = start + num_features
    for i in range(num_models - 1):
        train_images.append(X_train[:, start:end])
        test_images.append(X_test[:, start:end])
        start += num_features
        end += num_features
    if (num_features * num_models != D_in):
        train_images.append(X_train[:, start:D_in])
        test_images.append(X_test[:, start:D_in])
    else:
        train_images.append(X_train[:, start:end])
        test_images.append(X_test[:, start:end])

    print(X_train.shape)
    print(X_test.shape)

    id_vector = np.arange(len(Y_train)).reshape((len(Y_train), 1))
    test_id_vector = np.arange(len(Y_test)).reshape((len(Y_test), 1))
    for i in range(len(train_images)):
        train_images[i] = np.hstack((id_vector, train_images[i]))
        train_images[i] = pd.DataFrame(data=train_images[i][:, 1:], index=np.int_(train_images[i][:, 0]))
        # trainimages[i] = trainimages[i].reindex(np.random.permutation(trainimages[i].index))

        test_images[i] = np.hstack((test_id_vector, test_images[i]))
        test_images[i] = pd.DataFrame(data=test_images[i][:, 1:], index=np.int_(test_images[i][:, 0]))
        # testimages[i] = testimages[i].reindex(np.random.permutation(testimages[i].index))

    print(train_images)
    print(test_images)

    train_labels = pd.DataFrame(data=Y_train)
    # trainlabels = trainlabels.reindex(np.random.permutation(trainlabels.index))
    test_labels = pd.DataFrame(data=Y_test)
    # testlabels = testlabels.reindex(np.random.permutation(testlabels.index))

    print(train_labels)
    print(test_labels)

    os.makedirs(os.path.expanduser(f'{FILES_DIR}/files/ccfraud/dataset'), exist_ok=True)
    # os.makedirs(os.path.expanduser(f'{FILES_DIR}/models/ccfraud'), exist_ok=True)
    # pandas dataframe with integer ids
    for i in range(num_models):
        with open(os.path.expanduser(f"{FILES_DIR}/files/ccfraud/dataset/X_train_{i}.pkl"), "wb") as f:
            train_images[i].to_pickle(f)
        with open(os.path.expanduser(f"{FILES_DIR}/files/ccfraud/dataset/X_test_{i}.pkl"), "wb") as f:
            test_images[i].to_pickle(f)

    with open(os.path.expanduser(f"{FILES_DIR}/files/ccfraud/dataset/Y_train.pkl"), "wb") as f:
        train_labels.to_pickle(f)
    with open(os.path.expanduser(f"{FILES_DIR}/files/ccfraud/dataset/Y_test.pkl"), "wb") as f:
        test_labels.to_pickle(f)

    return len(Y_train), len(Y_test)

def prepareModels():
    submodel_D_in = int(D_in / num_models)
    submodels = [FourLayerNet(submodel_D_in, H_1, H_2, H_3, H_4) for i in range(num_models - 1)]
    for i in range(num_models - 1):
        args_dict[i] = submodel_D_in
    submodels.append(FourLayerNet(D_in - ((num_models - 1) * submodel_D_in), H_1, H_2, H_3, H_4))
    args_dict[num_models - 1] = D_in - ((num_models - 1) * submodel_D_in)
    submodel_H = H_4 * num_models
    gc_model = CCGCNet(submodel_H, D_out)

    os.makedirs("./models/ccfraud/", exist_ok=True)
    for i in range(len(submodels)):
        torch.save(submodels[i].state_dict(), "./models/ccfraud/submodel_{}_statedict".format(i))
    torch.save(gc_model.state_dict(), "./models/ccfraud/submodel_{}_statedict".format(len(submodels)))

def prepareConfig(N_train, N_test):
    config_dict = {
        'validation': {
            'N_train': N_train,
            'N_test': N_test,
        },
        'training_iters': 5,
        'batch_size': 256,
        'random_seed': 0,
        'is_data_aligned': True,
        'models': {
            'gc_model': {
                'ports': {
                    'submodel0': 8600,
                    'submodel1': 8601,
                    'submodel2': 8602
                },
                'checkpoint_dir': "/tmp/gc_model",
                'data_path': f"{FILES_DIR}/files/ccfraud/dataset/Y_{{}}.pkl",
                'model_path': "./models/ccfraud/submodel_{}_statedict".format(num_models),
                'model_def_path': 'models.cc_gcmodel_def',
                'model_class': 'CCGCNet',
                'param': {
                    'loss_fn': loss,
                    'model_args': {
                        'D_in': H_4 * num_models,
                        'D_out': D_out
                    },
                    'optimizer': {
                        'lr': lr,
                        'name': optimizer
                    },
                },
            },
            'sub_models': {

            }
        }
    }
    for i in range(num_models):
        config_dict['models']['sub_models']['submodel{}'.format(i)] = {
            'checkpoint_dir': "/tmp/submodel{}".format(i),
            'data_path': f"{FILES_DIR}/files/ccfraud/dataset/X_{{}}_{i}.pkl",
            'model_path': "./models/ccfraud/submodel_{}_statedict".format(i),
            'model_def_path': "models.cc_submodel_def",
            'model_class': 'FourLayerNet',
            'param': {
                'model_args': {
                    'D_in': args_dict[i],
                    'H_1': 5,
                    'H_2': 6,
                    'H_3': 7,
                    'H_4': 8
                },
                'optimizer': {
                    'lr': lr,
                    'name': optimizer
                }
            }
        }

    with open('config3.yaml', 'w') as f:
        yaml.dump(config_dict, f)

D_in = 30
H_1 = 5
H_2 = 6
H_3 = 7
H_4 = 8
D_out = 1
num_models = int(sys.argv[1])
loss = sys.argv[2]
optimizer = sys.argv[3]
lr = float(sys.argv[4])
print(num_models)
N_train, N_test = prepareDatasets()
prepareModels()
prepareConfig(N_train, N_test)

