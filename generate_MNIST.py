import torch
import sys
from torchvision import datasets, transforms
import numpy as np
import math
import yaml
from models.gcmodel_def import GCNet
from models.submodel_def import TwoLayerNet
import os
import pandas as pd

FILES_DIR = '~'

# set seed so the thing is determinsitic
np.random.seed(seed=0)

# replace downloadmnist
# sys.argv[1] will be num of models (excluding gc)
args_dict = {}

def prepareDatasets(): # download MNIST datasets and prepare npy for each model
    transform = transforms.Compose([transforms.ToTensor()])  # data will be in range [0, 1]
    train_set = datasets.MNIST(os.path.expanduser(f'{FILES_DIR}/files/'), download=True, train=True, transform=transform)
    test_set = datasets.MNIST(os.path.expanduser(f'{FILES_DIR}/files/'), download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=60000, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False)

    data_iter = iter(train_loader)
    images, train_labels = data_iter.next()
    images_train = images.reshape(images.shape[0], -1)
    data_iter = iter(test_loader)
    images, test_labels = data_iter.next()
    images_test = images.reshape(images.shape[0], -1)

    train_images = []
    test_images = []

    num_features = math.floor(784 / num_models)
    start = 0
    end = start + num_features
    for i in range(num_models - 1):
        train_images.append(images_train.numpy()[:, start:end])
        test_images.append(images_test.numpy()[:, start:end])
        start += num_features
        end += num_features
    if (num_features * num_models != 784):
        train_images.append(images_train.numpy()[:, start:784])
        test_images.append(images_test.numpy()[:, start:784])
    else:
        train_images.append(images_train.numpy()[:, start:end])
        test_images.append(images_test.numpy()[:, start:end])

    # generates 1000 rows of random fake data for each node
    # the 1000 ids may not be the same (ids in the range of [60000, 70000) )
    # will save them as pandas dataframe because numpy does not allow mixing of datatypes
    # and we are using integers for the id
    id_vector = np.arange(60000).reshape((60000, 1))
    test_id_vector = np.arange(10000).reshape((10000, 1))
    for i in range(len(train_images)):
        generated_test_ids = np.arange(10000 + i * 1000, 11000 + i * 1000)
        generated_ids = np.arange(60000 + i * 1000, 62000 + i * 1000)
        rng = np.random.default_rng()
        generated_id_vector = rng.choice(generated_ids, size=1000, replace=False).reshape((1000, 1))
        generated_features = np.random.rand(1000, train_images[i].shape[1])
        generated_data = np.hstack((generated_id_vector, generated_features))
        train_images[i] = np.hstack((id_vector, train_images[i]))
        train_images[i] = np.vstack((train_images[i], generated_data))
        train_images[i] = pd.DataFrame(data=train_images[i][:, 1:], index=np.int_(train_images[i][:, 0]))
        train_images[i] = train_images[i].reindex(np.random.permutation(train_images[i].index))

        generated_id_vector = rng.choice(generated_test_ids, size=1000, replace=False).reshape((1000, 1))
        generated_features = np.random.rand(1000, test_images[i].shape[1])
        generated_data = np.hstack((generated_id_vector, generated_features))
        test_images[i] = np.hstack((test_id_vector, test_images[i]))
        test_images[i] = np.vstack((test_images[i], generated_data))
        test_images[i] = pd.DataFrame(data=test_images[i][:, 1:], index=np.int_(test_images[i][:, 0]))
        test_images[i] = test_images[i].reindex(np.random.permutation(test_images[i].index))
        # testimages[i] = pd.DataFrame(data=testimages[i])
        print(train_images[i])
        print(test_images[i])

    generated_labels = np.random.randint(10, size=1000)
    train_labels = np.hstack((train_labels, generated_labels))
    train_labels = pd.DataFrame(data=train_labels)
    train_labels = train_labels.reindex(np.random.permutation(train_labels.index))
    generated_labels = np.random.randint(10, size=1000)
    test_labels = np.hstack((test_labels, generated_labels))
    test_labels = pd.DataFrame(data=test_labels)
    test_labels = test_labels.reindex(np.random.permutation(test_labels.index))

    print(train_labels)
    print(test_labels)

    # every node will have index 0-59999, but beyond that it's random fake data and ids
    
    os.makedirs(os.path.expanduser(f'{FILES_DIR}/files/MNIST/dataset'), exist_ok=True)
    # pandas dataframe with integer ids
    for i in range(num_models):
        with open(os.path.expanduser(f"{FILES_DIR}/files/MNIST/dataset/X_train_{i}.pkl"), "wb") as f:
            train_images[i].to_pickle(f)
        with open(os.path.expanduser(f"{FILES_DIR}/files/MNIST/dataset/X_test_{i}.pkl"), "wb") as f:
            test_images[i].to_pickle(f)

    with open(os.path.expanduser(f"{FILES_DIR}/files/MNIST/dataset/Y_train.pkl"), "wb") as f:
        # np.save(f, trainlabels)
        train_labels.to_pickle(f)
    with open(os.path.expanduser(f"{FILES_DIR}/files/MNIST/dataset/Y_test.pkl"), "wb") as f:
        # np.save(f, testlabels)
        test_labels.to_pickle(f)

    # numpy
    # for i in range(num_models):
    #     with open(f"{FILES_DIR}/files/MNIST/dataset/X_train_{i}.npy", "wb") as f:
    #         np.save(f, trainimages[i])
    #     with open(f"{FILES_DIR}/files/MNIST/dataset/X_test_{i}.npy", "wb") as f:
    #         np.save(f, testimages[i])
    #
    # with open(f"{FILES_DIR}/files/MNIST/dataset/Y_train.npy", "wb") as f:
    #     np.save(f, trainlabels)
    # with open(f"{FILES_DIR}/files/MNIST/dataset/Y_test.npy", "wb") as f:
    #     np.save(f, testlabels)

def prepareModels():
    D_in = 784
    submodel_D_in = int(D_in / num_models)
    submodels = [TwoLayerNet(submodel_D_in, H, H_2) for i in range(num_models - 1)]
    for i in range(num_models - 1):
        args_dict[i] = submodel_D_in
    submodels.append(TwoLayerNet(D_in - ((num_models - 1) * submodel_D_in), H, H_2))
    args_dict[num_models - 1] = D_in - ((num_models - 1) * submodel_D_in)
    submodel_H = H_2 * num_models
    gc_model = GCNet(submodel_H, D_out)

    os.makedirs("./models/MNIST/", exist_ok=True)
    for i in range(len(submodels)):
        torch.save(submodels[i].state_dict(), "./models/MNIST/submodel_{}_statedict".format(i))
    torch.save(gc_model.state_dict(), "./models/MNIST/submodel_{}_statedict".format(len(submodels)))

def prepareConfig():
    config_dict = {
        'validation': {
            'N_train': 60000,
            'N_test': 10000,
        },
        'training_iters': 5,
        'batch_size': 64,
        'random_seed': 0,
        'is_data_aligned': False,
        'models': {
            'gc_model': {
                'ports': {
                    'submodel0': 8600,
                    'submodel1': 8601,
                    'submodel2': 8602
                },
                'checkpoint_dir': "/tmp/gc_model",
                'data_path': f"{FILES_DIR}/files/MNIST/dataset/Y_{{}}.pkl",
                'model_path': "./models/MNIST/submodel_{}_statedict".format(num_models),
                'model_def_path': 'models.gcmodel_def',
                'model_class': 'GCNet',
                'param': {
                    'loss_fn': loss,
                    'model_args': {
                        'D_in': H_2 * num_models,
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
            'data_path': f"{FILES_DIR}/files/MNIST/dataset/X_{{}}_{i}.pkl",
            'model_path': "./models/MNIST/submodel_{}_statedict".format(i),
            'model_def_path': "models.submodel_def",
            'model_class': 'TwoLayerNet',
            'param': {
                'model_args': {
                    'D_in': args_dict[i],
                    'H': 100,
                    'D_out': H_2,
                },
                'optimizer': {
                    'lr': lr,
                    'name': optimizer
                }
            }
        }

    with open('config2.yaml', 'w') as f:
        yaml.dump(config_dict, f)

D_in = 784
H = 100 # H_c
H_2 = 10 # H_s
D_out = 10
num_models = int(sys.argv[1])
loss = sys.argv[2]
optimizer = sys.argv[3]
lr = float(sys.argv[4])
print(num_models)
prepareDatasets()
prepareModels()
prepareConfig()

