# CS6203 Project

Project brief - Split Neural Networks Learning in VFL: 
 - Code mainly in Python, following Google Python style. Code sanity checks are needed.
 - Use Pytorch, Singa, or TensorFlow as local training framework. Define local training framework abstract such that anyone can change it easily through configuration.
 - Make each party’s local NN architecture configurable such that the parties’ local model architectures can be different.
 - Apply partially homomorphic encryption for secure aggregation on the cut layer in the split learning approach.

## Setup 

Our system comes pre-configured for your convenience. There is no setup required on your end.

## Pre-configured Demo

We have configured a demo suite of test cases to showcase the different privacy techniques that our system can be configured to run.

We have 3 VMs setup for this demo, we will refer to this set as our cluster.
1. 34.67.88.74 - This is the master node, we have setup everything you need here.
2. 34.72.215.92 - This is Worker Node 1
3. 34.122.10.199 - This is Worker Node 2

You can see the reference to the nodes from our yaml configurations.
You can access 34.67.88.74 by running `sh connect-gcp.sh`

## Manual Setup

Tested only on Linux environments.

> You do not have to perform this as we have preconfigured the environment for your convenience.

Here are some brief notes if you wish to configure the system manually on your local machine:
1. [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
2. Create the cs6203 environment `conda env create -f cs6203.yml`
3. Activate the envionment `conda activate cs6203`
4. Install the datasets `sh generate_all.sh`

## Using our system

Here we will discuss running our demo and tests.

### Running our Demo

1. Access server 34.67.88.74 by running `sh connect-gcp.sh`
2. Once there, navigate to the code directory by `cd CS6203_proj`
3. Run our demo by running `sh run_demo.sh`, what it does is run all of our models as per our report on MNIST, each with the random seeds - 0,1,2. All of the models are run with 250 iterations, except smpc-network, pe-network and tpe-network with only 25 iterations. This is due to the slow running of pe and tpe models.
    1. `une-pytorch`: Unencrypted model using Pytorch framework to train and test directly
    1. `une-framework` - Unencrypted model using our framework locally without networking
    1. `une-network` - Unecrypted model over the network
    1. `smpc-network` - SMPC model over the network
    1. `pe-network` - PE model over the network
    1. `tpe-network` - TPE model over the network

> Note: The full demo will take about 3 hours to run.
> Do not stop the experiments half way, it will affect the sync between the nodes.

You can review previous runs, for example:
1. [MNIST-tpe-ba-networked](http://34.67.88.74:5000/mlflow/#/experiments/0/runs/b54bb0cd36d64f7fa064274f68769d6c)
1. [MNIST-pe-ba-networked](http://34.67.88.74:5000/mlflow/#/experiments/0/runs/61761a777c974d67884374bd8ebe232c)

You will be able to view the experiments running in real time on the MLflow tracking page here - [http://34.67.88.74:5000/mlflow/#/experiments/0](http://34.67.88.74:5000/mlflow/#/experiments/0).

### Running our tests

Our system is tested with the following tests written using pytorch, in the respective files:
1. `test_encryption.py` - Test our ThresholdPaillier implementation.
1. `test_pe_ser_deser.py` - Test serialization and deserialization of keys and pytorch tensors.
1. `test_psi.py` - Test Private Set Intersection (PSA-PSI).

You can run these test by the command `pytest test_encryption.py test_pe_ser_deser.py test_psi.py`

## Code 

We follow standard python code conventions, as much as possible.
We commented on parts that are non-trivial; the self-explanatory parts are left uncommented.
We use an object oriented approach to software design.

For example, our parent classes for networked models are as follows:

```
class NetworkedSubModel(torch.nn.Module):
    def setup(self, config_file, gc_conn, tp_conn):
        logging.info("Default Networked Model")
    def serialize(self, tensor):
        return tensor

class NetworkedGCModel(torch.nn.Module):
    def setup(self, config_file, gc_conn, tp_conn):
        logging.info("Default Networked Model")
    def deserialize(self, ser_tensor):
        return ser_tensor
    def cat(self, tensor_lists):
        return torch.cat(tensor_lists, 1).detach().clone()
    def grad(self, input_tensor):
        return input_tensor.grad.detach().clone()

class UnsecureTP(object):
    def __init__(self, **kwargs):
        pass
    def setup(self, config_file, gc_conn, tp_conn):
        logging.info(f"No Setup")
    def distribute_keys(self, config_file, target_conn):
        logging.info(f"No need to distribute keys")
    def decrypt(self, config_file, target_conn):
        logging.info(f"No need to decrypt")
```

The child classes would override the relevant methods to implement additional functionality.
This would allow anyone who wish to extend our code to do so with ease.

### Acknowledgements

We built our system with code from these sources, the following are packaged together with out code.

1. [https://coderzcolumn.com/tutorials/python/threshold-paillier](https://coderzcolumn.com/tutorials/python/threshold-paillier): Modified the code to work with pytorch and phe (`tpe.layers.py`)
1. [https://github.com/OpenMined/PyPSI](https://github.com/OpenMined/PyPSI): Packaged together with our code in `psi` for convenience.


The rest of the sources are as included inside the conda environment file `vfl.yml`
