import torch
from functools import reduce
import pandas as pd
import numpy as np
import importlib
import os
import pickle
from sklearn.metrics import roc_curve, auc, roc_auc_score
import logging
import time
import mlflow

# Transfer 10MiB
MAX_SIZE = 1*1024**3
RW_BUFFER = 2**24
MAX_Q = 1024
PING_TIMEOUT = 240
CLOSE_TIMEOUT = 200

class CombinedModel(torch.nn.Module):
    def __init__(self, gc_model, sub_models):
        super(CombinedModel, self).__init__()
        self.gc_model = gc_model
        self.sub_models = sub_models
    def forward(self, sub_xs):
        submodel_outputs = []
        for sub_model, sub_x in zip(self.sub_models, sub_xs):
            submodel_outputs.append(sub_model(sub_x))
        combined_output = torch.cat(submodel_outputs, 1)
        y_pred = self.gc_model(combined_output)
        return y_pred
    def parameters(self):
        param_list = [ list(super().parameters()) ] + [ list(sub_m.parameters()) for sub_m in self.sub_models ]
        return reduce(list.__add__, param_list)

class VFLModel(object):
    def __init__(self, gc_model, sub_models):
        # Initalize the GC 
        self.gc_model = GCModel(**gc_model)
        
        # This is a dict of all the models
        self.sub_models_dict = { sub_model : SubModel(**(sub_models[sub_model])) for sub_model in sub_models }
        self.sub_models_seq  = list(self.sub_models_dict.values())

    def align(self, train_ids, test_ids):
        self.gc_model.align(train_ids, test_ids)
        for submodel in self.sub_models_seq:
            submodel.align(train_ids, test_ids)

    def train(self, batch_ids):
        # make each submodel work
        pred_xs = [subm.train(batch_ids) for subm in self.sub_models_seq]
        submodel_outputs = []
        for pred_x in pred_xs:
            output = pred_x.detach().clone()
            submodel_outputs.append(output)
        combined_output = torch.cat(submodel_outputs, 1)
        combined_output.requires_grad = True

        # send combined output to gc and let gc work
        loss = self.gc_model.train(combined_output, batch_ids)
        self.gc_model.backprop(loss)

        combined_grad = combined_output.grad.detach().clone()

        # client backprop
        subtensor_size = int(combined_grad.size()[1] / len(self.sub_models_seq))
        submodel_grads = torch.split(combined_grad, subtensor_size, dim=1)

        for submodel_grad, pred_x, submodel in zip(submodel_grads, pred_xs, self.sub_models_seq):
            submodel.backprop(pred_x, submodel_grad)

        return loss.item()
    def predict(self, x=None):
        submodel_outputs = [subm.predict(x) for subm in self.sub_models_seq]
        combined_output = torch.cat(submodel_outputs, 1)
        return self.gc_model.predict(x, combined_output)

    def to_pytorch(self):
        return CombinedModel(self.gc_model.model, [ subm.model for subm in self.sub_models_seq ])

class NetworkedModel(object):
    #TODO
    def __init__(self, checkpoint_dir, data_path, model_path, model_def_path, model_class, param, model_ref=None, gc_conn=None, tp_conn=None):
        logging.debug("NetworkedModel", checkpoint_dir, data_path, model_path, model_def_path, param, model_ref, gc_conn, tp_conn)
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model = self.loadModel(model_path, model_def_path, model_class, param)
        self.param = param
        self.optimizer = self.loadOptimizer(self.model.parameters())
        # self.data_train, self.data_test = self.loadData(data_path)
        self.data = Dataset(data_path, checkpoint_dir)

        self.model_ref = model_ref
        self.gc_conn = gc_conn
        self.tp_conn = tp_conn

    def loadOptimizer(self, model_params):
        return (getattr(torch.optim, self.param['optimizer']['name'])
            (model_params, lr=self.param['optimizer']['lr']))

    def loadModel(self, model_path, model_def_path, model_class, param):
        print(model_def_path)
        module = importlib.import_module(model_def_path)
        cls = getattr(module, model_class)
        model = cls(**param['model_args'])
        if model_path != None:
            model.load_state_dict(torch.load(model_path))
        return model

    def align(self, train_ids, test_ids):
        self.data.align(train_ids, test_ids)

    def train_networked(self, config_file):
        # This is where we need to prepare the HE setup
        self.model.setup(config_file, self.gc_conn, self.tp_conn)

        
import asyncio
import threading
import random
import websockets
import queue
from pickle import dumps, loads
import torch
import sys

class GCModel(NetworkedModel):
    def __init__(self, ports, **kwargs):
        logging.debug(kwargs)
        super().__init__(**kwargs)
        self.ports = ports
        self.loss_fn = getattr(torch.nn, kwargs['param']['loss_fn'])()

    def parameters(self):
        return self.model.parameters()

    def train(self, combined_output_from_subs, batch_ids):
        if type(self.loss_fn) == torch.nn.modules.loss.BCELoss:  # if BCE
            y = torch.from_numpy(self.data.getData(batch_ids, train=True, dtype=np.float32))
        else:  # if crossentropyloss
            y = torch.from_numpy(self.data.getData(batch_ids, train=True, dtype=np.int64).squeeze())
        y_pred = self.model(combined_output_from_subs)
        loss = self.loss_fn(y_pred, y)
        return loss

    def backprop(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, x, combined_output_from_subs):
        with torch.no_grad():
            metric = {}
            outputs = self.model(combined_output_from_subs)
            if x is not None:
                return outputs
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == torch.from_numpy(self.data.test.to_numpy(dtype=np.int64).squeeze())).sum().item()
            accuracy = correct / len(self.data.test.to_numpy(dtype=np.int64).squeeze())
            metric['accuracy'] = accuracy
            logging.debug(f"Test Accuracy: {accuracy}")
            if type(self.loss_fn) == torch.nn.modules.loss.BCELoss:
                auc = roc_auc_score(self.data.test.to_numpy(dtype=np.int64), outputs)
                metric['roc_auc_score'] = auc
                logging.debug(f"roc_auc_score: {auc}")
            return outputs, metric
    def train_networked(self, config_file):

        super().train_networked(config_file)

        # At this point the data must be aligned
        assert(self.data.is_aligned())
        ports = config_file['models']['gc_model']['ports']
        ssl_context, _, _ = self.gc_conn

        training_iters = config_file['training_iters']
        batch_size = config_file['batch_size']
        np.random.seed(config_file['random_seed'])

        train_ids, test_ids = self.data.get_ids()
        
        N_train = len(train_ids)
        indices = np.array(list(train_ids))
        np.random.shuffle(indices)
        batch_i = 0

        model_refs = list(sorted(self.ports.keys()))
        n_submodels = len(model_refs)

        def serve_submodels(ssl_context, port, out_q, grad_q):
            # we need to create a new loop for the thread, and set it as the 'default'
            # loop that will be returned by calls to asyncio.get_event_loop() from this
            # thread.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            stop = loop.create_future()

            async def comm_submodel(websocket, path):

                logging.info(f"Subm connected to {port}")

                for i in range(training_iters):
                    submodel_output = await websocket.recv()
                    submodel_output = loads(submodel_output)
                    out_q.put(submodel_output)
                    logging.debug(f"recv {port} < {submodel_output}")

                    submodel_grad = grad_q.get(block=True)

                    await websocket.send(dumps(submodel_grad))
                    logging.debug(f"send {port} > {submodel_grad}")

                stop.set_result(None)

            async def comms_server(stop):
                async with websockets.serve(comm_submodel, '0.0.0.0', port, ssl=ssl_context, max_size=MAX_SIZE, 
                read_limit=RW_BUFFER, write_limit=RW_BUFFER, max_queue=MAX_Q, ping_timeout=PING_TIMEOUT):
                    await stop
                    return stop

            a = loop.run_until_complete(comms_server(stop))

        output_q_map = { model_ref : queue.Queue() for model_ref in model_refs }
        grad_q_map = { model_ref : queue.Queue() for model_ref in model_refs }

        # https://stackoverflow.com/questions/38804988/what-does-sys-exit-really-do-with-multiple-threads
        threads = [ threading.Thread(target = serve_submodels, args=(ssl_context, self.ports[model_ref], output_q_map[model_ref], grad_q_map[model_ref]), daemon=True) for model_ref in model_refs ]
        for thread in threads:
            thread.start()

        try:
            for t in range(training_iters):

                prev_batch_i = batch_i
                batch_i = (t * batch_size) % N_train
                # check for reset!
                if prev_batch_i > batch_i:
                    logging.debug("Protocol - Shuffle together")
                    # Shuffle together
                    np.random.shuffle(indices)

                batch_ids = indices[batch_i : batch_i + batch_size]
                logging.debug(f"Iteration {t} - Data={hash(tuple(batch_ids))}")

                # This is the computation from the submodel
                train_submodel_outputs = []
                test_submodel_outputs = []
                for model_ref in model_refs:
                    train_submodel_output, test_submodel_output = output_q_map[model_ref].get(block=True)
                    train_submodel_output = self.model.deserialize(train_submodel_output)
                    test_submodel_output = self.model.deserialize(test_submodel_output)

                    train_submodel_outputs.append(train_submodel_output)
                    test_submodel_outputs.append(test_submodel_output)
                
                train_combined_output = self.model.cat(train_submodel_outputs)
                test_combined_output = self.model.cat(test_submodel_outputs)

                train_combined_output.requires_grad = True

                # send combined output to gc and let gc work
                loss = self.train(train_combined_output, batch_ids)
                y_pred, metric = self.predict(None, test_combined_output)
                logging.info(f"Iteration {t} - Loss {loss.item()} - Metric {metric}")
                # Log to mlflow
                metric['loss'] = loss.item()
                mlflow.log_metrics(metric, step=t)

                self.backprop(loss)

                combined_grad = self.model.grad(train_combined_output)

                # client backprop
                subtensor_size = int(combined_grad.size()[1] / n_submodels)
                submodel_grads = torch.split(combined_grad, subtensor_size, dim=1)

                # Now we need to send the gradients
                for model_ref, submodel_grad in zip(model_refs, submodel_grads):
                    grad_q_map[model_ref].put(submodel_grad)

        except Exception as e:
            logging.exception("Unexpected error in GC")
            # Stop all loops
            sys.exit()

        # Should not hang unless there is a bug
        # Wait for termination
        for thread in threads:
            thread.join()


class SubModel(NetworkedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parameters(self):
        return self.model.parameters()

    def train(self, batch_ids):
        # x = torch.from_numpy(self.data_train[batch_i: (batch_i) + N])
        x = torch.from_numpy(self.data.getData(batch_ids, train=True, dtype=np.float32))
        return self.model(x)

    def backprop(self, pred_x, grad):
        self.optimizer.zero_grad()
        pred_x.backward(grad)
        self.optimizer.step()

    def predict(self, x=None):
        with torch.no_grad():
            if x is None:
                return self.model(torch.from_numpy(self.data.test.to_numpy(dtype=np.float32)))
            else:
                return self.model(x)

    async def train_task(self, config_file, target_conn):

        ssl_context, hostname, port = target_conn

        training_iters = config_file['training_iters']
        batch_size = config_file['batch_size']
        np.random.seed(config_file['random_seed'])

        train_ids, test_ids = self.data.get_ids()
        
        N_train = len(train_ids)
        indices = np.array(list(train_ids))
        np.random.shuffle(indices)
        batch_i = 0

        uri = f"wss://{hostname}:{port}"
        async with websockets.connect(uri, ssl=ssl_context, max_size=MAX_SIZE, 
            read_limit=RW_BUFFER, write_limit=RW_BUFFER, max_queue=MAX_Q, ping_timeout=PING_TIMEOUT) as websocket:

            for t in range(training_iters):
            
                prev_batch_i = batch_i
                batch_i = (t * batch_size) % N_train
                # check for reset!
                if prev_batch_i > batch_i:
                    logging.debug("Protocol - Shuffle together")
                    # Shuffle together
                    np.random.shuffle(indices)

                batch_ids = indices[batch_i : batch_i + batch_size]
                logging.debug(f"Iteration {t} - Data={hash(tuple(batch_ids))}")

                # Need to transmit this to GC, 
                # Predict piggy back
                model_out = self.train(batch_ids)
                predict_out = self.predict()
                model_out_ser = self.model.serialize(model_out)
                predict_out_ser = self.model.serialize(predict_out)
                await websocket.send(dumps((model_out_ser, predict_out_ser)))

                submodel_grad = await websocket.recv()
                submodel_grad = loads(submodel_grad)
                self.backprop(model_out, submodel_grad)

                metric = {
                    'iter': t
                }
                mlflow.log_metrics(metric, step=t)

    def train_networked(self, config_file):

        super().train_networked(config_file)
        
        # Further ensure that the submodels start later than the gc
        time.sleep(5)

        # At this point the data must be aligned
        assert(self.data.is_aligned())
        asyncio.get_event_loop().run_until_complete(self.train_task(config_file, self.gc_conn))

class Dataset(object):
    def __init__(self, data_path, checkpoint_dir):
        self.train, self.test = self.read(data_path)
        self.checkpoint_dir = checkpoint_dir
        self.align_ids_chk_path = os.path.join(checkpoint_dir, f"align_ids.chkpt")

        if self.is_aligned():
            logging.info(f"Loading Aligned ID checkpoints found at {self.align_ids_chk_path}")
            with open(self.align_ids_chk_path, 'rb') as handle:
                train_ids, test_ids = pickle.load(handle)
            logging.info(f"IDs - {len(train_ids)}, {len(test_ids)}")
            self.train = self.train.loc[train_ids, :].sort_index()
            self.test = self.test.loc[test_ids, :].sort_index()

    def read(self, data_path):
        logging.info(f"Loading data from {data_path.format('train')}, {data_path.format('test')}")
        return pd.read_pickle(data_path.format('train')), pd.read_pickle(data_path.format('test'))

    def align(self, train_ids, test_ids):
        if self.is_aligned():
            logging.warning("Dataset is already aligned.")
            return
        # Write to directory
        self.train = self.train.loc[train_ids, :].sort_index()
        self.test = self.test.loc[test_ids, :].sort_index()
        with open(self.align_ids_chk_path, 'wb') as handle:
            pickle.dump((train_ids, test_ids), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def getData(self, ids, train, dtype):
        if train:
            return self.train.loc[ids, :].to_numpy(dtype=dtype)
        else:
            return self.test.loc[ids, :].to_numpy(dtype=dtype)

    def get_ids(self):
        return set(self.train.index.tolist()), set(self.test.index.tolist())

    def is_aligned(self):
        return os.path.exists(self.align_ids_chk_path)

class TrustedParty(NetworkedModel):
    def __init__(self, tp_def_path, tp_class, param, ports, model_ref, gc_conn, tp_conn, **kwargs):
        self.ports = ports
        self.data = None
        module = importlib.import_module(tp_def_path)
        Cls = getattr(module, tp_class)
        self.tp_impl = Cls(**param)
        self.model_ref = model_ref
        self.gc_conn = gc_conn
        self.tp_conn = tp_conn
    def train_networked(self, config_file):
        # Things like key distribution
        self.tp_impl.setup(config_file, self.gc_conn, self.tp_conn)

        threads = []
        # https://stackoverflow.com/questions/38804988/what-does-sys-exit-really-do-with-multiple-threads
        for model_ref, port in self.ports.items():
            target_tp_conn = self.tp_conn[:-1] + (port,)

            if model_ref == "gc_model":
                threads.append(threading.Thread(target = self.tp_impl.decrypt, args=(config_file, target_tp_conn), daemon=True))
            else:
                threads.append(threading.Thread(target = self.tp_impl.distribute_keys, args=(config_file, target_tp_conn), daemon=True))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        logging.info("Completed.")