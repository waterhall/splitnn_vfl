# Remote utilities
import logging
import os
import shutil
import copy
import pickle
import yaml

class File(object):
    def __init__(self):
        self.exp_path = None
    # Returns a list of file names
    def file_names(self):
        raise NotImplementedError
    def file_paths(self, exp_path):
        return [ os.path.join(exp_path, file_name) for file_name in self.file_names() ]
    def create(self, exp_path):
        self.exp_path = exp_path
    def load(self, exp_path):
        self.exp_path = exp_path
    def transfer(self, exp_path):
        raise NotImplementedError

class PickledFile(File):
    def __init__(self, py_obj, pkl_ref):
        self.pkl_ref = pkl_ref
        self.py_obj = py_obj
    def file_names(self):
        return [ f'{self.pkl_ref}.pkl' ]
    def create(self, exp_path):
        super().create(exp_path)
        with open( self.file_paths(exp_path)[0], "wb" ) as file:
            pickle.dump(self.py_obj, file)
    def transfer(self, exp_path, pkl_ref=None):
        copied_pkl_file = copy.deepcopy(self)
        copied_pkl_file.exp_path = exp_path
        if pkl_ref == None:
            copied_pkl_file.pkl_ref = pkl_ref
        return copied_pkl_file


class ConfigFile(File):
    def __init__(self, file_path):
        self.config_ref = os.path.splitext(file_path)[0]
        self.source_path = file_path
        
        assert(os.path.isfile(file_path))
    # Returns a list of file names
    def file_names(self):
        return [ f'{self.config_ref}.yaml' ]
    def create(self, exp_path):
        super().create(exp_path)
        shutil.copyfile(self.source_path, self.file_paths(exp_path)[0])
    def transfer(self, exp_path, config_ref=None):
        copied_config_file = copy.deepcopy(self)
        copied_config_file.exp_path = exp_path
        if config_ref == None:
            copied_config_file.config_ref = config_ref
        return copied_config_file
    def to_dict(self):
        yml_file_path = self.file_names()[0]
        with open(yml_file_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return config

class FileManager(object):
    def __init__(self, host_ref, files=None, tmp_file=os.path.join('/', 'tmp')):
        self.tmp_file = tmp_file
        self.host_ref = host_ref
        # Create the exp directory
        self.exp_path = os.path.join(self.tmp_file, host_ref)
        self.exp_mkdir()

        if files == None:
            local_pkl_path = os.path.join(self.exp_path, f'{self.host_ref}.pkl')
            logging.info(f"Loaded from {local_pkl_path}")
            with open(local_pkl_path, "rb") as pkl_file:
                self.files = pickle.load(pkl_file)
        else:
            self.files = files
    def exp_mkdir(self):
        raise NotImplementedError
    def exp_cleanup(self):
        raise NotImplementedError

class LocalFileManager(FileManager):
    def create_all(self):
        for file in self.files:
            file.create(self.exp_path)
    def exp_mkdir(self):
        os.makedirs(self.exp_path, exist_ok=True)
    def exp_cleanup(self):
        logging.info(f"Cleaning up {self.exp_path}")
        shutil.rmtree(self.exp_path)

class RemoteFileManager(FileManager):
    def __init__(self, cxn, **kwargs):
        self.cxn = cxn
        super().__init__(files=[], **kwargs)
    def exp_mkdir(self):
        self.cxn.run(f'mkdir -p {self.exp_path}')
    def transfer_from(self, l_fm, file, **kwargs):
        
        local_file_paths = file.file_paths(l_fm.exp_path)
        remote_file = file.transfer(self.exp_path, **kwargs)
        remote_file_paths = remote_file.file_paths(self.exp_path)
        for local_file_path, remote_file_path in zip(local_file_paths, remote_file_paths):
            logging.info(f"{local_file_path} to {self.cxn.host}:{remote_file_path}")
            self.cxn.put(local_file_path, remote=remote_file_path)
        self.files.append(remote_file)
    def end_transfer(self, l_fm):
        # Transfer the pkl
        local_pkl_path = os.path.join(l_fm.exp_path, f'{self.host_ref}.pkl')
        remote_pkl_path = os.path.join(self.exp_path, f'{self.host_ref}.pkl')
        with open(local_pkl_path, "wb" ) as pkl_file:
            pickle.dump(self.files, pkl_file)
        self.cxn.put(local_pkl_path, remote=remote_pkl_path)


