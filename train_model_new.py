import os
import argparse
import importlib
import fnet
import fnet.data
import fnet.transforms
import pandas as pd
import numpy as np
import torch
import pdb
import time
import logging
import sys
import shutil
import json
import warnings
import argschema
from argschema.fields import Int, Str,InputFile, Float, List, Boolean, OutputDir
from marshmallow import post_load


class DataSetParameters(argschema.schemas.DefaultSchema):
    path_save = OutputFile(required=False, description = "path to save training set (default")

class CziDataSetParameters(DataSetParameters):
    path_train_csv = InputFile(required=True,description="path to train set csv")
    path_test_csv = InputFile(required=True,description="path to test set csv")
    scale_z = Float(default=.3, help = "desired um/px scale for z dimension")
    scale_xy = Float(default=.3, help = "desired um/px scale for x, y dimensions")
    transforms_signal = List(Str, default=['fnet.data.sub_mean_norm'],
        description='transform to be applied to signal images')
    transforms_target = List(Str, default=['fnet.data.sub_mean_norm'],
        description='transform to be applied to target images')
    
class TrainParameter(argschema.ArgSchema):


    batch_size = Int(required=False, default=24,description="size of each batch")
    buffer_size = Int(required=False,default=5,description="number of images to cache in memory")

    gpu_ids = List(Int, required=False, default = [0], description = "GPU IDs")
    iter_checkpoint = Int(required=False, default = 500, description="iterations between saving log/model checkpoints")
    lr = Float(required=False, default =0.001,description="learning rate")
    model_module = Str(required=False, default ="fnet_model", help = "name of the model module")
    n_iter = Int(required=False, default=500, description="number of training iterations")

    no_checkpoint_testing = Boolean(default=False,description='set to disable testing at checkpoints')
    nn_module = Str(default='ttf_v8_nn', description= 'name of neural network module')
    replace_interval = Int(default=-1, description = 'iterations between replacements of images in cache' )
    path_run_dir = OutputDir(default = 'saved_models', description = 'base directory for saved models')
    seed = Int(description = "random seed")
    name_dataset_class = Str(default = 'fnet.data.czidataset.CziDataSet', description = 'name of dataset module with DataSet class')
    choices_augmentation = List(Int, default=[], description="list of augmentation choices")

    dataset = Nested(CziDataSetParameters, description="parameters to control the dataset")

    @post_load
    def add_missing_values(self, data):
        if data['dataset']['path_save'] is None:
            data['dataset']['path_save'] = os.path.join(data['path_run_dir'], 'model.p')

def checkpoint_model(dict_iter, model, data_provider_nonchunk, path_checkpoint_dir):
    data_provider_nonchunk.use_train_set()
    dict_iter.update(fnet.test_model(
        model,
        data_provider_nonchunk,
        n_images = 4,
        path_save_dir = path_checkpoint_dir,
    )[0])
    data_provider_nonchunk.use_test_set()
    dict_iter.update(fnet.test_model(
        model,
        data_provider_nonchunk,
        n_images = 4,
        path_save_dir = path_checkpoint_dir,
    )[0])
    return dict_iter

def train_model(model,
                data_provider,
                n_iter,
                iter_checkpoint,
                path_checkpoint_dir,
                logger,
                data_provider_noncheck=None):

    for i in range(model.count_iter, n_iter):
        x, y = data_provider.get_batch()
        l2_batch = model.do_train_iter(x, y)
        
        logger.info('num_iter: {:4d} | l2_batch: {:.4f} | sources: {:s}'.format(i + 1, l2_batch, data_provider.last_sources))
        dict_iter = dict(
            num_iter = i + 1,
            l2_batch = l2_batch,
            sources = data_provider.last_sources,
        )
        df_losses_curr = pd.concat([df_losses, pd.DataFrame([dict_iter])], ignore_index=True)
        if ((i + 1) % iter_checkpoint == 0) or ((i + 1) == n_iter):
            if data_provider_nonchunk is not None:
                dict_iter = checkpoint_model(dict_iter,model,data_provider_noncheck,path_checkpoint_dir)
                df_losses_curr = pd.concat([df_losses, pd.DataFrame([dict_iter])], ignore_index=True)
                
            model.save_state(path_model)
            df_losses_curr.to_csv(path_losses_csv, index=False)
            logger.info('model saved to: {:s}'.format(path_model))
            logger.info('elapsed time: {:.1f} s'.format(time.time() - time_start))
        df_losses = df_losses_curr


class TrainModel(argschema.ArgSchemaParser):
    default_schema = TrainParameter

    def __init__(*args,**kwargs):
        super(self,TrainModel).__init__(*args,**kwargs)

        fh = logging.FileHandler(os.path.join(self.args['path_run_dir'], 'run.log'), mode='a')
        sh = logging.StreamHandler(sys.stdout)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        warnings.showwarning = lambda *args, **kwargs : self.logger.warning(warnings.formatwarning(*args, **kwargs))
        
        self.model_module = importlib.import_module('model_modules.' + self.args['model_module'])
        self.DataSetClass = importlib.import_module(self.args["name_dataset_class"])
        self.setup_gpu()
        self.setup_random_seed()

    def setup_random_seed(self):
        seed = self.args.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def setup_directory(self):
        if not os.path.exists(self.args['path_run_dir']):
            os.makedirs(opts.args['path_run_dir'])

    def setup_gpu(self):
        main_gpu_id = opts.args['gpu_ids'] if isinstance(opts.args['gpu_ids'], int) else opts.args['gpu_ids'][0]
        torch.cuda.set_device(main_gpu_id)
        self.logger.info('main GPU ID: {:d}'.format(torch.cuda.current_device()))

    def initialize_model(self):
        self.model = self.model_module.Model(
            nn_module=opts.args['nn_module'],
            lr=opts.args['lr'],
            gpu_ids=opts.args['gpu_ids'])
        path_model = os.path.join(self.args['path_run_dir'], 'model.p')
        if os.path.exists(path_model):
            self.model.load_state(path_model)
            logger.info('model loaded from: {:s}'.format(path_model))
        logger.info(self.model)

        path_losses_csv = os.path.join(opts.args['path_run_dir'], 'losses.csv')
        self.df_losses = pd.DataFrame()
        if os.path.exists(path_model):
            self.df_losses = pd.read_csv(path_losses_csv)
       
    def load_dataset(): 
        self.dataset=self.DataSetClass(**self.args['dataset'])
        self.dataset.save_dataset_as_json()
        self.logger(self.dataset)
        self.data_provider = fnet.data.ChunkDataProvider(
                self.dataset,
                buffer_size=self.args['buffer_size'],
                batch_size=self.args['batch_size'],
                replace_interval=self.args['replace_interval'],
                choices_augmentation=self.args['choices_augmentation'])
        
        self.data_provider_nonchunk = None
        if not self.args['no_checkpoint_testing']:
            dims_cropped = (32, '/16', '/16')
            cropper = fnet.transforms.Cropper(dims_cropped, offsets=('mid', 0, 0))
            transforms_nonchunk = (cropper, cropper)
            self.data_provider_nonchunk = fnet.data.TestImgDataProvider(
                self.dataset,
                transforms=transforms_nonchunk,
            )

    def save_module_parameters(self,filepath):
        with open(filepath, 'w') as fo:
            json.dump(self.args, fo, indent=4, sort_keys=True)

    def run():
        time_start = time.time()
        self.initialize_model()
        self.load_dataset()
        self.save_module_parameters(os.path.join(self.args['path_run_dir'], 'train_options.json'))
        path_checkpoint_dir = os.path.join(self.args['path_run_dir'], 'output')
        self.model = train_model(model, 
                                 self.data_provider,
                                 self.args['n_iter'],
                                 self.args['iter_checkpoint'],
                                 path_checkpoint_dir,
                                 self.logger,
                                 self.data_provider_nonchunk)

def main():
    
    mod = argschema.ArgSchemaParser(schema_type=TrainParameter)
    mod.run()
    
    
    

    
    


    logger.info('total training time: {:.1f} s'.format(time.time() - time_start))

    
if __name__ == '__main__':
    main()
