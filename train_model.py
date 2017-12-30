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

class TrainParameter(argschema.ArgSchema):
    batch_size = Int(required=False, default=24,description="size of each batch")
    buffer_size = Int(required=False,default=5,description="number of images to cache in memory")
    path_train_csv = InputFile(required=True,description="path to train set csv")
    path_test_csv = InputFile(required=True,description="path to test set csv")
    gpu_ids = List(Int, required=False, default = [0], description = "GPU IDs")
    iter_checkpoint = Int(required=False, default = 500, description="iterations between saving log/model checkpoints")
    lr = Float(required=False, default =0.001,description="learning rate")
    model_module = Str(required=False, default ="fnet_model", help = "name of the model module")
    n_iter = Int(required=False, default=500, description="number of training iterations")
    scale_z = Float(default=.3, help = "desired um/px scale for z dimension")
    scale_xy = Float(default=.3, help = "desired um/px scale for x, y dimensions")
    transforms_signal = List(Str, default=['fnet.data.sub_mean_norm'],
        description='transform to be applied to signal images')
    transforms_target = List(Str, default=['fnet.data.sub_mean_norm'],
        description='transform to be applied to target images')
    no_checkpoint_testing = Boolean(default=False,description='set to disable testing at checkpoints')
    nn_module = Str(default='ttf_v8_nn', description= 'name of neural network module')
    replace_interval = Int(default=-1, description = 'iterations between replacements of images in cache' )
    path_run_dir = OutputDir(default = 'saved_models', description = 'base directory for saved models')
    seed = Int(description = "random seed")
    name_dataset_module = Str(default = 'fnet.data.dataset', description = 'name of dataset module with DataSet class')
    choices_augmentation = List(Int, default=[], description="list of augmentation choices")


def main():
    time_start = time.time()
    opts = argschema.ArgSchemaParser(schema_type=TrainParameter)

    model_module = importlib.import_module('model_modules.' + opts.args['model_module'])
    
    if not os.path.exists(opts.args['path_run_dir']):
        os.makedirs(opts.args['path_run_dir'])

    logger = logging.getLogger('model training')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(opts.args['path_run_dir'], 'run.log'), mode='a')
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)
    warnings.showwarning = lambda *args, **kwargs : logger.warning(warnings.formatwarning(*args, **kwargs))

    main_gpu_id = opts.args['gpu_ids'] if isinstance(opts.args['gpu_ids'], int) else opts.args['gpu_ids'][0]
    torch.cuda.set_device(main_gpu_id)
    logger.info('main GPU ID: {:d}'.format(torch.cuda.current_device()))

    if opts.args['seed'] is not None:
        np.random.seed(opts.args['seed'])
        torch.manual_seed(opts.args['seed'])
        torch.cuda.manual_seed_all(opts.args['seed'])

    model = model_module.Model(
        nn_module=opts.args['nn_module'],
        lr=opts.args['lr'],
        gpu_ids=opts.args['gpu_ids'],
    )
    path_model = os.path.join(opts.args['path_run_dir'], 'model.p')
    if os.path.exists(path_model):
        model.load_state(path_model)
        logger.info('model loaded from: {:s}'.format(path_model))
    logger.info(model)
    
    path_losses_csv = os.path.join(opts.args['path_run_dir'], 'losses.csv')
    df_losses = pd.DataFrame()
    if os.path.exists(path_model):
        df_losses = pd.read_csv(path_losses_csv)
        
    path_ds = os.path.join(opts.args['path_run_dir'], 'ds.json')
    if not os.path.exists(path_ds):
        path_train_csv_copy = os.path.join(opts.args['path_run_dir'], os.path.basename(opts.args['path_train_csv']))
        path_test_csv_copy = os.path.join(opts.args['path_run_dir'], os.path.basename(opts.args['path_test_csv']))
        if not os.path.exists(path_train_csv_copy):
            shutil.copyfile(opts.args['path_train_csv'], path_train_csv_copy)
        if not os.path.exists(path_test_csv_copy):
            shutil.copyfile(opts.args['path_test_csv'], path_test_csv_copy)
        fnet.data.save_dataset_as_json(
            path_train_csv = path_train_csv_copy,
            path_test_csv = path_test_csv_copy,
            scale_z = opts.args['scale_z'] if opts.args['scale_z'] > 0 else None,
            scale_xy = opts.args['scale_xy'] if opts.args['scale_xy'] > 0 else None,
            transforms_signal = opts.args['transforms_signal'],
            transforms_target = opts.args['transforms_target'],
            path_save = path_ds,
            name_dataset_module = opts.args['name_dataset_module'],
        )
    dataset = fnet.load_dataset_from_json(
        path_load = path_ds,
    )
    logger.info(dataset)

    data_provider = fnet.data.ChunkDataProvider(
        dataset,
        buffer_size=opts.args['buffer_size'],
        batch_size=opts.args['batch_size'],
        replace_interval=opts.args['replace_interval'],
        choices_augmentation=opts.args['choices_augmentation'],
    )

    data_provider_nonchunk = None
    if not opts.args['no_checkpoint_testing']:
        dims_cropped = (32, '/16', '/16')
        cropper = fnet.transforms.Cropper(dims_cropped, offsets=('mid', 0, 0))
        transforms_nonchunk = (cropper, cropper)
        data_provider_nonchunk = fnet.data.TestImgDataProvider(
            dataset,
            transforms=transforms_nonchunk,
        )
    
    with open(os.path.join(opts.args['path_run_dir'], 'train_options.json'), 'w') as fo:
        json.dump(vars(opts), fo, indent=4, sort_keys=True)

    for i in range(model.count_iter, opts.args['n_iter']):
        x, y = data_provider.get_batch()
        l2_batch = model.do_train_iter(x, y)
        
        logger.info('num_iter: {:4d} | l2_batch: {:.4f} | sources: {:s}'.format(i + 1, l2_batch, data_provider.last_sources))
        dict_iter = dict(
            num_iter = i + 1,
            l2_batch = l2_batch,
            sources = data_provider.last_sources,
        )
        df_losses_curr = pd.concat([df_losses, pd.DataFrame([dict_iter])], ignore_index=True)
        if ((i + 1) % opts.args['iter_checkpoint'] == 0) or ((i + 1) == opts.args['n_iter']):
            if data_provider_nonchunk is not None:
                # path_checkpoint_dir = os.path.join(path_run_dir, 'output_{:05d}'.format(i + 1))
                path_checkpoint_dir = os.path.join(opts.args['path_run_dir'], 'output')
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
                df_losses_curr = pd.concat([df_losses, pd.DataFrame([dict_iter])], ignore_index=True)
            model.save_state(path_model)
            df_losses_curr.to_csv(path_losses_csv, index=False)
            logger.info('model saved to: {:s}'.format(path_model))
            logger.info('elapsed time: {:.1f} s'.format(time.time() - time_start))
        df_losses = df_losses_curr

    logger.info('total training time: {:.1f} s'.format(time.time() - time_start))

    
if __name__ == '__main__':
    main()
