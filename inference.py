#!/usr/bin/env python
# encoding: utf-8

import os, platform
import yaml
import torch
import datetime
import importlib
import numpy as np
import pytorch_lightning as pl

from easydict import EasyDict
from argparse import ArgumentParser
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataloader.dataset import get_model_class, get_collate_class
from dataloader.pc_dataset import get_pc_model_class
from pytorch_lightning.callbacks import LearningRateMonitor

from utils.vis_utils import draw_points_image_labels

import warnings
warnings.filterwarnings("ignore")


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = ArgumentParser()

    # general
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,), help='specify gpu devices')
    parser.add_argument('--config_path', default='config/2DPASS-semantickitti.yaml')
    # inference
    parser.add_argument('--baseline_only', action='store_true', default=False, help='training without 2D')
    parser.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')
    parser.add_argument('--submit_to_server', action='store_true', default=False, help='submit on benchmark') # IGNORE
    parser.add_argument('--outputs', type=str, default='./outputs', help='outputs location')
    # debug
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    config = load_yaml(args.config_path)
    config.update(vars(args))  # override the configuration using the value in args

    return EasyDict(config)


def build_loader(config):
    # Sets up class templates
    pc_dataset = get_pc_model_class(config['dataset_params']['pc_dataset_type'])
    dataset_type = get_model_class(config['dataset_params']['dataset_type'])
    val_config = config['dataset_params']['val_data_loader']
    test_dataset_loader = None

    # For testing
    test_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='test', num_vote=val_config["batch_size"])
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset_type(test_pt_dataset, config, val_config, num_vote=val_config["batch_size"]),
        batch_size=val_config["batch_size"],
        collate_fn=get_collate_class(config['dataset_params']['collate_type']),
        shuffle=val_config["shuffle"],
        num_workers=val_config["num_workers"]
    )

    return test_dataset_loader

# ##############
# def build_dataset(config):
#     # Sets up class templates
#     pc_dataset = get_pc_model_class(config['dataset_params']['pc_dataset_type'])
#     dataset_type = get_model_class(config['dataset_params']['dataset_type'])
#     val_config = config['dataset_params']['val_data_loader']

#     test_pt_dataset = pc_dataset(config, data_path=val_config['data_path'], imageset='test', num_vote=val_config["batch_size"])
#     dataset = dataset_type(test_pt_dataset, config, val_config, num_vote=val_config["batch_size"])
#     return dataset
# ##############


if __name__ == '__main__':
    # parameters
    configs = parse_config()
    print(configs)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, configs.gpu))
    num_gpu = len(configs.gpu)

    if platform.system() == "Windows":
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

    # output path
    # log_folder = 'logs/' + configs['dataset_params']['pc_dataset_type']
    # tb_logger = pl_loggers.TensorBoardLogger(log_folder, name=configs.log_dir, default_hp_metric=False)
    # os.makedirs(f'{log_folder}/{configs.log_dir}', exist_ok=True)
    # profiler = SimpleProfiler(filename=f'{log_folder}/{configs.log_dir}/profiler.txt') #(output_filename=f'{log_folder}/{configs.log_dir}/profiler.txt')
    # np.set_printoptions(precision=4, suppress=True)

    # # save the backup files
    # backup_dir = os.path.join(log_folder, configs.log_dir, 'backup_files_%s' % str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    # if not configs['test']:
    #     os.makedirs(backup_dir, exist_ok=True)
    #     os.system('cp main.py {}'.format(backup_dir))
    #     os.system('cp dataloader/dataset.py {}'.format(backup_dir))
    #     os.system('cp dataloader/pc_dataset.py {}'.format(backup_dir))
    #     os.system('cp {} {}'.format(configs.config_path, backup_dir))
    #     os.system('cp network/base_model.py {}'.format(backup_dir))
    #     os.system('cp network/spvcnn.py {}'.format(backup_dir))
    #     os.system('cp {}.py {}'.format('network/' + configs['model_params']['model_architecture'], backup_dir))

    # # reproducibility
    # torch.manual_seed(configs.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # np.random.seed(configs.seed)
    config_path = configs.config_path

    # Creates data loaders for each set of the split dataset
    test_dataset_loader = build_loader(configs)

    # Specifies arch_2dpass.py from networks as the model file
    model_file = importlib.import_module('network.' + configs['model_params']['model_architecture'])
    my_model = model_file.get_model(configs)

    # Loads checkpoint
    assert os.path.exists(configs.checkpoint)
    my_model = my_model.load_from_checkpoint(configs.checkpoint, config=configs, strict=True)

    ##################
    # Inference over iterator of test set
    print('Start prediction/s...')
    for i, sample in enumerate(iter(test_dataset_loader)):
        my_model.eval()
        with torch.no_grad():
            prediction = my_model(sample)
        print(f'PREDICTION {i}')
        print(prediction["logits"]) # logits in shape [number of points (masked), number of classes]
        input("Press to continue")

    import sys
    sys.exit()


    # VISUALIZATION
    # draw_points_image_labels(
    #     np.array(image),
    #     points_img.astype(int),
    #     labels[keep_idx],
    #     color_palette_type='SemanticKITTI_long'
    # )