import os, sys, getopt
import cv2
import yaml
import time
import json
import torch
import numpy as np
import pandas as pd
import albumentations as A
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from osgeo import gdal

from argparse import Namespace
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from SAR import SARModel, SARDataModule

if __name__ != "__main__":
   exit()

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "hr:c:s:")
except getopt.GetoptError:
    print('train_model.py -r <project_root_path> -c <config_filename> -s <data_split_path>')
    sys.exit(2)

SUBGROUPS_DIR = None
CONFIG_FILENAME = None
DATA_SPLIT_PATH = None

for opt, arg in opts:
    if opt == '-h':
        print('train_model.py -r <project_root_path> -c <config_filename> -s <data_split_path>')
        sys.exit()
    elif opt == '-r':
        SUBGROUPS_DIR = arg
    elif opt == '-c':
        CONFIG_FILENAME = arg
    elif opt == '-s':
        DATA_SPLIT_PATH = arg

###################
# SETUP RANDOM SEED
###################

pl.seed_everything(15151515, workers=True)
np.random.seed(15151515)

################
# CHECK FOR CUDA
################

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ", device)

if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

##########################
# SETUP CONSTS FOR FOLDERS
##########################

# Path to project root folder
DATA_DIR = os.path.join(SUBGROUPS_DIR, 'data')

MODELS_DIR = os.path.join(SUBGROUPS_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
LOGS_DIR = os.path.join(SUBGROUPS_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(SUBGROUPS_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Local Folders
ROOT_DIR = os.getcwd()
INPUT_DIR = os.path.join(ROOT_DIR, 'input')
os.makedirs(INPUT_DIR, exist_ok=True)
IMAGES_DIR = os.path.join(INPUT_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)
MASKS_DIR = os.path.join(INPUT_DIR, 'masks')
os.makedirs(MASKS_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

###################
# OPEN MODEL CONFIG
###################

with open(os.path.join(SUBGROUPS_DIR, "configs", CONFIG_FILENAME), 'r') as file:
    config = yaml.safe_load(file)

########################################
# IMPORT *.PY FOR VV TO RGB CONVERTATION
########################################

import importlib.util

spec = importlib.util.spec_from_file_location("vv2rgb", os.path.join(SUBGROUPS_DIR, 'vv2rgb', f"{config['vv2rgb_name']}.py"))

vv2rgb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vv2rgb)
                
NODATA_COLOR = vv2rgb.NODATA_RGB
CONST_NORM = A.Normalize(mean=vv2rgb.NORM_MEAN, std=vv2rgb.NORM_STD, max_pixel_value=255.0, p=1.0)

# to convert by path
def convert_db_norm(file_path, show_plot=False):
    s2_ds = gdal.Open(file_path)

    s2 = np.empty((s2_ds.RasterYSize, s2_ds.RasterXSize, s2_ds.RasterCount))
    for i in range(s2.shape[2]):
        s2[:, :, i] = s2_ds.GetRasterBand(i + 1).ReadAsArray()
        
    rgb = vv2rgb.convert(s2)

    if show_plot is True:
        plt.imshow(rgb)
        plt.show()

    return rgb

#################
# LOAD DATA SPLIT
#################

dataset_df = pd.read_csv(DATA_SPLIT_PATH)

#############################################
# PREPARE IMAGES TO BE USED IN MODEL TRAINING
#############################################

NATIVE_RESOLUTION = 32

resolution_chain = [stage['image_size'] for stage in config['hparams']]
dataset_df_chain = [dataset_df.copy() for _ in resolution_chain]

for i in range(len(resolution_chain)):
    resolution = resolution_chain[i]
    target_df = dataset_df_chain[i]

    resolution_images_dir = os.path.join(IMAGES_DIR, str(resolution))
    resolution_masks_dir = os.path.join(MASKS_DIR, str(resolution))

    os.makedirs(resolution_images_dir, exist_ok=True)
    os.makedirs(resolution_masks_dir, exist_ok=True)

    for index, row in dataset_df.iterrows():
        file_path = os.path.join(row.dir_name, f"{row.image_name}_S2.tif")
        mask_path = os.path.join(row.dir_name, f"{row.image_name}_LABEL.tif")

        if not (os.path.exists(file_path) and os.path.exists(mask_path)):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # vv = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        # if resolution != NATIVE_RESOLUTION:
        #     vv = cv2.resize(vv, (resolution, resolution))
        #     # max pooling of mask using dilatation
        #     scale = NATIVE_RESOLUTION // resolution
        #     mask = cv2.dilate(mask, np.ones((scale, scale)))
        #     init = scale // 2 
        #     mask = mask[init::scale, init::scale]
        #     # resize by nearest to make sure of required dimentions
        #     mask = cv2.resize(mask, (resolution, resolution), interpolation = cv2.INTER_NEAREST)

        image = convert_db_norm(file_path)

        image_save2_path = os.path.join(resolution_images_dir, f"{row.image_name}_IMAGE.jpg")
        mask_save2_path = os.path.join(resolution_masks_dir, f"{row.image_name}_MASK.tif")

        cv2.imwrite(image_save2_path, image)
        cv2.imwrite(mask_save2_path, mask)

        target_df.loc[index, "image_path"] = image_save2_path
        target_df.loc[index, "mask_path"] = mask_save2_path
        print(f"Image and mask files generated for {row.image_name}")

    target_df.dropna(inplace=True)

##################
# DEFINE CALLBACKS
##################

class TempModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        # trainer.lightning_module.save_transformed_model = True
        super()._save_checkpoint(trainer, filepath)

class MetricsHistoryCallback(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_history = []
        self.validation_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        metrics = {k: v.item() for k, v in m.items()}
        metrics['epoch'] = trainer.current_epoch
        self.train_history.append(metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        metrics = {k: v.item() for k, v in m.items()}
        metrics['epoch'] = trainer.current_epoch
        self.validation_history.append(metrics)

    def keep_train_keys(self, key, start):
        return key.startswith(start)

def create_callbacks_and_logger(model_params):
    model_name_base = f"{model_params.data_version}-{model_params.image_size}-REF-{model_params.arch_name}-{model_params.encoder_name}-{model_params.encoder_weights}-{model_params.model_version}"

    checkpoint_callback = ModelCheckpoint(dirpath=MODELS_DIR,
                                      filename=rf'{model_name_base}-{{epoch}}-{{valid_loss:.2f}}',
                                      monitor='valid_loss',
                                      verbose=True,
                                      save_top_k=1,
                                      mode='min'
                                     )
    print(f"Checkpoint File Name Format: {checkpoint_callback.filename}")

    temporary_checkpoint_callback = TempModelCheckpoint(dirpath=MODELS_DIR,
                                        filename=rf'temp',
                                        monitor='valid_loss',
                                        verbose=True,
                                        save_top_k=1,
                                        mode='min'
                                        )
    print(f"Temporary Checkpoint File Name Format: {temporary_checkpoint_callback.filename}")

    history_callback = MetricsHistoryCallback()

    lr_monitor_callback = LearningRateMonitor()
    early_stop_callback = EarlyStopping(monitor='valid_loss', patience=10, verbose=True, mode='min')

    os.makedirs(os.path.join(LOGS_DIR, model_name_base), exist_ok=True)
    logger = TensorBoardLogger(save_dir=LOGS_DIR, name=model_name_base)
    
    return [checkpoint_callback, temporary_checkpoint_callback, history_callback, lr_monitor_callback, early_stop_callback], logger, model_name_base

############################################
# EVALUATE MODEL AND SAVE EVALUATION RESULTS
############################################

def eval_model(trainer, model, datamodule, save_to):
    train_dataloader = DataLoader(datamodule.train_dataset, shuffle=True, batch_size=datamodule.hparams.batch_size, drop_last=True)

    train_metrics = trainer.validate(model, dataloaders=train_dataloader, verbose=False)
    valid_metrics = trainer.validate(model, datamodule=datamodule, verbose=False)
    test_metrics = trainer.test(model, datamodule=datamodule, verbose=False)

    prefixes = ['train', 'valid', 'test']
    metrics = [train_metrics, valid_metrics, test_metrics]

    dic = { "mode": [] }
    for i in range(3):
        prefix = prefixes[i]
        scores = metrics[i][0]

        dic["mode"].append(prefix)
        for entry in scores.keys():
            clipped = '_'.join(entry.split('_')[1:])
            if clipped not in dic.keys():
                dic[clipped] = []

            dic[clipped].append(scores[entry])

    df = pd.DataFrame.from_dict(dic)
    print(df)

    df.to_csv(save_to)

#################
# TRAIN THE MODEL
#################

for train_for_resolution in range(len(resolution_chain)):
    hparams = config['hparams'][train_for_resolution]
    # INIT MODEL PARAMS
    model_params = dict(data_version=config['data_version'],
                        model_version=f"{config['vv2rgb_name']}-{config['model_version']}-{config['loss_function']}",
                        arch_name=config['arch_name'],
                        encoder_name=config['encoder_name'],
                        encoder_weights=config['encoder_weights'],
                        load_weights_from= None if train_for_resolution == 0 else os.path.join(MODELS_DIR, 'temp.ckpt'),
                        decoder_attention_type=None,
                        learning_rate=hparams['learning_rate'],
                        auto_lr_find=False,
                        max_epochs=hparams['max_epochs'],
                        image_size=hparams['image_size'],
                        batch_size=hparams['batch_size'],
                        accumulate_grad_batches=hparams['accumulate_grad_batches'],
                        auto_scale_batch_size=None, #'binsearch',
                        loss_function=config['loss_function'],
                        loss_alpha=None,
                        loss_beta=None,
                        loss_gamma=None,
                        in_channels=3,
                        input_shape=(resolution_chain[train_for_resolution], resolution_chain[train_for_resolution], 3),
                        out_classes=1)

    model_params = Namespace(**model_params)

    # CREATE TRAINER
    callbacks, logger, model_base_name = create_callbacks_and_logger(model_params)

    trainer = pl.Trainer(accelerator='gpu',
                        fast_dev_run=False,
                        max_epochs=model_params.max_epochs,
                        check_val_every_n_epoch=1,
                        logger=logger,
                        log_every_n_steps=20,
                        callbacks=callbacks,
                        accumulate_grad_batches=model_params.accumulate_grad_batches,
                        deterministic='warn')
    
    # CREATE MODEL
    dataset_split_df = dataset_df_chain[train_for_resolution].copy()

    sentinel1_data = SARDataModule(dataset_split_df, const_norm=CONST_NORM, nodata_color=NODATA_COLOR, image_size=model_params.image_size, batch_size=model_params.batch_size)
    sentinel1_data.setup()

    model = SARModel(model_params)

    print(f"Model Parameters: {model.hparams}")
    print(f"Learning Rate: {model.learning_rate}")

    if train_for_resolution > 0:
        os.remove(os.path.join(MODELS_DIR, 'temp.ckpt'))

    # TRAIN MODEL
    if model.hparams_.auto_lr_find is True or model.hparams_.auto_scale_batch_size is not None:
        trainer.tune(model, datamodule=sentinel1_data)

    start = time.time()
    trainer.fit(model, datamodule=sentinel1_data)
    end = time.time()

    time_used = end - start

    print(f"Path of best model checkpoint: {callbacks[0].best_model_path.split('/')[-1]}")
    print(f"GPU Memory Allocated: {torch.cuda.max_memory_allocated()}")
    print(f"GPU Memory Reserved : {torch.cuda.max_memory_reserved()}")

    base_path = os.path.join(RESULTS_DIR, model_base_name)
    times_path = base_path + ".txt"

    with open(times_path, 'a+') as f:
        f.write(f"Stage {train_for_resolution} ({hparams['image_size']}x{hparams['image_size']}) trained in {time_used} seconds!\n{time_used}\n")

    # SAVE METRICS HISTORY
    history_callback = callbacks[2]
    train_json_string = json.dumps(history_callback.train_history)
    with open(base_path + "_train_history.json", 'w+') as f:
        f.write(train_json_string)

    valid_json_string = json.dumps(history_callback.validation_history)
    with open(base_path + "_valid_history.json", 'w+') as f:
        f.write(valid_json_string)

    # EVALUATE MODEL
    eval_model(trainer, model, sentinel1_data, base_path + ".csv")

#############################
# DELETE TEMPORARY CHECKPOINT
#############################

os.remove(os.path.join(MODELS_DIR, 'temp.ckpt'))