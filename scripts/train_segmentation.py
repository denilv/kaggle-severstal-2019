TRAIN_IMAGES = 'data/train_images/'
TRAIN_CSV = 'data/segm_df/train.csv'
VALID_CSV = 'data/segm_df/valid.csv'
TEST_IMAGES = 'data/test_images/'

EPOCHS = 15
LR = 1e-3
BATCH_SIZE = 4
CROP_SIZE = None

CUDA_VISIBLE_DEVICES = '1, 0'

ENCODER = 'efficientnet-b5'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = None
ACTIVATION = 'softmax'
ONLY_DEFECTS = True
BACKGROUND = True

CONTINUE = None

LOGDIR = f'logs/fpn_{ENCODER}'


import os
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import albumentations as A

from tqdm.auto import tqdm
from modules.comp_tools import Dataset, AUGMENTATIONS_TRAIN, decode_masks, preprocessing_fn, to_tensor, get_segm_model
from modules.common import rle_decode
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, IouCallback, OptimizerCallback
from torch.utils.data import DataLoader as BaseDataLoader
from torch.utils.data import Dataset as BaseDataset


train_df = pd.read_csv(TRAIN_CSV).fillna('')
valid_df = pd.read_csv(VALID_CSV).fillna('')

# TODO: decode masks
train_df = decode_masks(train_df)
valid_df = decode_masks(valid_df)

arch_args = dict(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=5,
    activation=ACTIVATION,
)
model = get_segm_model('FPN', arch_args, load_weights=CONTINUE)

train_dataset = Dataset(
    train_df,
    img_prefix=TRAIN_IMAGES, 
    augmentations=AUGMENTATIONS_TRAIN, 
    background=BACKGROUND,
    preprocess_img=preprocessing_fn,
    preprocess_mask=to_tensor,
)
valid_dataset = Dataset(
    valid_df,
    img_prefix=TRAIN_IMAGES, 
    augmentations=None, 
    background=BACKGROUND, 
    preprocess_img=preprocessing_fn,
    preprocess_mask=to_tensor,
)
train_dl = BaseDataLoader(train_dataset, batch_size=BATCH_SIZE * 2, shuffle=True, num_workers=0)
valid_dl = BaseDataLoader(valid_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

# experiment setup
num_epochs = EPOCHS
logdir = LOGDIR
loaders = {
    "train": train_dl,
    "valid": valid_dl
}
criterion = smp.utils.losses.BCEDiceLoss(eps=1e-7)
optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': LR / 10},  
    {'params': model.decoder.parameters(), 'lr': LR},
])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 10, 12])

callbacks = [
    DiceCallback(
        threshold=0.5,
        activation=ACTIVATION,
    ),
    IouCallback(
        threshold=0.5,
        activation=ACTIVATION,
    ),
]
runner = SupervisedRunner()

## Step 1.

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=1,
    scheduler=scheduler,
)
