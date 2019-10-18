TRAIN_IMAGES = 'data/train_images/'
TRAIN_CSV = 'data/my_csv/train.csv'
VALID_CSV = 'data/my_csv/valid.csv'
TEST_IMAGES = 'data/test_images/'

EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 6
CROP_SIZE = None

CUDA_VISIBLE_DEVICES = '1'

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = None
ACTIVATION = 'sigmoid'

CONTINUE = 'logs/unet_se_resnext50_32x4d/checkpoints/best_full.pth'

LOGDIR = f'logs/unet_{ENCODER}'


import os
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp

from tqdm.auto import tqdm
from modules.comp_tools import Dataset, AUGMENTATIONS_TRAIN, decode_masks, preprocessing_fn, to_tensor, get_segm_model
from modules.common import rle_decode
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, IouCallback
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
    classes=4,
    activation=ACTIVATION,
)
model = get_segm_model('Unet', arch_args, load_weights=CONTINUE)

train_dataset = Dataset(
    train_df,
    img_prefix=TRAIN_IMAGES, 
    augmentations=AUGMENTATIONS_TRAIN, 
    preprocess_img=preprocessing_fn,
    preprocess_mask=to_tensor,
)
valid_dataset = Dataset(
    valid_df,
    img_prefix=TRAIN_IMAGES, 
    augmentations=None, 
    preprocess_img=preprocessing_fn,
    preprocess_mask=to_tensor,
)
train_dl = BaseDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_dl = BaseDataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# experiment setup
num_epochs = EPOCHS
logdir = LOGDIR
loaders = {
    "train": train_dl,
    "valid": valid_dl
}
criterion = smp.utils.losses.BCEJaccardLoss(eps=1e-7)
optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': LR / 10}, 
    {'params': model.encoder.parameters(), 'lr': LR},  
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)

callbacks = [
    DiceCallback(
        threshold=0.5
    ),
    IouCallback(
        threshold=0.5,
    )
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

## Step 2. 

criterion = smp.utils.losses.DiceLoss(eps=1e-7)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=3,
    cooldown=0,
    verbose=1,
    min_lr=1e-7,
)
optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-4}, 
    {'params': model.encoder.parameters(), 'lr': 1e-5},  
])

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