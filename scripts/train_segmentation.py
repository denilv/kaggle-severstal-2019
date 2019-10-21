TRAIN_IMAGES = 'data/train_images/'
TRAIN_CSV = 'data/segm_df/train.csv'
VALID_CSV = 'data/segm_df/valid.csv'
TEST_IMAGES = 'data/test_images/'

EPOCHS = 7
LR = 5e-4
BATCH_SIZE = 4
CROP_SIZE = None

CUDA_VISIBLE_DEVICES = '1, 0'

ENCODER = 'se_resnext101_32x4d'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = None
ACTIVATION = 'sigmoid'

CONTINUE = '/mnt/NVME1TB/Projects/kaggle-severstal-2019/logs/fpn_se_resnext101_32x4d_non_augm/checkpoints/best_full.pth'

LOGDIR = f'logs/fpn_{ENCODER}_non_augm'


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
    classes=4,
    activation=ACTIVATION,
)
model = get_segm_model('FPN', arch_args, load_weights=CONTINUE)

train_dataset = Dataset(
    train_df,
    img_prefix=TRAIN_IMAGES, 
    augmentations=A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ToFloat(max_value=1),
    ], p=1), 
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
train_dl = BaseDataLoader(train_dataset, batch_size=BATCH_SIZE * 2, shuffle=True, num_workers=4)
valid_dl = BaseDataLoader(valid_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)

# experiment setup
num_epochs = EPOCHS
logdir = LOGDIR
loaders = {
    "train": train_dl,
    "valid": valid_dl
}
criterion = smp.utils.losses.DiceLoss(eps=1e-7)
optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': LR / 10},  
    {'params': model.decoder.parameters(), 'lr': LR}, 
])

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6])

callbacks = [
    DiceCallback(
        threshold=0.5
    ),
    IouCallback(
        threshold=0.5,
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