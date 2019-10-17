TRAIN_IMAGES = '/home/denilv/Projects/severstal/data/train_images/'
TRAIN_CSV = '/mnt/NVME1TB/Projects/severstal/data/my_csv/train.csv'
VALID_CSV = '/mnt/NVME1TB/Projects/severstal/data/my_csv/valid.csv'
TEST_IMAGES = '/home/denilv/Projects/severstal/data/test_images/'

EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 6
CROP_SIZE = None

CUDA_VISIBLE_DEVICES = '1'

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = None
ACTIVATION = 'sigmoid'

CONTINUE = '/mnt/NVME1TB/Projects/severstal/logs/unet_se_resnext50_32x4d/checkpoints/best_full.pth'

LOGDIR = f'logs/unet_{ENCODER}'


import os
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp

from tqdm.auto import tqdm
from modules.comp_tools import Dataset, AUGMENTATIONS_TRAIN
from modules.common import rle_decode
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, IouCallback
from torch.utils.data import DataLoader as BaseDataLoader
from torch.utils.data import Dataset as BaseDataset


def decode_masks(df):
    decoded_masks = []
    for enc_mask in tqdm(df.EncodedPixels):
        dec_mask = rle_decode(enc_mask, (1600, 256)).astype(np.bool8).T
        decoded_masks.append(dec_mask)
    df['Mask'] = decoded_masks
    return df


encoder_preprocessing = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def preprocessing_fn(x):
    x = encoder_preprocessing(x)
    return to_tensor(x)


def to_float32(x):
    return x.transpose(2, 0, 1).astype('float32')


train_df = pd.read_csv(TRAIN_CSV).fillna('')
valid_df = pd.read_csv(VALID_CSV).fillna('')

# TODO: decode masks
train_df = decode_masks(train_df)
valid_df = decode_masks(valid_df)

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=4,
    activation=ACTIVATION,
)

train_dataset = Dataset(
    train_df,
    img_prefix=TRAIN_IMAGES, 
    augmentations=AUGMENTATIONS_TRAIN, 
    preprocess_img=preprocessing_fn,
    preprocess_mask=to_float32,
)
valid_dataset = Dataset(
    valid_df,
    img_prefix=TRAIN_IMAGES, 
    augmentations=None, 
    preprocess_img=preprocessing_fn,
    preprocess_mask=to_float32,
)
train_dl = BaseDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_dl = BaseDataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

if CONTINUE:
    print('Loading', CONTINUE)
    state_dict = torch.load(CONTINUE)
    print(model.load_state_dict(state_dict['model_state_dict']))

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