TRAIN_IMAGES = '/home/denilv/Projects/severstal/data/train_images/'
TRAIN_CSV = '/mnt/NVME1TB/Projects/severstal/data/cls_df/train.csv'
VALID_CSV = '/mnt/NVME1TB/Projects/severstal/data/cls_df/valid.csv'
TEST_IMAGES = '/home/denilv/Projects/severstal/data/test_images/'

EPOCHS = 30
LR = 1e-4
BATCH_SIZE = 10
CROP_SIZE = None

CUDA_VISIBLE_DEVICES = '1'

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = None
ACTIVATION = 'sigmoid'

CONTINUE = '/mnt/NVME1TB/Projects/severstal/logs/cls_resnet50_new_wave/checkpoints/last.pth'

LOGDIR = f'logs/cls_{ENCODER}_new_wave'


import os
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import albumentations as A

from albumentations.augmentations.functional import normalize
from tqdm.auto import tqdm
from modules.comp_tools import ClsDataset, AUGMENTATIONS_TRAIN, get_model, get_tv_model
from modules.common import rle_decode
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import F1ScoreCallback, AccuracyCallback
from torch.utils.data import DataLoader as BaseDataLoader
from torch.utils.data import Dataset as BaseDataset


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


train_df = pd.read_csv(TRAIN_CSV).fillna('')[:]
valid_df = pd.read_csv(VALID_CSV).fillna('')[:]

# TODO: model
model = get_model(ENCODER, 2, ENCODER_WEIGHTS)
preprocessing_fn = lambda x: to_tensor(normalize(x, model.mean, model.std))
# preprocessing_fn = lambda x: to_tensor(normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

train_dataset = ClsDataset(
    train_df,
    img_prefix=TRAIN_IMAGES, 
    augmentations=AUGMENTATIONS_TRAIN, 
    preprocess_img=preprocessing_fn,
)
valid_dataset = ClsDataset(
    valid_df,
    img_prefix=TRAIN_IMAGES, 
    augmentations=None, 
    preprocess_img=preprocessing_fn,
)
train_dl = BaseDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
valid_dl = BaseDataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)

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
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([
    {'params': model.layer1.parameters(), 'lr': LR / 10},
    {'params': model.layer2.parameters(), 'lr': LR / 5},
    {'params': model.layer3.parameters(), 'lr': LR / 2},
    {'params': model.layer4.parameters(), 'lr': LR / 1},
], lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, cooldown=2, min_lr=1e-7)

callbacks = [
    AccuracyCallback(num_classes=2, threshold=0.5, activation='Softmax'),
    F1ScoreCallback(input_key="targets_one_hot", activation='Softmax', threshold=0.5),
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

## Step 2. FT with HFlip

train_dataset.augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ToFloat(max_value=1),
], p=1)

optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': LR / 10},
    {'params': model.layer2.parameters(), 'lr': LR / 5},
    {'params': model.layer3.parameters(), 'lr': LR / 2},
    {'params': model.layer4.parameters(), 'lr': LR / 1},
], lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, cooldown=1, min_lr=1e-7)

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