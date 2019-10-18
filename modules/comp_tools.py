import os.path as osp
import numpy as np

import albumentations as A
import pretrainedmodels
import torchvision as tv
import torch
import cv2
import os
import segmentation_models_pytorch as smp

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader as BaseDataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm.auto import tqdm
from albumentations.augmentations.functional import normalize

from .common import rle_decode


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def preprocessing_fn(x):
    return to_tensor(normalize(x, MEAN, STD, max_pixel_value=1.0))


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

AUGMENTATIONS_TRAIN = A.Compose([
    #     A.RandomCrop(W, H),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
    ], p=0.3),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
    ], p=0.3),
    #     RandomSizedCrop(min_max_height=(176, 256), height=H, width=W,p=0.25),
    A.ToFloat(max_value=1)
], p=1)

AUGMENTATIONS_TEST = A.Compose([
    A.ToFloat(max_value=1)
], p=1)


def dice_channel_torch(probability, truth, threshold):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(channel_num):
                channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :], threshold)
                mean_dice_channel += channel_dice/(batch_size * channel_num)
    return mean_dice_channel

def dice_single_channel(probability, truth, threshold, eps=1e-9):
    p = (probability.view(-1) > threshold).float()
    t = (truth.view(-1) > 0.5).float()
    dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)
    return dice


def dummy(x): return x

class Dataset(BaseDataset):
    def __init__(
        self,
        df,
        img_prefix,
        augmentations=None,
        img_size=None,
        n_channels=3,
        shuffle=True,
        preprocess_img=dummy,
        preprocess_mask=dummy,
    ):
        self.df = df
        self.img_prefix = img_prefix
        self.img_ids = df.ImageId.unique()
        self.img_size = img_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.preprocess_img = preprocess_img
        self.preprocess_mask = preprocess_mask

    def __len__(self):
        return len(self.img_ids)

    def get_img(self, img_id):
        img_path = osp.join(self.img_prefix, img_id)
        if self.n_channels != 3:
            raise NotImplementedError
        img = np.array(Image.open(img_path))
        if self.img_size:
            img = cv2.resize(img, self.img_size)
        return img

    def get_masks(self, img_id):
        img_df = self.df[self.df.ImageId == img_id]
        img_df = img_df.sort_values('ClassId')
        masks = np.stack(img_df.Mask.values, axis=-1).astype(np.int8)
        if self.img_size:
            masks = cv2.resize(masks, self.img_size)
        return masks

    def augm_img(self, img, mask=None):
        pair = self.augmentations(image=img, mask=mask)
        return pair['image'], pair['mask']

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        img = self.get_img(img_id)
        masks = self.get_masks(img_id)
        if self.augmentations:
            img, masks = self.augm_img(img, masks)
        img = img / 255.
        img = np.clip(img, 0, 1)
        masks = masks / 255.
        masks = np.clip(masks, 0, 1)
        masks[masks > 0] = 1
        return self.preprocess_img(img), self.preprocess_mask(masks)


class TestDataset(BaseDataset):
    def __init__(
        self,
        img_prefix,
        img_size=None,
        n_channels=3,
        shuffle=True,
        preprocess_img=dummy,
        preprocess_mask=dummy,
    ):
        self.img_prefix = img_prefix
        self.img_ids = os.listdir(img_prefix)
        self.img_size = img_size
        self.n_channels = n_channels
        self.preprocess_img = preprocess_img

    def __len__(self):
        return len(self.img_ids)

    def get_img(self, img_id):
        img_path = osp.join(self.img_prefix, img_id)
        if self.n_channels != 3:
            raise NotImplementedError
        img = np.array(Image.open(img_path))
        if self.img_size:
            img = cv2.resize(img, self.img_size)
        return img

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        img = self.get_img(img_id)
        img = img / 255.
        img = np.clip(img, 0, 1)
        return self.preprocess_img(img)


def decode_masks(df):
    decoded_masks = []
    for enc_mask in tqdm(df.EncodedPixels):
        dec_mask = rle_decode(enc_mask, (1600, 256)).astype(np.bool8).T
        decoded_masks.append(dec_mask)
    df['Mask'] = decoded_masks
    return df


def predict_cls(model, dl, device='cuda:0'):
    logit_batches = []
    gt_batches = []
    for batch in tqdm(dl):
        with torch.no_grad():
            features = batch['features']
            gt = batch.get('targets_one_hot', None)
            features = features.to(device)
            logits = model(features).detach().cpu()
            logit_batches.append(logits)
            if gt is not None:
                gt_batches.append(gt)
    all_logits = torch.cat(logit_batches)
    all_gt = torch.cat(gt_batches)
    return all_logits, all_gt


def predict_semg(model, dl, device='cuda:0'):
    logit_batches = []
    masks_batches = []
    for batch in tqdm(dl):
        with torch.no_grad():
            if isinstance(batch, (tuple, list)):
                features, masks = batch
            else:
                features, masks = batch, None
            features = features.to(device)
            logits = model(features).detach().cpu()
            logit_batches.append(logits)
            if masks is not None:
                masks_batches.append(masks)
    all_logits = torch.cat(logit_batches)
    all_masks = torch.cat(masks_batches)
    return all_logits, all_masks


def get_segm_model(arch, arch_args, load_weights=None):
    model_builder = smp.__dict__[arch]
    model = model_builder(**arch_args)
    if load_weights:
        print('Loading', load_weights)
        state_dict = torch.load(load_weights)
        print(model.load_state_dict(state_dict['model_state_dict']))
    return model


def get_model(model_name, num_classes=2, pretrained='imagenet', load_weights=None):
    model_fn = pretrainedmodels.__dict__[model_name]
    model = model_fn(pretrained=pretrained)

    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    init_features = model.last_linear.in_features
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(init_features),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=init_features, out_features=init_features // 2),
        nn.ReLU(),
        nn.BatchNorm1d(init_features // 2, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=init_features // 2, out_features=num_classes),
    )

    if load_weights:
        print('Loading', load_weights)
        state_dict = torch.load(load_weights)
        print(model.load_state_dict(state_dict['model_state_dict']))

    return model


def get_tv_model(model_name, num_classes=2, pretrained='imagenet'):
    model = tv.models.resnet18(pretrained=True)
    init_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(init_features),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=init_features, out_features=init_features // 2),
        nn.ReLU(),
        nn.BatchNorm1d(init_features // 2, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=init_features // 2, out_features=num_classes),
    )
    return model


class ModelAgg:
    def __init__(self, models):
        self.models = models
    
    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)


class ClsDataset(Dataset):
    def __init__(
        self,
        df,
        img_prefix,
        augmentations=None,
        img_size=None,
        n_channels=3,
        shuffle=True,
        preprocess_img=dummy,
    ):
        self.df = df
        self.img_prefix = img_prefix
        self.img_ids = df.ImageId.values
        self.labels = df.has_defect.values
        self.img_size = img_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.preprocess_img = preprocess_img

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        img = self.get_img(img_id)
        if self.augmentations:
            img, _ = self.augm_img(img)
        img = img / 255.
        img = np.clip(img, 0, 1)
        y = self.labels[i]
        one_hot = np.zeros((2,), dtype='float32')
        one_hot[y] = 1
        return {
            'features': self.preprocess_img(img), 
            'targets': y,
            'targets_one_hot': one_hot, 
        }


class TestClsDataset(Dataset):
    def __init__(
        self,
        img_prefix,
        img_size=None,
        n_channels=3,
        shuffle=True,
        preprocess_img=dummy,
    ):
        self.img_prefix = img_prefix
        self.img_ids = os.listdir(img_prefix)
        self.img_size = img_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.preprocess_img = preprocess_img

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        img = self.get_img(img_id)
        img = img / 255.
        img = np.clip(img, 0, 1)
        return {
            'features': self.preprocess_img(img),
            'img_id': img_id,
        }

