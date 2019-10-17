import os.path as osp
import numpy as np

import albumentations as A
import pretrainedmodels
import torchvision as tv
import torch
import cv2
import os

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader as BaseDataLoader
from torch.utils.data import Dataset as BaseDataset

from tqdm.auto import tqdm

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


def predict_cls(model, dl, is_cuda=True):
    logit_batches = []
    gt_batches = []
    for batch in tqdm(dl):
        with torch.no_grad():
            features = batch['features']
            gt = batch.get('targets_one_hot', None)
            if is_cuda:
                features = features.cuda()
            logits = model(features).detach().cpu()
            logit_batches.append(logits)
            if gt is not None:
                gt_batches.append(gt)
    all_logits = torch.cat(logit_batches)
    all_gt = torch.cat(gt_batches)
    return all_logits, all_gt


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

