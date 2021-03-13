import argparse
import yaml
import copy
import logging


# pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# model
from models import *

# augmentation
import albumentations
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from albumentations.pytorch import ToTensorV2


#
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import time
import gc
import os

from scipy.sparse import coo_matrix
import random

from pathlib import Path

# =========================================================

ROOT = Path.cwd()
# ROOT = Path.cwd().parent  # for kaggle notebook
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
DATA = INPUT / "ranzcr-clip-catheter-line-classification"
TRAIN = DATA / "train"
TEST = DATA / "test"


TRAIN_NPY = INPUT / "ranzcr-clip-train-numpy"
TMP = ROOT / "tmp"
TMP.mkdir(exist_ok=True)


CLASSES = [
    'ETT - Abnormal',
    'ETT - Borderline',
    'ETT - Normal',
    'NGT - Abnormal',
    'NGT - Borderline',
    'NGT - Incompletely Imaged',
    'NGT - Normal',
    'CVC - Abnormal',
    'CVC - Borderline',
    'CVC - Normal',
    'Swan Ganz Catheter Present'
]


RANDAM_SEED = 107
N_CLASSES = 11
FOLDS = [0, 1, 2, 3, 4]
N_FOLD = len(FOLDS)

train = pd.read_csv(DATA / "train.csv")
smpl_sub = pd.read_csv(DATA / "sample_submission.csv")

# ======================================================
# logging 設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def multi_label_stratified_group_k_fold(label_arr: np.array, gid_arr: np.array, n_fold: int, seed: int = 107):
    """
    create multi-label stratified group kfold indexs.

    reference: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    input:
        label_arr: numpy.ndarray, shape = (n_train, n_class)
            multi-label for each sample's index using multi-hot vectors
        gid_arr: numpy.array, shape = (n_train,)
            group id for each sample's index
        n_fold: int. number of fold.
        seed: random seed.
    output:
        yield indexs array list for each fold's train and validation.
    """
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    n_train, n_class = label_arr.shape
    gid_unique = sorted(set(gid_arr))
    n_group = len(gid_unique)

    # # aid_arr: (n_train,), indicates alternative id for group id.
    # # generally, group ids are not 0-index and continuous or not integer.
    gid2aid = dict(zip(gid_unique, range(n_group)))
    # aid2gid = dict(zip(range(n_group), gid_unique))
    aid_arr = np.vectorize(lambda x: gid2aid[x])(gid_arr)

    # # count labels by class
    cnts_by_class = label_arr.sum(axis=0)  # (n_class, )

    # # count labels by group id.
    col, row = np.array(sorted(enumerate(aid_arr), key=lambda x: x[1])).T
    cnts_by_group = coo_matrix(
        (np.ones(len(label_arr)), (row, col))
    ).dot(coo_matrix(label_arr)).toarray().astype(int)
    del col
    del row
    cnts_by_fold = np.zeros((n_fold, n_class), int)

    groups_by_fold = [[] for _ in range(n_fold)]
    group_and_cnts = list(enumerate(cnts_by_group))  # pair of aid and cnt by group
    np.random.shuffle(group_and_cnts)
    logger.debug(f'finished preparation {time.time() - start_time}')

    for aid, cnt_by_g in sorted(group_and_cnts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for fid in range(n_fold):
            # # eval assignment.
            cnts_by_fold[fid] += cnt_by_g
            fold_eval = (cnts_by_fold / cnts_by_class).std(axis=0).mean()
            cnts_by_fold[fid] -= cnt_by_g

            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = fid

        cnts_by_fold[best_fold] += cnt_by_g
        groups_by_fold[best_fold].append(aid)
    logger.debug(f'finished assignment {time.time() - start_time}')

    gc.collect()
    idx_arr = np.arange(n_train)
    for fid in range(n_fold):
        val_groups = groups_by_fold[fid]

        val_indexs_bool = np.isin(aid_arr, val_groups)
        train_indexs = idx_arr[~val_indexs_bool]
        val_indexs = idx_arr[val_indexs_bool]

        logger.info(f'[fold {fid}]')
        logger.info(f'n_group: (train, val) = ({n_group - len(val_groups)} , {len(val_groups)})')
        logger.info(f'n_sample: (train. val) = ({len(train_indexs), len(val_indexs)})')

        yield train_indexs, val_indexs


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = deterministic


class ImageTransformBase:
    """
    Base Image Transform class.

    Args:
        data_augmentations: List of tuple(method: str, params :dict), each elems pass to albumentations
    """

    def __init__(self, data_augmentations: List[Tuple[str, Dict]]):
        """Initialize."""
        augmentations_list = [
            self._get_augmentation(aug_name)(**params)
            for aug_name, params in data_augmentations]
        self.data_aug = albumentations.Compose(augmentations_list)

    def __call__(self, pair: Tuple[np.ndarray]) -> Tuple[np.ndarray]:
        """You have to implement this by task"""
        raise NotImplementedError

    @staticmethod
    def _get_augmentation(aug_name: str) -> ImageOnlyTransform or DualTransform:
        """Get augmentations from albumentations"""
        if hasattr(albumentations, aug_name):
            return getattr(albumentations, aug_name)
        else:
            return eval(aug_name)


class ImageTransformForCls(ImageTransformBase):
    """Data Augmentor for Classification Task."""

    def __init__(self, data_augmentations: List[Tuple[str, Dict]]):
        """Initialize."""
        super(ImageTransformForCls, self).__init__(data_augmentations)

    def __call__(self, in_arrs: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Transform."""
        img, label = in_arrs
        augmented = self.data_aug(image=img)
        img = augmented["image"]

        return img, label


class LabeledImageDatasetNumpy(Dataset):
    def __init__(self, file_list, transform_list, copy_in_channels=True, in_channels=3):
        self.file_list = file_list
        self.transform = ImageTransformForCls(transform_list)
        self.copy_in_channels = copy_in_channels
        self.in_channels = in_channels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img, label = self.file_list[index]

        if img.shape[-1] == 2:
            img = img[..., None]

        if self.copy_in_channels:
            img = np.repeat(img, self.in_channels, axis=2)

        img, label = self.transform((img, label))
        return img, label


def train_one_fold(config, train_all, temp_path, print_progress=False):
    torch.backends.cudnn.benchmark = True
    device = torch.device(config['globals']['device'])
    set_seed(config['globals']['seed'])

    # read train, valid image files
    valid_fold = config['globals']['valid_fold']
    valid_idx = train_all[train_all['fold'] == valid_fold].index.values
    train_idx = train_all[train_all['fold'] != valid_fold].index.values
    if config['globals']['debug']:
        train_idx = train_idx[:len(train_idx) // 20]

    data_path = TRAIN_NPY / f'{config["globals"]["dataset_name"]}.npy'

    image_arrs = np.load(str(data_path), mmap_mode='r')
    label_arr = train_all[CLASSES].values.astype('f')

    # Data set
    train_arr_list = [(image_arrs[idx][:, :, None], label_arr[idx]) for idx in train_idx]
    valid_arr_list = [(image_arrs[idx][:, :, None], label_arr[idx]) for idx in valid_idx]

    train_dataset = LabeledImageDatasetNumpy(train_arr_list, **config['dataset']['train'])
    valid_dataset = LabeledImageDatasetNumpy(valid_arr_list, **config['dataset']['valid'])
    logger.debug(f'train_dataset: {len(train_dataset)}')
    logger.debug(f'valid_dataset: {len(valid_dataset)}')

    # DataLoader
    train_loader = DataLoader(train_dataset, **config['loader']['train'])
    valid_loader = DataLoader(valid_dataset, **config['loader']['valid'])

    # model
    model = eval(config['model']['name'])(**config['model']['params'])
    # model = MultiHeadModel(**config['model']['params'])
    model.to(device)

    # optimizer
    optimizer = getattr(torch.optim, config['optimizer']['name'])(model.parameters(), **config['optimizer']['params'])

    # scheduler
    if config['scheduler']['name'] == 'OneCycleLR':
        config['scheduler']['params']['epochs'] = config['globals']['max_epoch']
        config['scheduler']['params']['step_per_epoch'] = len(train_loader)
    scheduler = getattr(torch.optim.lr_scheduler, config['scheduler']['name'])(optimizer, **config['scheduler']['params'])

    # loss
    if hasattr(nn, config['loss']['name']):
        loss_func = getattr(nn, config['loss']['name'])(**config['loss']['params'])
    else:
        loss_func = eval(config['loss']['name'])(**config['loss']['params'])
    loss_func.to(device)

    # Early stopping
    early_stop = EarlyStopping(**config['early_stopping']['params'])

    # Train Loop
    train_losses = []
    valid_losses = []
    iteration = 0
    for epoch in range(config['globals']['max_epoch']):
        logger.info(f'epoch {epoch + 1} / {config["globals"]["max_epoch"]}')

        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            iteration += 1
            # ToDo 慶か時間を記録する

            optimizer.zero_grad()

            y = model(images)
            loss = loss_func(y, labels)
            running_loss += loss

            loss.backward()
            optimizer.step()

            del loss  # 計算グラフの削除によるメモリ節約
        train_losses.append(running_loss / len(train_loader))
        logger.info(f'lr: {scheduler.get_last_lr()[0]}')
        logger.info(f'train loss: {train_losses[-1]:.8f}')
        logger.info(f'iteration: {iteration}')
        scheduler.step()

        # evaluation

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                y = model(images)

                loss = loss_func(y, labels)
                running_loss += loss

                del loss

            valid_losses.append(running_loss / len(valid_loader))
            logger.info(f'valid loss: {valid_losses[-1]:.8f}')

        # save model
        # torch.save(model.state_dict(), f'models_trained/{config["globals"]["name"]}_epoch{epoch + 1}.pth')
        # ToDo パラメータを逐次保存しもっともよいパラメータを呼び出すように変更する
        # ToDo 保存したloss がオブジェクトになっているので改善する

        # early stopping
        if early_stop.step(valid_losses[-1]):
            break

        _ = gc.collect()

    torch.save(model.state_dict(), f'models_trained/{config["globals"]["name"]}.pth')

    epochs = [i + 1 for i in range(len(train_losses))]
    eval_df = pd.DataFrame(index=epochs, columns=['train_eval', 'valid_eval'])
    eval_df['train_eval'] = train_losses
    eval_df['valid_eval'] = valid_losses
    eval_df.to_csv(f'output/{config["globals"]["name"]}_eval.csv')


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False, on=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        self.on = on

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


def main():
    parser = argparse.ArgumentParser(description='pytorch runner')
    parser.add_argument('config_file', help='実行時の設定yamlファイルの読み込み')

    args = parser.parse_args()
    config_file = Path(args.config_file)

    # read config file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    folds = config['globals']['folds']
    n_fold = len(folds)
    n_classes = config['globals']['classes']
    seed = config['globals']['seed']
    config['globals']['name'] = config_file.stem

    if config['globals']['debug']:
        config['globals']['max_epoch'] = 1
        print('!' * 10, 'debug model', '!' * 10)

    # # logging 設定
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)

    # log 標準出力
    handler_st = logging.StreamHandler()
    handler_st.setLevel(logging.DEBUG)
    handler_st_format = logging.Formatter('%(asctime)s %(name)s: %(message)s')
    handler_st.setFormatter(handler_st_format)
    logger.addHandler(handler_st)

    # log ファイル
    log_file = f'log/log_{config["globals"]["name"]}.log'
    handler_f = logging.FileHandler(log_file, 'a')
    handler_f.setLevel(logging.DEBUG)
    handler_f_format = logging.Formatter('%(asctime)s %(name)s: %(message)s')
    handler_f.setFormatter(handler_f_format)
    logger.addHandler(handler_f)

    # make fold data
    label_arr = train[CLASSES].values
    patient_id = train.PatientID.values

    train_valid_indexs = list(multi_label_stratified_group_k_fold(label_arr, patient_id, n_fold=n_fold, seed=seed))

    train['fold'] = -1
    for fold_id, (train_idx, valid_idx) in enumerate(train_valid_indexs):
        train.loc[valid_idx, 'fold'] = fold_id

    configs = []
    for fold_id in range(n_fold):
        temp_config = copy.deepcopy(config)
        temp_config['globals']['valid_fold'] = fold_id
        temp_config['globals']['name'] += f'_fold{fold_id:02d}'
        configs.append(temp_config)

        if config['globals']['debug']:
            break

        if config['globals']['fold1']:
            break

    torch.cuda.empty_cache()
    gc.collect()

    for fold_id, config_fold in enumerate(configs):
        logger.info(f'start train fold : {fold_id}')
        train_one_fold(config_fold, train, TMP/f'fold{fold_id}', False)
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()
