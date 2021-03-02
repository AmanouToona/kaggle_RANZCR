import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
import typing as tp
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
import albumentations
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from pytorch_pfn_extras.training import extensions as ppe_extensions
import pytorch_pfn_extras as ppe
from sklearn.metrics import roc_auc_score
import pandas as pd
import os
import yaml
import random
import tqdm
import shutil
import copy
from models import *
from pathlib import Path
from scipy.sparse import coo_matrix
import time


# ROOT = Path.cwd().parent
ROOT = Path.cwd()
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


RANDAM_SEED = 1086
N_CLASSES = 11
FOLDS = [0, 1, 2, 3, 4]
N_FOLD = len(FOLDS)

train = pd.read_csv(DATA / "train.csv")
smpl_sub = pd.read_csv(DATA / "sample_submission.csv")


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

    groups_by_fold = [[] for fid in range(n_fold)]
    group_and_cnts = list(enumerate(cnts_by_group))  # pair of aid and cnt by group
    np.random.shuffle(group_and_cnts)
    print("finished preparation", time.time() - start_time)
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
    print("finished assignment.", time.time() - start_time)

    gc.collect()
    idx_arr = np.arange(n_train)
    for fid in range(n_fold):
        val_groups = groups_by_fold[fid]

        val_indexs_bool = np.isin(aid_arr, val_groups)
        train_indexs = idx_arr[~val_indexs_bool]
        val_indexs = idx_arr[val_indexs_bool]

        print("[fold {}]".format(fid), end=" ")
        print("n_group: (train, val) = ({}, {})".format(n_group - len(val_groups), len(val_groups)), end=" ")
        print("n_sample: (train, val) = ({}, {})".format(len(train_indexs), len(val_indexs)))

        yield train_indexs, val_indexs


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


class ImageTransformBase:
    """
    Base Image Transform class.

    Args:
        data_augmentations: List of tuple(method: str, params :dict), each elems pass to albumentations
    """

    def __init__(self, data_augmentations: tp.List[tp.Tuple[str, tp.Dict]]):
        """Initialize."""
        augmentations_list = [
            self._get_augmentation(aug_name)(**params)
            for aug_name, params in data_augmentations]
        self.data_aug = albumentations.Compose(augmentations_list)

    def __call__(self, pair: tp.Tuple[np.ndarray]) -> tp.Tuple[np.ndarray]:
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

    def __init__(self, data_augmentations: tp.List[tp.Tuple[str, tp.Dict]]):
        """Initialize."""
        super(ImageTransformForCls, self).__init__(data_augmentations)

    def __call__(self, in_arrs: tp.Tuple[np.ndarray, np.ndarray]) -> tp.Tuple[np.ndarray, np.ndarray]:
        """Apply Transform."""
        img, label = in_arrs
        augmented = self.data_aug(image=img)
        img = augmented["image"]

        return img, label


def get_file_list_with_array(stgs, train_all):
    """Get file path and target info."""
    # train_all = pd.read_csv(DATA / stgs["globals"]["meta_file"])
    use_fold = stgs["globals"]["val_fold"]
    train_idx = train_all[train_all["fold"] != use_fold].index.values
    if stgs["globals"]["debug"]:
        train_idx = train_idx[:len(train_idx) // 20]
    val_idx = train_all[train_all["fold"] == use_fold].index.values

    train_data_path = TRAIN_NPY / "{}.npy".format(stgs["globals"]["dataset_name"])
    # train_data_arr = np.load(train_data_path)
    train_data_arr = np.load(str(train_data_path), mmap_mode="r")
    label_arr = train_all[CLASSES].values.astype("f")  # ToDo astype 必要？


    train_file_list = [
        (train_data_arr[idx][..., None], label_arr[idx]) for idx in train_idx]
    val_file_list = [
        (train_data_arr[idx][..., None], label_arr[idx]) for idx in val_idx]

    return train_file_list, val_file_list


def get_dataloaders_cls(stgs: tp.Dict, train_file_list: tp.List[np.array], val_file_list: tp.List[np.array],
                        dataset_class: Dataset):
    """Create DataLoader"""
    train_loader = val_loader = None
    if train_file_list is not None:
        train_dataset = dataset_class(train_file_list, **stgs["dataset"]["train"])
        print(f'train_dataset: {len(train_dataset)}')
        train_loader = data.DataLoader(train_dataset, **stgs["loader"]["train"])

    if val_file_list is not None:
        val_dataset = dataset_class(val_file_list, **stgs["dataset"]["val"])
        val_loader = data.DataLoader(val_dataset, **stgs["loader"]["val"])

    return train_loader, val_loader


class EvalFuncManager(nn.Module):
    """Manager Class for evaluation at the end of epoch"""

    def __init__(
            self,
            iters_per_epoch: int,
            evalfunc_dict: tp.Dict[str, nn.Module],
            prefix: str = "val"
    ) -> None:
        """Initialize"""
        self.tmp_iter = 0
        self.iters_per_epoch = iters_per_epoch
        self.prefix = prefix
        self.metric_names = []
        super(EvalFuncManager, self).__init__()
        for k, v in evalfunc_dict.items():
            setattr(self, k, v)
            self.metric_names.append(k)
        self.reset()

    def reset(self) -> None:
        """Reset State."""
        self.tmp_iter = 0
        for name in self.metric_names:
            getattr(self, name).reset()

    def __call__(self, y: torch.Tensor, t: torch.Tensor) -> None:
        """Forward."""
        for name in self.metric_names:
            getattr(self, name).update(y, t)
        self.tmp_iter += 1

        if self.tmp_iter == self.iters_per_epoch:
            ppe.reporting.report({
                "{}/{}".format(self.prefix, name): getattr(self, name).compute()
                for name in self.metric_names
            })
            self.reset()


class MeanLoss(nn.Module):

    def __init__(self):
        super(MeanLoss, self).__init__()
        self.loss_sum = 0
        self.n_examples = 0

    def forward(self, y: torch.Tensor, t: torch.Tensor):
        """Compute metric at once"""
        return self.loss_func(y, t)

    def reset(self):
        """Reset state"""
        self.loss_sum = 0
        self.n_examples = 0

    def update(self, y: torch.Tensor, t: torch.Tensor):
        """Update metric by mini batch"""
        self.loss_sum += float(self(y, t).item() * y.shape[0])
        self.n_examples += y.shape[0]

    def compute(self):
        """Compute metric for dataset"""
        return self.loss_sum / self.n_examples


class MyLogLoss(MeanLoss):

    def __init__(self, **params):
        super(MyLogLoss, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss(**params)


class MyROCAUC(nn.Module):
    """ROC AUC score"""

    def __init__(self, average="macro") -> None:
        """Initialize."""
        self.average = average
        self._pred_list = []
        self._true_list = []
        super(MyROCAUC, self).__init__()

    def reset(self) -> None:
        """Reset State."""
        self._pred_list = []
        self._true_list = []

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Forward."""
        self._pred_list.append(y_pred.detach().cpu().numpy())
        self._true_list.append(y_true.detach().cpu().numpy())

    def compute(self) -> float:
        """Calc and return metric value."""
        y_pred = np.concatenate(self._pred_list, axis=0)
        y_true = np.concatenate(self._true_list, axis=0)
        score = roc_auc_score(y_true, y_pred, average=self.average)
        return score

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Forward."""
        self.reset()
        self.update(y_pred, y_true)
        return self.compute()


def set_random_seed(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = deterministic  # type: ignore


def get_stepper(manager, stgs, scheduler):
    """"""

    def dummy_step():
        pass

    def step():
        scheduler.step()

    def step_with_epoch_detail():
        scheduler.step(manager.epoch_detail)

    if stgs["scheduler"]["name"] == None:
        return dummy_step, dummy_step

    elif stgs["scheduler"]["name"] == "CosineAnnealingWarmRestarts":
        return dummy_step, step_with_epoch_detail

    elif stgs["scheduler"]["name"] == "OneCycleLR":
        return dummy_step, step

    else:
        return step, dummy_step


def run_train_loop(
        manager, stgs, model, device, train_loader, optimizer, scheduler, loss_func
):
    """Run minibatch training loop"""
    step_scheduler_by_epoch, step_scheduler_by_iter = get_stepper(manager, stgs, scheduler)

    if stgs["globals"]["use_amp"]:
        while not manager.stop_trigger:
            model.train()
            print('len train_loader', len(train_loader))
            scaler = torch.cuda.amp.GradScaler()
            for x, t in train_loader:
                with manager.run_iteration():
                    x, t = x.to(device), t.to(device)
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        y = model(x)
                        loss = loss_func(y, t)
                    ppe.reporting.report({'train/loss': loss.item()})
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    step_scheduler_by_iter()
            step_scheduler_by_epoch()
    else:
        while not manager.stop_trigger:
            model.train()
            for x, t in train_loader:
                with manager.run_iteration():
                    x, t = x.to(device), t.to(device)
                    optimizer.zero_grad()
                    y = model(x)
                    loss = loss_func(y, t)
                    ppe.reporting.report({'train/loss': loss.item()})
                    loss.backward()
                    optimizer.step()
                    step_scheduler_by_iter()
            step_scheduler_by_epoch()


def run_eval(stgs, model, device, batch, eval_manager):
    """Run evaliation for val or test. this function is applied to each batch."""
    model.eval()
    x, t = batch
    if stgs["globals"]["use_amp"]:
        with torch.cuda.amp.autocast():
            y = model(x.to(device))
            eval_manager(y, t.to(device))
    else:
        y = model(x.to(device))
        eval_manager(y, t.to(device))


def set_extensions(
        manager, args, model, device,
        val_loader, optimizer,
        eval_manager, print_progress: bool = False,
):
    """Set extensions for PPE"""
    eval_names = ["val/{}".format(name) for name in eval_manager.metric_names]

    log_extentions = [
        ppe_extensions.observe_lr(optimizer=optimizer),
        ppe_extensions.LogReport(),
        ppe_extensions.PlotReport(["train/loss", "val/loss"], 'epoch', filename='loss.png'),
        ppe_extensions.PlotReport(["lr"], 'epoch', filename='lr.png'),
        ppe_extensions.PrintReport([
            "epoch", "iteration", "lr", "train/loss", *eval_names, "elapsed_time"])
    ]
    if print_progress:
        log_extentions.append(ppe_extensions.ProgressBar(update_interval=20))

    for ext in log_extentions:
        manager.extend(ext)

    manager.extend(  # evaluation
        ppe_extensions.Evaluator(
            val_loader, model,
            eval_func=lambda *batch: run_eval(args, model, device, batch, eval_manager)),
        trigger=(1, "epoch"))

    manager.extend(  # model snapshot
        ppe_extensions.snapshot(target=model, filename="snapshot_epoch_{.epoch}.pth"),
        trigger=ppe.training.triggers.MaxValueTrigger(key="val/metric", trigger=(1, 'epoch')))

    return manager


def train_one_fold(settings, train_all, output_path, print_progress=False):
    """train one fold"""
    torch.backends.cudnn.benchmark = True
    set_random_seed(settings["globals"]["seed"])

    # # prepare train, valid paths
    # train_file_list, val_file_list = get_file_list(settings, train_all, "png")
    train_file_list, val_file_list = get_file_list_with_array(settings, train_all)
    print("train: {}, val: {}".format(len(train_file_list), len(val_file_list)))

    device = torch.device(settings["globals"]["device"])
    # # get data_loader
    train_loader, val_loader = get_dataloaders_cls(
        settings, train_file_list, val_file_list, LabeledImageDatasetNumpy)

    # # get model
    model = MultiHeadModel(**settings["model"]["params"])
    model.to(device)

    # # get optimizer
    optimizer = getattr(
        torch.optim, settings["optimizer"]["name"]
    )(model.parameters(), **settings["optimizer"]["params"])

    # # get scheduler
    if settings["scheduler"]["name"] == "OneCycleLR":
        settings["scheduler"]["params"]["epochs"] = settings["globals"]["max_epoch"]
        settings["scheduler"]["params"]["steps_per_epoch"] = len(train_loader)
    scheduler = getattr(
        torch.optim.lr_scheduler, settings["scheduler"]["name"]
    )(optimizer, **settings["scheduler"]["params"])

    # # get loss
    if hasattr(nn, settings["loss"]["name"]):
        loss_func = getattr(nn, settings["loss"]["name"])(**settings["loss"]["params"])
    else:
        loss_func = eval(settings["loss"]["name"])(**settings["loss"]["params"])
    loss_func.to(device)

    eval_manager = EvalFuncManager(
        len(val_loader), {
            metric["report_name"]: eval(metric["name"])(**metric["params"])
            for metric in settings["eval"]
        })
    eval_manager.to(device)

    # # get manager
    # trigger = None
    trigger = ppe.training.triggers.EarlyStoppingTrigger(
        check_trigger=(1, 'epoch'),
        # monitor='val/metric', mode="min",
        monitor='val/metric', mode="max",
        patience=settings["globals"]["patience"], verbose=False,
        max_trigger=(settings["globals"]["max_epoch"], 'epoch'),
    )
    manager = ppe.training.ExtensionsManager(
        model, optimizer, settings["globals"]["max_epoch"],
        iters_per_epoch=len(train_loader),
        stop_trigger=trigger, out_dir=output_path
    )
    manager = set_extensions(
        manager, settings, model, device, val_loader, optimizer, eval_manager, print_progress)

    # # run training.
    run_train_loop(
        manager, settings, model, device, train_loader,
        optimizer, scheduler, loss_func)


def main():

    stgs_str = """
    globals:
      seed: 1086
      device: cuda
      max_epoch: 8
      patience: 3
      dataset_name: train_512x512
      use_amp: True
      val_fold: 0
      debug: False
    
    dataset:
      name: LabeledImageDatasetNumpy
      train:
        transform_list:
          - [HorizontalFlip, {p: 0.5}]
          - [ShiftScaleRotate, {
              p: 0.5, shift_limit: 0.2, scale_limit: 0.2,
              rotate_limit: 20, border_mode: 0, value: 0, mask_value: 0}]
          - [RandomResizedCrop, {height: 512, width: 512, scale: [0.9, 1.0]}]
          - [Cutout, {max_h_size: 51, max_w_size: 51, num_holes: 5, p: 0.5}]
          - [Normalize, {
              always_apply: True, max_pixel_value: 255.0,
              mean: [0.4887381077884414], std: [0.23064819430546407]}]
          - [ToTensorV2, {always_apply: True}]
      val:
        transform_list:
          - [Normalize, {
              always_apply: True, max_pixel_value: 255.0,
              mean: [0.4887381077884414], std: [0.23064819430546407]}]
          - [ToTensorV2, {always_apply: True}]
    
    loader:
      train: {batch_size: 8, shuffle: True, num_workers: 2, pin_memory: True, drop_last: True}
      val: {batch_size: 16, shuffle: False, num_workers: 2, pin_memory: True, drop_last: False}
    
    model:
      name: MultiHeadModel
      params:
        base_name: regnety_032
        out_dims_head: [3, 4, 3, 1]
        pretrained: True
    
    loss: {name: BCEWithLogitsLoss, params: {}}
    
    eval:
      - {name: MyLogLoss, report_name: loss, params: {}}
      - {name: MyROCAUC, report_name: metric, params: {average: macro}}
    
    
    
    optimizer:
        name: Adam
        params:
          lr: 1.0e-03
    
    scheduler:
      name: CosineAnnealingWarmRestarts
      params:
        T_0: 8
        T_mult: 1
    """

    label_arr = train[CLASSES].values
    group_id = train.PatientID.values

    train_val_indexs = list(
        multi_label_stratified_group_k_fold(label_arr, group_id, N_FOLD, RANDAM_SEED))

    train["fold"] = -1
    for fold_id, (trn_idx, val_idx) in enumerate(train_val_indexs):
        train.loc[val_idx, "fold"] = fold_id

    # train.groupby("fold")[CLASSES].sum()

    stgs = yaml.safe_load(stgs_str)

    if stgs["globals"]["debug"]:
        stgs["globals"]["max_epoch"] = 1

    stgs_list = []
    for fold_id in range(N_FOLD):
        tmp_stgs = copy.deepcopy(stgs)
        tmp_stgs["globals"]["val_fold"] = fold_id
        stgs_list.append(tmp_stgs)

    torch.cuda.empty_cache()
    gc.collect()

    for fold_id, tmp_stgs in enumerate(stgs_list):
        train_one_fold(tmp_stgs, train, TMP / f"fold{fold_id}", False)
        torch.cuda.empty_cache()
        gc.collect()

    best_log_list = list()
    for fold_id, tmp_stgs in enumerate(stgs_list):
        exp_dir_path = TMP / f"fold{fold_id}"
        log = pd.read_json(exp_dir_path / "log")
        best_log = log.iloc[[log["val/metric"].idxmax()],]
        best_epoch = best_log.epoch.values[0]
        best_log_list.append(best_log)

        best_model_path = exp_dir_path / f"snapshot_epoch_{best_epoch}.pth"
        copy_to = f"./best_model_fold{fold_id}.pth"
        shutil.copy(best_model_path, copy_to)

        for p in exp_dir_path.glob("*.pth"):
            p.unlink()

        shutil.copytree(exp_dir_path, f"./fold{fold_id}")

        with open(f"./fold{fold_id}/settings.yml", "w") as fw:
            yaml.dump(tmp_stgs, fw)

    pd.concat(best_log_list, axis=0, ignore_index=True)


    def run_inference_loop(stgs, model, loader, device):
        model.to(device)
        model.eval()
        pred_list = []
        with torch.no_grad():
            for x, t in tqdm(loader):
                y = model(x.to(device))
                pred_list.append(y.sigmoid().detach().cpu().numpy())
                # pred_list.append(y.detach().cpu().numpy())

        pred_arr = np.concatenate(pred_list)
        del pred_list
        return pred_arr


    oof_pred_arr = np.zeros((len(train), N_CLASSES))
    label_arr = train[CLASSES].values
    score_list = []

    for fold_id in range(N_FOLD):
        tmp_dir = Path(f"./fold{fold_id}")
        with open(tmp_dir / "settings.yml", "r") as fr:
            tmp_stgs = yaml.safe_load(fr)
        device = torch.device(tmp_stgs["globals"]["device"])
        val_idx = train.query("fold == @fold_id").index.values

        # # get data_loader
        _, val_file_list = get_file_list_with_array(tmp_stgs, train)
        _, val_loader = get_dataloaders_cls(
            tmp_stgs, None, val_file_list, LabeledImageDatasetNumpy)

        # # get and load model
        model_path = f"./best_model_fold{fold_id}.pth"
        # model = SingleHeadModel(**tmp_stgs["model"]["params"])
        model = MultiHeadModel(**tmp_stgs["model"]["params"])
        model.load_state_dict(torch.load(model_path, map_location=device))

        val_pred = run_inference_loop(tmp_stgs, model, val_loader, device)
        val_score = roc_auc_score(label_arr[val_idx], val_pred, average="macro")
        print(f"[fold {fold_id}] val score: {val_score:.5f}")
        oof_pred_arr[val_idx] = val_pred
        score_list.append([fold_id, val_score])
        break


    oof_score = roc_auc_score(label_arr, oof_pred_arr)
    score_list.append(["oof", oof_score])


    pd.DataFrame(score_list, columns=["fold", "metric"])


    oof_df = train.copy()
    oof_df[CLASSES] = oof_pred_arr
    oof_df.to_csv("./oof_prediction.csv", index=False)


if __name__ == '__main__':
    main()