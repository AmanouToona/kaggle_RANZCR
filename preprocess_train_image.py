import cv2
import numpy as np
import gc

from pathlib import Path
from tqdm import tqdm

from joblib import Parallel, delayed
import pandas as pd


ROOT = Path.cwd()
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
DATA = INPUT / "ranzcr-clip-catheter-line-classification"
TRAIN = DATA / "train"
TEST = DATA / "test"


TRAIN_NPY = INPUT / "ranzcr-clip-train-numpy"
TMP = ROOT / "tmp"
TMP.mkdir(exist_ok=True)

RANDAM_SEED = 1086
N_CLASSES = 11
FOLDS = [0, 1, 2, 3, 4]
N_FOLD = len(FOLDS)


train = pd.read_csv(DATA / "train.csv")
smpl_sub = pd.read_csv(DATA / "sample_submission.csv")


def resize_images(img_id, input_dir, output_dir, resize_to=(512, 512)):
    img_path = input_dir / (img_id + ".jpg")
    save_path = output_dir / (img_id + ".jpg")

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, resize_to)
    cv2.imwrite(str(save_path), img, )
    return


# IMAGE_SIZE = (320, 320)
IMAGE_SIZE = (512, 512)
TRAIN_RESIZED = TMP / "train_{0}x{1}".format(*IMAGE_SIZE)
TRAIN_RESIZED.mkdir(exist_ok=True)


_ = Parallel(n_jobs=-1, verbose=5)([
    delayed(resize_images)(img_id, TRAIN, TRAIN_RESIZED, IMAGE_SIZE)
    for img_id in train.StudyInstanceUID.values
])


def save_as_numpy(input_dir, output_path, meta_file, size=(512, 512), ext="jpg"):
    arr = np.zeros((len(meta_file), *size), dtype="uint8")
    for idx, img_id in enumerate(tqdm(meta_file["StudyInstanceUID"].values)):
        img_path = input_dir / f"{img_id}.{ext}"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        arr[idx] = img

    np.save(output_path, arr)
    return arr


train_arr_320 = save_as_numpy(TRAIN_RESIZED.resolve(), TMP / "train_{0}x{1}.npy".format(*IMAGE_SIZE), train, IMAGE_SIZE)


del train_arr_320
gc.collect()
