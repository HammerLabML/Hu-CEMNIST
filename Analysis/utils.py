import os
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import idx2numpy
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
import tensorflow_datasets as tfds


def load_mnist():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        return tf.cast(tf.reshape(image, (28*28,)), tf.float32) / 255., label

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    X_train, y_train = [], []
    for x, y in tfds.as_numpy(ds_train):
        X_train.append(x)
        y_train.append(y)
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train)

    X_test, y_test = [], []
    for x, y in tfds.as_numpy(ds_test):
        X_test.append(x)
        y_test.append(y)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test)

    return (X_train, y_train), (X_test, y_test)


def load_original_images(path_in="../Server/site/static/target_images_SSIM/"):
    data = [{"label": 0, "idx": 13280, "file": "digit_0_best_SSIM_idx_13280_rescaled.png"},
            {"label": 1, "idx": 22398, "file": "digit_1_best_SSIM_idx_22398_rescaled.png"},
            {"label": 2, "idx": 52316, "file": "digit_2_best_SSIM_idx_52316_rescaled.png"},
            {"label": 3, "idx": 21717, "file": "digit_3_best_SSIM_idx_21717_rescaled.png"},
            {"label": 4, "idx": 47566, "file": "digit_4_best_SSIM_idx_47566_rescaled.png"},
            {"label": 5, "idx": 5752, "file": "digit_5_best_SSIM_idx_5752_rescaled.png"},
            {"label": 6, "idx": 3774, "file": "digit_6_best_SSIM_idx_3774_rescaled.png"},
            {"label": 7, "idx": 66586, "file": "digit_7_best_SSIM_idx_66586_rescaled.png"},
            {"label": 8, "idx": 51272, "file": "digit_8_best_SSIM_idx_51272_rescaled.png"},
            {"label": 9, "idx": 60471, "file": "digit_9_best_SSIM_idx_60471_rescaled.png"}]

    for i in range(len(data)):
        with Image.open(os.path.join(path_in, data[i]["file"])) as img:
            img = ImageOps.grayscale(img)
            img = img.resize((28, 28))

            data[i]["img"] = (255 - np.array(img.getdata()).flatten()) / 255.

    return data


def load_human_generated_counterfactuals(path_in: str = "../Data/Hu-CEMNIST_valid_images/"):
    user_info = pd.read_csv(os.path.join(path_in, "user_info_complete.csv"))
    user_info = user_info[["userID", "trial1_baseDigit", "trial1_targetDigit",
                           "trial2_baseDigit", "trial2_targetDigit",
                           "trial3_baseDigit", "trial3_targetDigit"]]

    r = []

    for _, (userID, trial1_baseDigit, trial1_targetDigit, trial2_baseDigit, trial2_targetDigit,
            trial3_baseDigit, trial3_targetDigit) in user_info.iterrows():
        def _load_img(trial_id: int) -> np.ndarray:
            padding_userid = 8 - len(str(userID))
            f_in = os.path.join(path_in, f"{''.join(['0']*padding_userid) + str(userID)}_trial{trial_id}.npz")
            if os.path.exists(f_in):
                img = np.load(f_in)["final_image"]
                img = ImageOps.grayscale(Image.fromarray(img))
                img = img.resize((28, 28))

                return 255 - np.array(img.getdata()).flatten() / 255.
            else:
                return None

        # Load images
        img1 = _load_img(1)
        img2 = _load_img(2)
        img3 = _load_img(3)

        if img1 is not None:
            r += [(trial1_baseDigit, trial1_targetDigit, img1)]
        if img2 is not None:
            r += [(trial2_baseDigit, trial2_targetDigit, img2)]
        if img3 is not None:
            r += [(trial3_baseDigit, trial3_targetDigit, img3)]

    return r



def load_kaggle_mnist():
    imagefile_train = '../Data/train-images.idx3-ubyte'
    imagefile_test = '../Data/t10k-images.idx3-ubyte'
    imagearray_train = idx2numpy.convert_from_file(imagefile_train)
    imagearray_test = idx2numpy.convert_from_file(imagefile_test)

    imagearray_all = np.concatenate((imagearray_train, imagearray_test), axis=0)

    """
    digit_0_best_SSIM_idx_13280_rescaled.png —> imagearray_train[13280]
    digit_1_best_SSIM_idx_22398_rescaled.png —> imagearray_train[22398]
    digit_2_best_SSIM_idx_52316_rescaled.png —> imagearray_train[52316]
    digit_3_best_SSIM_idx_21717_rescaled.png —> imagearray_train[21717]
    digit_4_best_SSIM_idx_47566_rescaled.png —> imagearray_train[47566]
    digit_5_best_SSIM_idx_5752_rescaled.png —> imagearray_train[5752]
    digit_6_best_SSIM_idx_3774_rescaled.png —> imagearray_train[3774]
    digit_7_best_SSIM_idx_66586_rescaled.png —> imagearray_test[6586]
    digit_8_best_SSIM_idx_51272_rescaled.png —> imagearray_train[51272]
    digit_9_best_SSIM_idx_60471_rescaled.png —> imagearray_test[471]
    """

    return imagearray_all


def compute_percentage_change(img_orig, img_cf) -> float:
    score = 0.

    for i in range(len(img_orig)):
        if img_orig[i] != img_cf[i]:
            score += 1

    score /= len(img_orig)

    return score


def compare_area_of_change(area_a: np.ndarray, area_b: np.ndarray) -> float:
    idx_a = set(np.argwhere(area_a != 0).flatten().tolist())
    idx_b = set(np.argwhere(area_b != 0).flatten().tolist())

    return len(idx_a.intersection(idx_b)) / len(idx_a.union(idx_b))


def compute_img_quality(img_cf, ground_truth_imgs) -> float:
    s = []

    for x in ground_truth_imgs:
        s.append(ssim(x.reshape((28, 28)), img_cf.reshape((28, 28)),
                      data_range=x.max() - x.min()))

    return np.max(s)


def compute_area_of_change(img_orig, img_cf):
    return np.array(img_orig != img_cf).astype(np.float32)
