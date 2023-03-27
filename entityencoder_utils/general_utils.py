import os
import shutil
import numpy as np
import tomli
import torch


def hardcoded_log(n_categories: int) -> int:
    """
    A hardcoded logarithmic function

    :param n_categories: int, num of categories of one categorical feature
    :return: hardcoded logarithmic of num of categories
    """
    if n_categories <= 2:
        size_meb = 1
    elif n_categories < 8:
        size_meb = 2
    elif n_categories < 32:
        size_meb = 3
    else:
        size_meb = 4
    return size_meb


def reconstruct_cat_features_1nn(
        synthetic_cat: torch.Tensor,
        nn_classifiers: list,
        embedding_sizes: list
) -> np.ndarray:
    """
    Reconstruct categorical features with 1nn classifier.

    :param synthetic_cat: tensor
    :param nn_classifiers: list of 1nn classifiers
    :param embedding_sizes: list of embedding info
    :return: np.ndarray
    """

    cat_list = []
    i = 0
    for idx, t in enumerate(embedding_sizes):
        x_cat = synthetic_cat[:, i:i + t[1]]
        i += embedding_sizes[idx][1]
        res = nn_classifiers[idx].predict(x_cat)
        cat_list.append(res)
    reconstructed_cat = np.stack(cat_list, axis=1)
    return reconstructed_cat


def tensor2ndarray(t) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    if t.device == torch.device('cpu'):
        return t.numpy()
    else:
        return t.detach().cpu().numpy()


def save_config(parent_dir, config_path):
    """
    Copy paste configure file to parent dir
    """
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass


def load_config(path):
    with open(path, 'rb') as f:
        return tomli.load(f)