#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np


def list_files_path(path):
    """
    List files from a path.

    :param path: Folder path
    :type path: str
    :return: A list containing all files in the folder
    :rtype: List
    """
    return sorted_alphanumeric(
        [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    )


def shuffle_lists(lista, listb, seed=42):
    """
    Shuffle two list with the same seed.

    :param lista: List of elements
    :type lista: List
    :param listb: List of elements
    :type listb: List
    :param seed: Seed number
    :type seed: int
    :return: lista and listb shuffled
    :rtype: (List, List)
    """
    random.seed(seed)
    random.shuffle(lista)
    random.seed(seed)
    random.shuffle(listb)
    return lista, listb


def print_red(skk):
    """
    Print in red.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[91m{}\033[00m".format(skk))


def print_gre(skk):
    """
    Print in green.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[92m{}\033[00m".format(skk))


def sorted_alphanumeric(data):
    """
    Sort function.

    :param data: str list
    :type data: List
    :return: Sorted list
    :rtype: List
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa
    return sorted(data, key=alphanum_key)


def visualize(imgs, pred):
    fig = plt.figure(figsize=(15, 10))
    columns = 2
    rows = 5  # nb images
    ax = []  # loop around here to plot more images
    i = 0
    for j, img in enumerate(imgs):
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("Input")
        plt.imshow(img, cmap="gray")
        ax.append(fig.add_subplot(rows, columns, i + 2))
        ax[-1].set_title("Mask")
        plt.imshow(pred[j], cmap="gray")
        i += 2
        if i >= 15:
            break
    # plt.show()
    fig.savefig("plots/prediction.png")
    plt.close(fig)


def plot_att_map(img, map):
    fig = plt.figure(figsize=(15, 10))
    columns = 2
    rows = 1  # nb images
    ax = []
    ax.append(fig.add_subplot(rows, columns, 1))
    ax[-1].set_title("Input")
    plt.imshow(img, cmap="gray")
    ax.append(fig.add_subplot(rows, columns, 2))
    ax[-1].set_title("Attention map")
    map = np.array(map).reshape(256, 256)
    plt.imshow(map * 255)
    plt.colorbar()
    # plt.show()
    fig.savefig("plots/att_map.png")
    plt.close(fig)
