#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import argparse
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


def learning_curves(train, val):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    fig.suptitle("Training Curves")
    ax.plot(train, label="Entraînement")
    ax.plot(val, label="Validation")
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=14)
    fig.savefig("plots/plot.png")
    plt.close(fig)


def get_args():
    """
    Argument parser.

    :return: Object containing all the parameters needed to train a model
    :rtype: Dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", "-e", type=int, default=100, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=16, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument(
        "--att", "-a",
        dest="att",
        action="store_true",
        help="If flag, use attention block",
    )
    parser.add_argument(
        "--size", type=int, default=512, help="Size of the image, one number"
    )
    parser.add_argument("--drop_r", "-d", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--filters", "-f",
        type=int,
        default=8,
        help="Number of filters in first conv block",
    )
    args = parser.parse_args()
    print_red(args)
    return args
