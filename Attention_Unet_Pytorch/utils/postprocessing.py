#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def remove_blobs(img, min_size=4):
    img = 255 - img
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8
    )
    sizes = stats[1:, -1]
    nb_components = nb_components - 1  # remove background
    img2 = np.zeros(output.shape)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    # img2 = cv2.bitwise_not(img2 / 255) * 255
    return 255 - img2


def remove_blobs_list(imgs):
    res = []
    for im in imgs:
        res.append(remove_blobs(im))
    return np.array(res)
