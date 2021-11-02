# -*- coding: utf-8 -*-

import cv2
import numpy as np
import skimage.io as io

"""
3D tiff file -> slices -> patches -> pred -> slices -> 3D
"""

FILE = "/home/edgar/Documents/Datasets/JB/new_images/Images/8_bits/s10.tif"


def create_image_patches(img, size=512):
    """
    Partition an image into multiple patches of approximately equal size.
    The patch size is based on the desired number of rows and columns.
    Returns a list of image patches, in row-major order.
    """

    patch_list = []
    width, height = img.shape[1], img.shape[0]
    w, h = size, size
    for y in range(0, height, h):
        y_end = min(y + h, width)
        for x in range(0, width, w):
            x_end = min(x + w, height)
            patch = img[y:y_end, x:x_end]
            patch_list.append(patch)
    return patch_list


def img_to_slices(img):
    """
    From 3D image to slices list
    """
    res = []

    for i, slice_img in enumerate(img):
        res.append(slice_img)
    return res


def read_and_normalize(im_path):
    """
    Open 3D tiff file, reshape if needed, normalize values
    """
    im3d = io.imread(im_path, plugin="tifffile").astype(np.uint8)
    sh = np.shape(im3d)
    if len(sh) > 3:
        im3d = im3d.reshape(sh[1], sh[2], sh[3])
    return im3d / 255


def reconstruct_image(patch_list, patch_nb=2):
    """
    patch_nb = patch_nb per line
    """
    line_list = []
    for i in range(0, patch_nb ** 2 - 1, patch_nb):
        line_list.append(cv2.hconcat(patch_list[i : i + patch_nb]))
    final_img = cv2.vconcat(line_list)
    return final_img


if __name__ == "__main__":
    im3D = read_and_normalize(FILE)
    slice_list = img_to_slices(im3D)
    reco_list = []  # list containing mask slices
    for slice in slice_list:
        patch_list = create_image_patches(slice)
        # prediction on patches
        # output_list = pred(patch_list)
        reco_list.append(reconstruct_image(patch_list))
    output_stack = np.array(reco_list)
    # todo check if 3D works well
