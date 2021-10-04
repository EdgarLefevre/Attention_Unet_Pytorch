# -*- coding: utf-8 -*-

import numpy as np
import skimage.io as io
import torch

import Attention_Unet_Pytorch.models.unet as unet
import Attention_Unet_Pytorch.utils.postprocessing as pp
import Attention_Unet_Pytorch.utils.utils as utils


def pred(net, imgs):
    output, _ = net(torch.Tensor(imgs).cuda())
    return output


def load_model(path):
    model = unet.Unet(filters=8, attention=True).cuda()
    model.load_state_dict(torch.load(path))
    return model


def yield_img(file, size=512):
    img = (np.array(io.imread(file)) / 255).reshape(1, 1, size, size)
    return img


def get_images(path_folder):
    file_list = utils.list_files_path(path_folder)
    img_list = []
    for file in file_list:
        img_list.append(yield_img(file))
    return img_list, file_list


if __name__ == "__main__":
    net = load_model("saved_models/net.pth")
    # print(net)
    img_list, file_list = get_images("/home/edgar/Documents/Datasets/JB/good/")
    for i, img in enumerate(img_list):
        file_name = file_list[i].split("/")[-1]
        print(file_name)
        mask = pred(net, img)
        mask = mask.cpu().detach().numpy()
        mask = pp.remove_blobs((mask > 0.5).astype(np.uint8).reshape(512, 512) * 255)
        io.imsave("/home/edgar/Documents/Datasets/JB/good_labels/" + file_name, mask)
