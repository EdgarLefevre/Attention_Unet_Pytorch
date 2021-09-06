# -*- coding: utf-8 -*-
import torch.nn as nn
import Attention_Unet_Pytorch.models.unet as unet
import Attention_Unet_Pytorch.utils.data as data
import Attention_Unet_Pytorch.utils.utils as utils
import torch.optim as optim
import sklearn.model_selection as sk
import torch
import progressbar
import skimage.io as io
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# widget list for the progress bar
widgets = [
    " [",
    progressbar.Timer(),
    "] ",
    progressbar.Bar(),
    " (",
    progressbar.ETA(),
    ") ",
]

BASE_PATH = "/home/edgar/Documents/Datasets/JB/supervised/"


def get_datasets(path_img, path_label):
    img_path_list = utils.list_files_path(path_img)
    label_path_list = utils.list_files_path(path_label)
    img_path_list, label_path_list = utils.shuffle_lists(img_path_list, label_path_list)

    # not good if we need to do metrics
    img_train, img_val, label_train, label_val = sk.train_test_split(
        img_path_list, label_path_list, test_size=0.2, random_state=42
    )
    dataset_train = data.JB_Dataset(16, 512, img_train, label_train)
    dataset_val = data.JB_Dataset(16, 512, img_val, label_val)
    return dataset_train, dataset_val


def train(path_imgs, path_labels, epochs=5):
    net = unet.Unet(8, attention=True).cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    #  get dataset
    dataset_train, dataset_val = get_datasets(path_imgs, path_labels)

    for epoch in range(epochs):
        utils.print_gre("Epoch {}/{}".format(epoch+1, epochs))
        with progressbar.ProgressBar(max_value=len(dataset_train), widgets=widgets) as bar:
            net.train()
            for i in range(len(dataset_train)):  # boucle inf si on ne fait pas comme ça
                bar.update(i)
                imgs, labels = dataset_train[i]
                optimizer.zero_grad()  # zero the gradient buffers
                output, _ = net(imgs.cuda())  # return attention map
                loss_train = criterion(output, labels.cuda())
                loss_train.backward()
                optimizer.step()
                # print(loss_train.item())
        with progressbar.ProgressBar(max_value=len(dataset_val), widgets=widgets) as bar2:
            net.eval()
            for j in range(len(dataset_val)):
                bar2.update(j)
                output, _ = net(imgs.cuda())
                loss_val = criterion(output, labels.cuda())
        utils.print_gre("Loss train {}\nLoss val {}".format(loss_train, loss_val))
    pred(net)


def create_pred_dataset(path_img):
    img = io.imread(path_img).astype(np.uint8)
    img = np.array(img) / 255
    img = data.contrast_and_reshape(img)
    img = np.array(img).reshape(-1, 1, 512, 512)
    return img


def pred_(model, path_list):
    pred_list = []
    img_list = []
    for path in path_list:
        img = create_pred_dataset(path)
        res, att_map = model(torch.Tensor(img).cuda())
        res = np.array(res.detach().cpu())
        img_list.append(img.reshape(512, 512) * 255)
        pred_list.append((res > 0.5).astype(np.uint8).reshape(512, 512) * 255)
    return img_list, pred_list, att_map


def pred(model):
    base_path = BASE_PATH + "test/"
    pathlist = [
        base_path + "Spheroid_D31000_02_w2soSPIM-405_135_5.png",
        base_path + "Spheroid_D31000_02_w2soSPIM-405_135_6.png",
        base_path + "Spheroid_D31000_02_w2soSPIM-405_135_9.png",
        base_path + "Spheroid_D31000_02_w2soSPIM-405_136_5.png",
        base_path + "Spheroid_D31000_02_w2soSPIM-405_136_9.png"
    ]
    imgs, preds, att_map = pred_(model, pathlist)
    utils.visualize(imgs, preds)
    utils.plot_att_map(imgs[-1], att_map.detach().cpu())

if __name__ == "__main__":
    train(BASE_PATH + "imgs/", BASE_PATH + "labels/", epochs=10)

