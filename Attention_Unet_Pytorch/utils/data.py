import numpy as np
import skimage.exposure as exposure
import skimage.io as io
import torch
from torch.utils.data import Dataset


def contrast_and_reshape(img):
    """
    For some mice, we need to readjust the contrast.

    :param img: Slices of the mouse we want to segment
    :type img: np.array
    :return: Images list with readjusted contrast
    :rtype: np.array

    .. warning:
       If the contrast pf the mouse should not be readjusted,
        the network will fail prediction.
       Same if the image should be contrasted and you do not run it.
    """
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return np.array(img_adapteq)


class JB_Dataset(Dataset):
    def __init__(
        self, batch_size, img_size, input_img_paths, label_img_paths, contrast=True
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.label_img_paths = label_img_paths
        self.contrast = contrast
        assert len(input_img_paths) == len(label_img_paths)
        print("Nb of images : {}".format(len(input_img_paths)))

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        Returns tuple (input, target) correspond to batch #idx.
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_label_img_paths = self.label_img_paths[i : i + self.batch_size]
        x = np.zeros(
            (self.batch_size, 1, self.img_size, self.img_size), dtype="float32"
        )
        for j, path in enumerate(batch_input_img_paths):
            img = np.array(io.imread(path)) / 255
            if self.contrast:
                img = contrast_and_reshape(img)
            x[j] = np.expand_dims(img, 0)

        y = np.zeros(
            (self.batch_size, 1, self.img_size, self.img_size), dtype="float32"
        )
        for k, path_lab in enumerate(batch_label_img_paths):
            img = np.array(io.imread(path_lab)) / 255
            y[k] = np.expand_dims(img, 0)
        return torch.Tensor(x), torch.Tensor(y)
