from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from skimage.filters import gaussian, sobel
from skimage.color import rgb2gray
from skimage import exposure, io
from os import listdir, path
import torch
import numpy as np


class VertebraDataset(Dataset):
    def __init__(self, dataset_path, train=False, image_folder="image", mask_folder="label"):
        self.train = train
        self.dataset_path = dataset_path
        if self.train:
            self.image_folder = image_folder
            self.images = sorted(listdir(path.join(dataset_path, image_folder)))
            self.mask_folder = mask_folder
            self.masks = sorted(listdir(path.join(dataset_path, mask_folder)))
        else:
            self.images = sorted(listdir(path.join(dataset_path)))
        self.transform = Compose([ToTensor()])

    def __getitem__(self, idx):
        if self.train:
            img_path = path.join(self.dataset_path, self.image_folder, self.images[idx])
            img = io.imread(img_path)
            img = self.preprocess(img)
            out_img = np.zeros((1,) + img.shape, dtype=np.float)
            out_img[:, ] = img
            mask_path = path.join(self.dataset_path, self.mask_folder, self.masks[idx])
            mask = np.array(rgb2gray(io.imread(mask_path))) / 255
            return torch.as_tensor(out_img, dtype=torch.float), torch.as_tensor(mask, dtype=torch.long)
        else:
            img_path = path.join(self.dataset_path, self.images[idx])
            img = io.imread(img_path)
            img = self.preprocess(img)
            out_img = np.zeros((1,) + img.shape, dtype=np.float)
            out_img[:, ] = img
            return torch.as_tensor(out_img, dtype=torch.float), self.images[idx]

    def __len__(self):
        return len(self.images)

    @classmethod
    def preprocess(cls, img):
        img = rgb2gray(img)
        bound = img.shape[0] // 3
        up = exposure.equalize_adapthist(img[:bound, :])
        down = exposure.equalize_adapthist(img[bound:, :])
        enhance = np.append(up, down, axis=0)
        edge = sobel(gaussian(enhance, 2))
        enhance = enhance + edge * 3
        return np.where(enhance > 1, 1, enhance)


if __name__ == '__main__':
    dataset = VertebraDataset("..\\..\\extend_dataset", train=True)
    a, b = dataset[0]
    print(a.shape)
    print(b.shape)
