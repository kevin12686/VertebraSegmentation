from torch import as_tensor, long
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from numpy import array
from PIL import Image, ImageOps, ImageFilter
from os import listdir, path


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
            img = Image.open(img_path)
            img = self.Preprocess(img)
            mask_path = path.join(self.dataset_path, self.mask_folder, self.masks[idx])
            mask = ImageOps.grayscale(Image.open(mask_path))
            return self.transform(img), as_tensor(array(mask)[:, 2:498] / 255, dtype=long)
        else:
            img_path = path.join(self.dataset_path, self.images[idx])
            img = Image.open(img_path)
            img = self.Preprocess(img)
            return self.transform(img), self.images[idx]

    def __len__(self):
        return len(self.images)

    @classmethod
    def Preprocess(cls, img):
        img = ImageOps.grayscale(img)
        img = img.filter(ImageFilter.GaussianBlur(2))
        img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        img = ImageOps.autocontrast(img, 2)
        img = ImageOps.equalize(img)
        return img
