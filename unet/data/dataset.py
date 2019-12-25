from torch import as_tensor, long
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from numpy import array
from PIL import Image, ImageOps
from os import listdir, path


class VertebraDataset(Dataset):
    def __init__(self, dataset_path, image_folder="image", mask_folder="label"):
        self.dataset_path = dataset_path
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.images = sorted(listdir(path.join(dataset_path, image_folder)))
        self.masks = sorted(listdir(path.join(dataset_path, mask_folder)))
        self.transform = Compose([ToTensor()])

    def __getitem__(self, idx):
        img_path = path.join(self.dataset_path, self.image_folder, self.images[idx])
        mask_path = path.join(self.dataset_path, self.mask_folder, self.masks[idx])
        img = ImageOps.equalize(Image.open(img_path))
        mask = Image.open(mask_path)
        return self.transform(img), as_tensor(array(mask)[:, 2:498], dtype=long)

    def __len__(self):
        return len(self.images)
