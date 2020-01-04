from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage import io
from net.data import VertebraDataset
from net.model import Unet
import torch
import numpy as np


def predict(model, loader, save_path="..\\test\\predict"):
    model.eval()
    with torch.no_grad():
        for _, (img, filename) in tqdm(enumerate(loader), total=len(loader), desc="Predict"):
            img = img.to(device)
            output = model(img)
            output = (torch.softmax(output, dim=1)[:, 1]) * 255
            output = output.cpu().numpy().astype(np.uint8)
            for dim in range(output.shape[0]):
                io.imsave(f"{save_path}\\p{filename[dim]}", output[dim])


def predict_one(img, model_path):
    img = VertebraDataset.preprocess(img)
    format_img = np.zeros([1, 1, img.shape[0], img.shape[1]])
    format_img[0, 0] = img
    format_img = torch.tensor(format_img, dtype=torch.float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(in_channels=1, out_channels=2)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        format_img = format_img.to(device)
        output = model(format_img)
        output = (torch.softmax(output, dim=1)[:, 1]) * 255
        output = output.cpu().numpy().astype(np.uint8)
    return output[0]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VertebraDataset("..\\original_data\\f03")
    model = Unet(in_channels=1, out_channels=2)
    checkpoint = torch.load("save\\best.pt")
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(dataset, batch_size=1)
    model = model.to(device)
    predict(model, loader)
    print("Done.")
