from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage import io
from net.data import VertebraDataset
from net.model import ResUnet
import torch
import numpy as np


def predict(model, loader, save_path="..\\test\\predict"):
    model.eval()
    with torch.no_grad():
        for _, (img, filename) in tqdm(enumerate(loader), total=len(loader), desc="Predict"):
            img = img.to(device)
            output = model(img)
            output = (torch.sigmoid(output) > 0.5) * 255
            output = output.cpu().numpy().astype(np.uint8)
            for dim in range(output.shape[0]):
                io.imsave(f"{save_path}\\predict_{filename[dim]}", output[dim][0])


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VertebraDataset("..\\test\\image")
    model = ResUnet(in_channels=1, out_channels=1)
    checkpoint = torch.load("save\\best.pt")
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(dataset, batch_size=checkpoint["batchsize"])
    model = model.to(device)
    predict(model, loader)
    print("Done.")
