from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch import device, cuda, save
from tqdm import tqdm
from unet.model import Unet
from unet.data import VertebraDataset
from os import path


def train(dataset, model, optimizer, criterion, epoch, device, save_dir=path.join("save")):
    loader = DataLoader(dataset=dataset, shuffle=True, num_workers=4, pin_memory=True)
    model = model.to(device)
    for ep in range(epoch):
        for batch, (img, mask) in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {ep + 1}"):
            img = img.to(device)
            mask = mask.to(device)
            output = model(img)
            loss = criterion(output, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        save(model, path.join(save_dir, "epoch1.pt"))


if __name__ == '__main__':
    # device = device("cuda:0" if cuda.is_available() else "cpu")
    device = device("cpu")
    dataset = VertebraDataset(path.join("..\\extend_dataset"))
    model = Unet()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.99)
    criterion = CrossEntropyLoss()
    epoch = 5
    train(dataset, model, optimizer, criterion, epoch, device)
