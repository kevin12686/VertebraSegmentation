from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import device, cuda, save
from tqdm import tqdm
from net.model import Unet
from net.data import VertebraDataset
from os import path


def train(dataset, model, criterion, epoch, device, save_dir=path.join("save")):
    loader = DataLoader(dataset=dataset, shuffle=True, num_workers=4, pin_memory=True)
    model = model.to(device)
    model.train()
    for ep in range(epoch):
        optimizer = Adam(model.parameters(), lr=1e-3 * (0.1 * (ep // 20)))
        loss_ = 0
        for batch, (img, mask) in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {ep + 1}/{epoch}"):
            img = img.to(device)
            mask = mask.to(device)
            output = model(img)
            loss = criterion(output, mask)
            loss_ += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss_ / len(loader)}")
        if (ep + 1) % 5 == 0:
            save(model, path.join(save_dir, f"epoch{ep + 1}(loss={int(loss_ / len(loader) * 1000)}).pt"))


if __name__ == '__main__':
    device = device("cuda" if cuda.is_available() else "cpu")
    dataset = VertebraDataset("..\\extend_dataset", train=True)
    model = Unet(in_channels=1, out_channels=2)
    criterion = CrossEntropyLoss()
    epoch = 100
    train(dataset, model, criterion, epoch, device)
