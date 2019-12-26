from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import device, cuda, save
from tqdm import tqdm
from unet.model import Unet
from unet.data import VertebraDataset
from os import path


def train(dataset, model, optimizer, criterion, epoch, device, save_dir=path.join("save")):
    loader = DataLoader(dataset=dataset, shuffle=True, num_workers=4, pin_memory=True)
    model = model.to(device)
    model.train()
    postfix_data = {"Loss": 0}
    for ep in range(epoch):
        for batch, (img, mask) in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {ep + 1}/{epoch}", postfix=postfix_data):
            img = img.to(device)
            mask = mask.to(device)
            output = model(img)
            loss = criterion(output, mask)
            postfix_data["Loss"] = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        save(model, path.join(save_dir, f"epoch{ep + 1}.pt"))


if __name__ == '__main__':
    device = device("cuda" if cuda.is_available() else "cpu")
    dataset = VertebraDataset("..\\extend_dataset", train=True)
    model = Unet()
    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss()
    epoch = 100
    train(dataset, model, optimizer, criterion, epoch, device)
