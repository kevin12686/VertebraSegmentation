from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import torch
from tqdm import tqdm
from os import path
from net.model import ResUnet
from net.data import VertebraDataset
from utils import dice_coef
import numpy as np
import matplotlib.pyplot as plt
import time


def save_fig(epoch, loss, trainscore, testscore, save_dir=path.join("save")):
    plt.plot(epoch, loss, label="Loss")
    plt.title("Loss Status")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path.join(save_dir, "loss.png"))
    plt.clf()

    plt.plot(epoch, trainscore, label="Train")
    plt.plot(epoch, testscore, label="Test")
    plt.title("Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(path.join(save_dir, "score.png"))
    plt.clf()


def eval(model, loader, device):
    model.eval()
    scores = list()
    with torch.no_grad():
        for _, (img, mask) in tqdm(enumerate(loader), total=len(loader), desc="Evaluate"):
            img = img.to(device)
            output = model(img)
            output = (torch.sigmoid(output) > 0.5) * 255
            output = output.cpu().numpy().astype(np.uint8)
            mask = mask.numpy().astype(np.uint8) * 255
            for dim in range(output.shape[0]):
                score = dice_coef(output[dim][0], mask[dim][0])
                scores.append(score)
    return np.mean(scores)


def run_one_epoch(model, loader, device, criterion, optimizer):
    total_loss = 0
    model.train()
    for _, (img, mask) in tqdm(enumerate(loader), total=len(loader), desc="Train"):
        img = img.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, mask)
        total_loss += loss
        loss.backward()
        optimizer.step()
    return total_loss / len(loader)


def train(model, dataset, device, epochs, criterion, batch_size=1, test_factor=0.1, save_dir=path.join("save")):
    fig_epoch = list()
    fig_loss = list()
    fig_train_score = list()
    fig_test_score = list()

    highest_epoch = 0
    highest_score = 0
    loss_mean = 0

    total_data = len(dataset)
    test_size = int(total_data * test_factor)
    train_size = total_data - test_size
    trainset, testset = random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=4)

    model = model.to(device)

    for ep in range(epochs):
        print(f"[ Epoch {ep + 1}/{epochs} ]")
        timer = time.clock()
        optimizer = Adam(model.parameters(), lr=1e-4 * (0.1 ** (ep // 5)))
        loss_mean = run_one_epoch(model, trainloader, device, criterion, optimizer)
        train_score = eval(model, trainloader, device)
        test_score = eval(model, testloader, device)

        fig_epoch.append(ep + 1)
        fig_loss.append(loss_mean)
        fig_train_score.append(train_score)
        fig_test_score.append(test_score)
        save_fig(fig_epoch, fig_loss, fig_train_score, fig_test_score, save_dir=save_dir)

        if test_score > highest_score:
            highest_score = test_score
            highest_epoch = ep + 1
            torch.save({"state_dict": model.state_dict(), "loss": loss_mean, "batchsize": batch_size, "Epoch": ep + 1}, path.join(save_dir, "best.pt"))

        print(f"""
Best Score {highest_score} @ Epoch {highest_epoch}
Loss: {loss_mean}
Train Dice: {train_score}
Test Dice: {test_score}
Time passed: {time.clock() - timer} seconds.
""")

    torch.save({"state_dict": model.state_dict(), "loss": loss_mean, "batchsize": batch_size, "Epoch": ep + 1}, path.join(save_dir, "last.pt"))


if __name__ == '__main__':
    EPOCH = 5
    BATCHSIZE = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VertebraDataset("..\\extend_dataset", train=True)
    model = ResUnet(in_channels=1, out_channels=1)
    criterion = BCEWithLogitsLoss()

    train(model, dataset, device, EPOCH, criterion, batch_size=BATCHSIZE)
    print("Done.")
