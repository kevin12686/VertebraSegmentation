from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from os import path
from net.model import ResUnet
from net.data import VertebraDataset
import matplotlib.pyplot as plt
import time


class DiceCoefLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, truth):
        output = torch.sigmoid(target)
        return 1 - dice_coef(output, truth)


def dice_coef(target, truth, smooth=1.0):
    target = target.contiguous().view(-1)
    truth = truth.contiguous().view(-1)
    target_obj = (target * target).sum()
    truth_obj = (truth * truth).sum()
    intersection = torch.sum(target * truth)
    dice = (2 * intersection) / (target_obj + truth_obj + smooth)
    return dice


def save_fig(epoch, loss, trainscore, testscore, save_dir=path.join("save")):
    plt.plot(epoch, loss, label="Loss")
    plt.title("Loss")
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
    scores = list()
    model.eval()
    with torch.no_grad():
        for _, (img, mask) in tqdm(enumerate(loader), total=len(loader), desc="Evaluate"):
            img = img.to(device)
            mask = mask.to(device)
            output = model(img)
            output = torch.softmax(output, dim=1)
            score = dice_coef(output[:, 1], mask)
            scores.append(score)
    return torch.mean(torch.stack(scores, dim=0))


def run_one_epoch(model, loader, device, criterion, optimizer):
    total_loss = 0
    model.train()
    for _, (img, mask) in tqdm(enumerate(loader), total=len(loader), desc="Train"):
        img = img.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(loader)


def train(model, traindataset, testdataset, device, epochs, criterion, optimizer, batch_size=1, save_dir=path.join("save")):
    fig_epoch = list()
    fig_loss = list()
    fig_train_score = list()
    fig_test_score = list()

    highest_epoch = highest_score = 0

    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = model.to(device)

    for ep in range(epochs):
        timer = time.clock()
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']
        print(f"[ Epoch {ep + 1}/{epochs} ]")
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

        torch.save({"state_dict": model.state_dict(), "loss": loss_mean, "batchsize": batch_size, "Epoch": ep + 1}, path.join(save_dir, "last.pt"))

        print(f"""
Best Score {highest_score} @ Epoch {highest_epoch}
Learning Rate: {learning_rate}
Loss: {loss_mean}
Train Dice: {train_score}
Test Dice: {test_score}
Time passed: {round(time.clock() - timer)} seconds.
""")


if __name__ == '__main__':
    EPOCH = 120
    BATCHSIZE = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traindataset = VertebraDataset("..\\extend_dataset", train=True)
    testdataset = VertebraDataset("..\\original_data\\f03", train=True)
    model = ResUnet(in_channels=1, out_channels=2)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    train(model, traindataset, testdataset, device, EPOCH, criterion, optimizer, batch_size=BATCHSIZE)
    print("Done.")
