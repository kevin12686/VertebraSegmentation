from torch import load, device, cuda, no_grad, softmax
from torch.utils.data import DataLoader
from net.data import VertebraDataset
from tqdm import tqdm
import matplotlib.pyplot as plt


def reverseTransform(img, filename, save=False, path="..\\test\\predict"):
    img = (img[0][1].to("cpu") > 0.5) * 255
    plt.imshow(img)
    plt.show()
    if save:
        plt.imsave(f"{path}\\{filename[0]}", img, cmap="gray")


if __name__ == '__main__':
    device = device("cuda" if cuda.is_available() else "cpu")
    dataset = VertebraDataset("..\\test\\image")
    loader = DataLoader(dataset, shuffle=False)
    model = load(".\\save\\epoch100(loss=8).pt")
    model.eval()
    with no_grad():
        for data in tqdm(loader, total=len(loader)):
            filename = data[1]
            data = data[0].to(device)
            predict = model(data)
            reverseTransform(softmax(predict, dim=1), filename, save=True)
