from torch import load, device, cuda, softmax
from torch.utils.data import DataLoader
from unet.data import VertebraDataset
from tqdm import tqdm
import matplotlib.pyplot as plt


def reverseTransform(img):
    img = (img[0][0].to("cpu") > 0.5) * 255
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    device = device("cuda" if cuda.is_available() else "cpu")
    model = load(".\\save\\epoch50.pt")
    model.eval()
    dataset = VertebraDataset("..\\test\\image")
    loader = DataLoader(dataset, shuffle=False)
    for data in tqdm(loader, total=len(loader)):
        data = data.to(device)
        predict = model(data)
        reverseTransform(softmax(predict, dim=1))
