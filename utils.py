from skimage.measure import label
from skimage.morphology import binary_dilation, binary_erosion, square
import numpy as np


# import matplotlib.pyplot as plt


def dice_coef(target, truth, t_val=255):
    target_val = target == t_val
    truth_val = truth == t_val
    target_obj = np.sum(target_val)
    truth_obj = np.sum(truth_val)
    intersection = np.sum(np.logical_and(target_val, truth_val))
    dice = (2 * intersection) / (target_obj + truth_obj)
    """
    # Debug Code
    plt.subplot(1, 2, 1)
    plt.title(f"Target dice:{dice}")
    plt.imshow(target == t_val)
    plt.subplot(1, 2, 2)
    plt.title(f"Truth ({t_val}TH)")
    plt.imshow(truth == t_val)
    plt.savefig(f"temp\\{t_val}.png")
    """
    return dice


def dice_coef_each_region(target, truth, num_objs=16, width=15):
    results = list()
    target = binary_dilation(binary_erosion(target, square(width)), square(width))
    target = label(target, connectivity=1)
    truth = binary_dilation(binary_erosion(truth, square(width)), square(width))
    truth = label(truth, connectivity=1)
    for i in range(1, num_objs + 1):
        results.append(dice_coef(target, truth, t_val=i))
    return results, np.mean(results)


if __name__ == '__main__':
    from skimage import io
    from skimage.color import rgb2gray

    img_label = rgb2gray(io.imread("test\\label\\0060.png"))
    img_target = rgb2gray(io.imread("test\\predict\\0060.png"))
    img_target = np.pad(img_target, [(0,), (2,)])
    print(dice_coef_each_region(img_target, img_label))
