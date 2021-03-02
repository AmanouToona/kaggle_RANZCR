import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def tensor_to_image(x, show=False):
    x = transforms.ToPILImage()(x)

    if show:
        fig, ax = plt.subplots()
        ax.imshow(x.permute(1, 2, 0))  # batch が最後

    return transforms.ToPILImage()(x)


