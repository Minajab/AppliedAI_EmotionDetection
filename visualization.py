import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_img(p, s=(128, 128)):
    img = Image.open(p).convert('L')
    img = img.resize(s)
    return np.array(img)


def class_dist(d):
    counts = {}
    for folder in os.listdir(d):
        f_path = os.path.join(d, folder)
        if os.path.isdir(f_path):
            imgs = [i for i in os.listdir(f_path) if i.endswith(('.png', '.jpg', '.jpeg'))]
            counts[folder] = len(imgs)

    plt.bar(counts.keys(), counts.values())
    plt.xlabel('Class')
    plt.ylabel('# Images')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def sample_imgs(d, s=(128, 128)):
    imgs = []
    labels = []
    for folder in os.listdir(d):
        f_path = os.path.join(d, folder)
        if os.path.isdir(f_path):
            i_files = [i for i in os.listdir(f_path) if i.endswith(('.png', '.jpg', '.jpeg'))]
            for i in random.sample(i_files, min(5, len(i_files))):
                imgs.append(load_img(os.path.join(f_path, i), s))
                labels.append(folder)

    _, ax = plt.subplots(5, 5)
    for idx, a in enumerate(ax.flatten()):
        if idx < len(imgs):
            a.imshow(imgs[idx], cmap='gray')
            a.set_title(labels[idx])
            a.axis('off')
    plt.tight_layout()
    plt.show()
    return imgs


def pixel_dist(img_list):
    vals = [v for img in img_list for v in img.ravel()]
    plt.hist(vals, bins=256, range=(0, 256), color='gray')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    plt.show()


if __name__ == "__main__":
    dir_path = "E:\\University\\pythonProject1\\AppliedAI_EmotionDetection\\Dataset\\"
    class_dist(dir_path)
    imgs = sample_imgs(dir_path)
    pixel_dist(imgs)
