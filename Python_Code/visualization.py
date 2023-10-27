import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_img(p, s=(128, 128)):
    """
    Load an image from a path, convert it to grayscale, and resize it.
    """
    
    # Open and convert the image to grayscale
    img = Image.open(p).convert('L') 
    
    # Resize the image
    img = img.resize(s)  
    return np.array(img)

def class_dist(d):
    """
    Visualize the number of images for each emotion class.
    """
    counts = {}
    for folder in os.listdir(d):
        # Get the full path of the folder
        f_path = os.path.join(d, folder)  
        if os.path.isdir(f_path):
            # List all image files in the directory
            imgs = [i for i in os.listdir(f_path) if i.endswith(('.png', '.jpg', '.jpeg'))]
            # Count the images for each class
            counts[folder] = len(imgs)  

    # Plot the distribution
    plt.bar(counts.keys(), counts.values())  
    plt.xlabel('Class')
    plt.ylabel('# Images')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def sample_imgs(d, s=(128, 128)):
    """
    Display a few random images from each emotion class.
    """
    imgs = []
    labels = []
    for folder in os.listdir(d):
        f_path = os.path.join(d, folder)
        if os.path.isdir(f_path):
            # List all image files in the directory
            i_files = [i for i in os.listdir(f_path) if i.endswith(('.png', '.jpg', '.jpeg'))]
            # Randomly sample up to 5 images from each class
            for i in random.sample(i_files, min(5, len(i_files))):
                imgs.append(load_img(os.path.join(f_path, i), s))
                labels.append(folder)
                
    # Create a 5x5 grid for displaying images
    _, ax = plt.subplots(5, 5)  
    for idx, a in enumerate(ax.flatten()):
        if idx < len(imgs):
            # Display the image in grayscale
            a.imshow(imgs[idx], cmap='gray') 
            # Set the emotion class as the title
            a.set_title(labels[idx])  
            a.axis('off')
    plt.tight_layout()
    plt.show()

def pixel_dist(img_list):
    """
    Analyze and visualize the pixel intensity distribution.
    """
    # Extract all pixel values from the list of images
    vals = [v for img in img_list for v in img.ravel()]
    # Plot a histogram of pixel intensities
    plt.hist(vals, bins=256, range=(0, 256), color='gray')  
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
    class_dist(dir_path)
    imgs = sample_imgs(dir_path)
    pixel_dist(imgs)
