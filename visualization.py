
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files

emotion_dataset_directory = "E:\\University\\pythonProject1\\AppliedAI_EmotionDetection\\Dataset\\"

emotion_dataset = load_files(emotion_dataset_directory, shuffle=False)

emotions = emotion_dataset.target_names

class_distribution = [len(class_files) for class_files in emotion_dataset.filenames]

plt.figure(figsize=(10, 5))
plt.bar(emotions, class_distribution, color='lightblue')
plt.xlabel('Emotion Class')
plt.ylabel('Number of Images')
plt.title('Distribution of Images Across Different Emotions')
plt.xticks(rotation=45)
plt.show()

sample_images = []
num_samples_per_emotion = 5

for emotion in emotions:
    emotion_dir = os.path.join(emotion_dataset_directory, emotion)
    image_files = random.sample(os.listdir(emotion_dir), num_samples_per_emotion)

    for image_file in image_files:
        image_path = os.path.join(emotion_dir, image_file)
        image = cv2.imread(image_path)
        sample_images.append((image, emotion))

fig, axs = plt.subplots(5, 5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        img, emotion = sample_images[i * 5 + j]
        axs[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i, j].set_title(emotion)
        axs[i, j].axis('off')
plt.show()

for image, emotion in sample_images:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.hist(gray_image.ravel(), bins=256, range=(0, 256), density=True, alpha=0.5, label=emotion)

plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Pixel Intensity Distribution in Sample Images')
plt.legend()
plt.show()

