

import os
import cv2
import matplotlib.pyplot as plt
import random

def visualize_dataset(data_dir, num_samples_per_class=5):
    classes = ["Angry", "Bored", "Focused", "Neutral"]

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        files = os.listdir(class_dir)
        sample_files = random.sample(files, min(num_samples_per_class, len(files)))

        for i, file_name in enumerate(sample_files):
            img_path = os.path.join(class_dir, file_name)
            img = cv2.imread(img_path)

            plt.figure(figsize=(15, 3))
            plt.subplot(1, num_samples_per_class, i + 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f'{class_name}')
            plt.axis('off')

        plt.show()

def main():
    input_directory = "C:\\University\\pythonProject1\\AppliedAI_EmotionDetection\\Dataset\\"
    visualize_dataset(input_directory)

if __name__ == "__main__":
    main()
