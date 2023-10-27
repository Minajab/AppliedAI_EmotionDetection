# Facial Emotion Recognition

## Overview

The objective of this project is to develop a Deep Learning Convolutional Neural Network (CNN) using PyTorch that can analyze images of people and categorize them into distinct states. Our system is able to analyze images in order to recognize four classes:

1. Neutral: An image presenting neither active engagement nor disengagement, with relaxed facial features.
2. Focused: An image evidencing signs of active concentration, with sharp and attentive eyes.
3. Tired: An image displaying signs of weariness or a lack of interest. This can be evidenced by droopy eyes or vacant stares.
4. Angry: Signs of agitation or displeasure, which might manifest as tightened facial muscles, a tight-lipped frown, or narrowed eyes.

The whole project is split across three parts, which build on each other to deliver the final, complete project.


### Project Parts

#### Part 1: Data Collection, Cleaning, Labeling & Preliminary Analysis 

In this part, we developed suitable datasets that we will later need for training our model. This includes examining existing datasets, mapping them to our classes, performing some pre-processing, and basic analysis.

## Contents

- **Dataset Folder**: In the `Dataset` folder, you'll find essential resources related to our dataset:

1. **Dataset Download Links**: Due to the dataset's large size, we've made it available for download via Google Drive. Access the dataset [Here](https://drive.google.com/drive/folders/1-O9mxlY-pK7YS0uhr4juOBKvFHw5oN1C?usp=drive_link).

2. **Sample Images**: To offer a visual glimpse of the dataset, we've included ten sample images from each class. These samples are located in the `Sample_Images` subfolder

- **Code Folder**: Outlines the purpose of code files, scripts, or notebooks used in the project.

- **Results Folder**: Describes the content and purpose of files that contain the project's results, such as plots, tables, or outputs.

- **Documentation Folder**: If applicable, specify where documentation files, reports, or presentations are located.

## Data cleaning and processing

  ## List of dependencies
    1. os
    2. PIL 
    3. cv2
    4. numpy
    5. skimage 
    6. random

  ## Follow below steps for cleaning and processing your images:
    1. Download the original data from the dataset folder. It serves as input for the clearance code.
    2. Download the "haarcascade_frontalface_alt.xml" file, which can be found in the dependencies folder.
    3. Find "clearance.py" file in the code folder. You'll notice that the main function calls all other processing and cleaning functions in a specific order.

    4. There are two lines that require manual changes. First, in the main function, modify the "input_directory" to point to the data folder, which contains four subfolders with images representing four emotional classes. Second, in the "deleting_background" function, update the directory of the CascadeClassifier to the directory of the file "haarcascade_frontalface_alt.xml."

    5. Ultimately, the data file will contain the cleared data, as the processed and cleaned images have been overwritten in their original directories.  
- List dependencies or libraries required for data cleaning.

## Visualization


This part is integral for understanding our dataset's structure:

- **Class Distribution Visualization**: A graphical representation to visualize the number of images within each emotion category.
- **Sample Image Display** : View random images from each class, offering a quick snapshot of our dataset.
- **Pixel Intensity Distribution** : Understand pixel intensity spread, aiding in recognizing variations in image brightness and contrast.

For effective visualizations, refer to the following scripts:

`class_distribution.py`: Generates a bar chart for emotion class distribution.
`display_sample_images.py`: Displays sample images from each class.
`pixel_intensity_distribution.py`: Analyzes and displays pixel intensity distribution.


Facial Emotion Recognition: Data Visualization
Overview
In the realm of our Facial Emotion Recognition project, understanding and visualizing the dataset is of paramount importance. The visualization segment seeks to provide insights into the distribution, representation, and nature of our image dataset.

Visualization Objectives
Class Distribution Visualization: To understand the balance or imbalance between different emotional classes in the dataset.
Sample Image Display: Showcase random images from each class, providing a snapshot of dataset variety.
Pixel Intensity Distribution: Dive into the intricacies of pixel values to recognize variations in brightness and contrast across images, which can be pivotal in preprocessing steps.
Visualization Scripts
class_distribution.py: This script generates a bar chart showcasing the distribution of images across the various emotion classes.

display_sample_images.py: A handy script to pull and display random sample images from each emotion class.

pixel_intensity_distribution.py: Analyzes the pixel intensity spread across the dataset and visualizes it, assisting in decisions related to normalization or other preprocessing techniques.

To run these scripts, utilize the following commands:

bash
Copy code
python class_distribution.py
python display_sample_images.py
python pixel_intensity_distribution.py

**Visualization Results**

After executing the above scripts, you'll obtain:

**Bar Chart** : Representing the number of images in each class, aiding in identifying if data augmentation or oversampling is needed.

**Image Gallery**: A quick glance at the kind of images present in each class, helping in understanding dataset diversity.

**Pixel Intensity Graphs**: Depicting the spread of pixel values, which can guide further preprocessing steps.
