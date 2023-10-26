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

## Data Cleaning

1. `Clearance.py`: This file contains some preprocessing steps needed for...
2. `Data cleaning phase.py`: This file ...
- List dependencies or libraries required for data cleaning.
- Provide clear steps to execute the data cleaning process.

### Example:

```bash
python data_cleaning_script.py

## Visualization


This part is integral for understanding our dataset's structure:

- **Class Distribution Visualization**: A graphical representation to visualize the number of images within each emotion category.
- **Sample Image Display** : View random images from each class, offering a quick snapshot of our dataset.
- **Pixel Intensity Distribution** : Understand pixel intensity spread, aiding in recognizing variations in image brightness and contrast.

For effective visualizations, refer to the following scripts:

`class_distribution.py`: Generates a bar chart for emotion class distribution.
`display_sample_images.py`: Displays sample images from each class.
`pixel_intensity_distribution.py`: Analyzes and displays pixel intensity distribution.

Use the below commands for execution:
