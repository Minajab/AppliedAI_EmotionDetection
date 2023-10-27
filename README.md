# Facial Emotion Recognition

## Overview

The objective of this project is to develop a Deep Learning Convolutional Neural Network (CNN) using PyTorch that can analyze images of people and categorize them into distinct states. Our system is able to analyze images in order to recognize four classes:

1. Neutral: An image presenting neither active engagement nor disengagement, with relaxed facial features.
2. Focused: An image evidencing signs of active concentration, with sharp and attentive eyes.
3. Tired: An image displaying signs of weariness or a lack of interest. This can be evidenced by droopy eyes or vacant stares.
4. Angry: Signs of agitation or displeasure, which might manifest as tightened facial muscles, a tight-lipped frown, or narrowed eyes.

The whole project is split across three parts, which build on each other to deliver the final, complete project.

To be able to run the project, first use `python -m venv venv` to create a virtual environment called venv. Then use `source venv/bin/activate` to activate the virtual environment. Finally, use `pip install -r requirements.txt` to install all needed python packages.

### Project Parts

#### Part 1: Data Collection, Cleaning, Labeling & Preliminary Analysis 

In this part, we developed suitable datasets that we will later need for training our model. This includes examining existing datasets, mapping them to our classes, performing some pre-processing, and basic analysis.

## Contents

### Dataset
It contains essential resources related to our dataset:

1. **Dataset Download Links**: Due to the dataset's large size, we've made it available for download via Google Drive. The links are provided in `Dataset.md`.
2. **dataset.pdf File**: We provided some information about the raw datasets we used (including the source of each dataset). The file also contains 10 samples of each class. The is located in the Dataset folder.


The raw dataset should be downloaded from [https://drive.google.com/drive/folders/1-O9mxlY-pK7YS0uhr4juOBKvFHw5oN1C?usp=sharing](https://drive.google.com/drive/folders/1-O9mxlY-pK7YS0uhr4juOBKvFHw5oN1C?usp=sharing) and placed inside the Raw_Data directory.

The preprocessed dataset can be downloaded from [https://drive.google.com/file/d/14MvCEBK-_dkIoo2P6YkbLfdVIoSBEfht/view?usp=drive_link](https://drive.google.com/file/d/14MvCEBK-_dkIoo2P6YkbLfdVIoSBEfht/view?usp=drive_link) and placed inside the Dataset directory.

### Python_Code
This folder contains two important files:

1. **Clearance.py**: This file is responsible for cleaning and preprocessing the dataset.

2. **visualization.py**: This file is used for data visualization tasks.

### Documentation
This folder includes the following files:

- **Report.pdf**: This is the main report document containing the project's findings, methodology, and results.

- **Expectations-Originality-Form.pdf**


## Data cleaning and processing

  ## List of dependencies
  
    1. matplotlib
    2. Pillow
    3. opencv-python
    4. numpy
    5. scikit-image 
    

  ## Follow below steps for cleaning and processing your images:
    1. Download the original data from the dataset folder. It serves as input for the clearance code.
    2. Download the "haarcascade_frontalface_alt.xml" file, which can be found in the dependencies folder.
    3. Find "clearance.py" file in the preprocessing_code folder. You'll notice that the main function calls all other processing and cleaning functions in a   
    specific order.
    4. There are two lines that require manual changes. First, in the main function, modify the "input_directory" to point to the data folder, which contains four 
    subfolders with images representing four emotional classes. Second, in the "deleting_background" function, update the directory of the CascadeClassifier to 
    the directory of the file "haarcascade_frontalface_alt.xml."
    5. Ultimately, the data file will contain the cleared data, as the processed and cleaned images have been overwritten in their original directories.  

## Visualization

In the realm of our Facial Emotion Recognition project, understanding and visualizing the dataset is of paramount importance. The visualization segment seeks to provide insights into the distribution, representation, and nature of our image dataset.
This part is integral for understanding our dataset's structure:

- **Class Distribution Visualization**: A graphical representation to visualize the number of images within each emotion category.
- **Sample Image Display** : View random images from each class, offering a quick snapshot of our dataset.
- **Pixel Intensity Distribution** : Understand pixel intensity spread, aiding in recognizing variations in image brightness and contrast.

For effective visualizations, refer to the following scripts:

`class_distribution.py`: Generates a bar chart for emotion class distribution.
`display_sample_images.py`: Displays sample images from each class.
`pixel_intensity_distribution.py`: Analyzes and displays pixel intensity distribution.

**Visualization Results**

After executing the above scripts, you'll obtain:

**Bar Chart** : Representing the number of images in each class, aiding in identifying if data augmentation or oversampling is needed.

**Image Gallery**: A quick glance at the kind of images present in each class, helping in understanding dataset diversity.

**Pixel Intensity Graphs**: Depicting the spread of pixel values, which can guide further preprocessing steps.
