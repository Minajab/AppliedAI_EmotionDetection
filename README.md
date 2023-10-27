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


The raw dataset should be downloaded from [https://drive.google.com/drive/folders/1-O9mxlY-pK7YS0uhr4juOBKvFHw5oN1C?usp=sharing](https://drive.google.com/drive/folders/1-O9mxlY-pK7YS0uhr4juOBKvFHw5oN1C?usp=sharing) and placed inside the Dataset directory. Alternatively, the cleaned and preprocessed dataset can be downloaded from [https://drive.google.com/file/d/1vs_3zNQ_ZaeZeA0udOAZJsVgBjK4YfPY/view?usp=sharing](https://drive.google.com/file/d/1vs_3zNQ_ZaeZeA0udOAZJsVgBjK4YfPY/view?usp=sharing) and placed inside the Dataset directory.

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
    1. Download the zip file of the project and unzip it. Afterwards, open your cmd and moveo the directory of the project.
    2. Use `python -m venv venv` to create a virtual environment called venv.
    3. If you are using macOS or Linux, use the command source venv/bin/activate to activate the virtual environment. However, if you are on Windows,         navigate to the 'Scripts' directory within the virtual environment and execute the command 'activate'.  
    4. Use `pip install -r requirements.txt` to install all needed Python packages.
    5. Download the original data from [https://drive.google.com/drive/folders/1-O9mxlY-pK7YS0uhr4juOBKvFHw5oN1C?usp=sharing]                  
    (https://drive.google.com/drive/folders/1-O9mxlY-pK7YS0uhr4juOBKvFHw5oN1C?usp=sharing) and please place four folders containing emotional classes         into the 'Dataset' directory. This 'Dataset' directory will serve as the input for the clearance code.
    6. Change directory to Python_Code. Run `python Clearance.py`. You'll notice that the main function calls all other processing and cleaning functions 
    in a specific order.
    7. Ultimately, the Dataset diretory will contain the cleaned and preprocessed data, as the processed and cleaned images have been overwritten in 
    their original directories. 

## Visualization

In the realm of our Facial Emotion Recognition project, understanding and visualizing the dataset is of paramount importance. The visualization segment seeks to provide insights into the distribution, representation, and nature of our image dataset.
This part is integral for understanding our dataset's structure:

- **Class Distribution Visualization**: A graphical representation to visualize the number of images within each emotion category.
- **Sample Image Display** : View random images from each class, offering a quick snapshot of our dataset.
- **Pixel Intensity Distribution** : Understand pixel intensity spread, aiding in recognizing variations in image brightness and contrast.

**Visualization Results**

After executing the scripts, you'll obtain:

**Bar Chart** : Representing the number of images in each class, aiding in identifying if data augmentation or oversampling is needed.

**Image Gallery**: A quick glance at the kind of images present in each class, helping in understanding dataset diversity.

**Pixel Intensity Graphs**: Depicting the spread of pixel values, which can guide further preprocessing steps.
