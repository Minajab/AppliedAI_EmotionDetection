# Facial Emotion Recognition

## Overview

The objective of this project is to develop a Deep Learning Convolutional Neural Network (CNN) using PyTorch that can analyze images of people and categorize them into distinct states. Our system is able to analyze images in order to recognize four classes:

1. Neutral: An image presenting neither active engagement nor disengagement, with relaxed facial features.
2. Focused: An image evidencing signs of active concentration, with sharp and attentive eyes.
3. Tired: An image displaying signs of weariness or a lack of interest. This can be evidenced by droopy eyes or vacant stares.
4. Angry: Signs of agitation or displeasure, which might manifest as tightened facial muscles, a tight-lipped frown, or narrowed eyes.

The whole project is split across three parts, which build on each other to deliver the final, complete project.

To be able to run the project, first use `python -m venv venv` to create a virtual environment called venv. Then use `source venv/bin/activate` to activate the virtual environment. Finally, use `pip install -r requirements.txt` to install all needed python packages.

__Note:__ The commands provided work with Linux and Mac operating systems. For Windows, please refer to the Python documentation page, especially [https://docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html).

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


### Data cleaning and processing

## List of dependencies
The following dependencies have been installed for Python 3.9.6 and above.

1. matplotlib==3.8.0
2. Pillow==10.1.0
3. opencv-python==4.8.1.78
4. numpy==1.26.1
5. scikit-image==0.22.0
6. tifffile==2023.9.26
7. torch==2.1.0
8. torchaudio==2.1.0
9. torchvision==0.16.0
10. tqdm==4.66.1
11. scikit-learn==1.3.2
12. seaborn==0.13.0
    

## Follow below steps for cleaning and processing your images:
1. Download the project's ZIP file and unzip it. Then, open your command prompt (CMD) and navigate to the project directory.
2. Use `python -m venv venv` to create a virtual environment called venv. For more information about Python virtual environments, please refer to [https://docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html).
3. If you are using macOS or Linux, use the command `source venv/bin/activate` to activate the virtual environment. However, if you are on Windows, navigate to the 'Scripts' directory within the virtual environment and execute the command 'activate'.  
4. Use `pip install -r requirements.txt` to install all needed Python packages (if you have GPU with cuda installed, you can run `pip install -r requirements_gpu.txt`).
5. Download the original data from [https://drive.google.com/drive/folders/1-O9mxlY-pK7YS0uhr4juOBKvFHw5oN1C?usp=sharing](https://drive.google.com/drive/folders/1-O9mxlY-pK7YS0uhr4juOBKvFHw5oN1C?usp=sharing) and please place four folders containing emotional classes into the 'Dataset' directory. This 'Dataset' directory will serve as the input for the clearance code.
6. Change directory to Python_Code. Run `python Clearance.py`. You'll notice that the main function calls all other processing and cleaning functions in a specific order.
7. Ultimately, the Dataset diretory will contain the cleaned and preprocessed data, as the processed and cleaned images have been overwritten in their original directories. 

__Note:__ Please inspect the data directory and subdirectories and remove any `.DS_Store` files that you find. These files are automatically created by the operating system, especially the Mac operating system. They are hidden, so you would need to remove them using a terminal. These files will cause an error while running the preprocessing files. For more information about `.DS_Store` please refer to [https://buildthis.com/ds_store-files-and-why-you-should-know-about-them/](https://buildthis.com/ds_store-files-and-why-you-should-know-about-them/).

### Visualization

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

### Training

The model is available in the ___model.py___ file. The training file is the ___train.py___ file. You can train a CNN model with arbitrary number of convolution layers and any desired hidden neurons using the ___train.py___ file without the need to change the code. The training script accepts the following arguments:

1. -h, --help            show the help message and exit
2. --epochs EPOCHS       Number of Epochs (default: 100)
3. --batch_size BATCH_SIZE
                        Batch Size (default: 10)
4. --conv_kernel CONV_KERNEL
                        Kernel Size for the Conv Module (default: 3)
5. --pooling_kernel POOLING_KERNEL
                        Kernel Size for the Pooling Module (default: 2)
6. --layers LAYERS       Layers in Comma Separated Format (default: 64,128)

For example, running the following command `python train.py --epochs=100 --batch_size=10 --conv_kernel=3 --pooling_kernel=2 --layers="64,128,256"` will train a model for 100 epochs, with batch size of 10, and three convolutional layers with 64, 128, and 256 hidden neurons respectively and with a 3x3 kernel and 2x2 pooling kernel size.

The training code splits the data into 70% training, 15% validation, and 15% testing data. To split the data, stratified splitting has been used to ensure having the same distribution of labels in the training, validation, and testing sets. Because the random_state variable and torch.manual_seed has been set, the code will produce the same results and splits when you run them multiple times.

After running the code, the training and validation loss of the model will be stored in a folder called ___losses___ in the same directory as the README. The best performing model (i.e., the model with the lowest validation loss) after the _10_ th epoch and the model trained after the _n_ th epoch are stored in a folder called ___saved_models___ in the same directory as the README file.

### Evaluation

The evaluation file is the ___eval.py___ file. You can evaluate any of you trained CNN models using the ___eval.py___ file without the need to change the code. The evaluation script accepts the following arguments:

1. -h, --help            show the help message and exit
2. --batch_size BATCH_SIZE
                        Batch Size (default: 10)
3. --conv_kernel CONV_KERNEL
                        Kernel Size for the Conv Module (default: 3)
4. --pooling_kernel POOLING_KERNEL
                        Kernel Size for the Pooling Module (default: 2)
5. --layers LAYERS       Layers in Comma Separated Format (default: 64,128)

For example, running the following command `python eval.py --batch_size=10 --conv_kernel=3 --pooling_kernel=2 --layers="64,128,256"` will evaluate the previously trained model with three convolutional layers with 64, 128, and 256 hidden neurons respectively and with a 3x3 kernel and 2x2 pooling kernel size. The code uses the best and final models stored in the ___saved_models___ directory. The batch size is used to feed the test data in mini-batches to the model.

The evaluation code first plots the training and evaluation losses of the model to better understand whether the model has converged. Then it will calculate the following for both the best and final models:
- Confusion Matrix
- Micro Precision
- Micro Recall
- Micro F1 Score
- Micro Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score

Finally, the code will store the confusion matrix, metrics, and the train and validation losses plot in a directory called ___results___ living in the same directory as the README file.

### Application Mode

The application mode file is the ___application_mode.py___ file. You can run the model in the application mode for any of you trained CNN models using the ___application_mode.py___ file without the need to change the code and with any of the datapoints you desire. The application mode script accepts the following arguments:

1. -h, --help            show the help message and exit
2. --conv_kernel CONV_KERNEL
                        Kernel Size for the Conv Module (default: 3)
3. --pooling_kernel POOLING_KERNEL
                        Kernel Size for the Pooling Module (default: 2)
4. --layers LAYERS       Layers in Comma Separated Format (default: 64,128)
5. --data DATA           The Data You Want to Run the Model On. The value can be "all", "rand", any number between 0 and 1 representing a split of the dataset, or a single number       
                        representing a specific datapoint index from the dataset (default: "rand")


For example, running the following command `python application_mode.py --conv_kernel=3 --pooling_kernel=2 --layers="64,128,256" --data=0.1` will run the previously trained model with three convolutional layers with 64, 128, and 256 hidden neurons respectively and with a 3x3 kernel and 2x2 pooling kernel size and on 10% of the data in the application mode. Or for example, running the following command `python application_mode.py --conv_kernel=3 --pooling_kernel=2 --layers="64,128,256" --data=46` will run the previously trained model with three convolutional layers with 64, 128, and 256 hidden neurons respectively and with a 3x3 kernel and 2x2 pooling kernel size and on the datapoint at index 46 in the application mode.

The code uses the best and final models stored in the ___saved_models___ directory.

The code will return the predicted and true label of the datapoint if the code is used to run the model on one datapoint. Otherwise, the code will calculate the following for both the best and final models:
- Confusion Matrix
- Micro Precision
- Micro Recall
- Micro F1 Score
- Micro Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score

### K-Fold Cross Validation Training and Evaluation

The model is available in the ___model.py___ file. The training file is the ___k_fold_train.py___ file. You can train a CNN model with arbitrary number of convolution layers, any desired hidden neurons and for any number of folds using the ___k_fold_train.py___ file without the need to change the code. The training script accepts the following arguments:

1. -h, --help            show the help message and exit
2. --epochs EPOCHS       Number of Epochs (default: 100)
3. --batch_size BATCH_SIZE
                        Batch Size (default: 10)
4. --conv_kernel CONV_KERNEL
                        Kernel Size for the Conv Module (default: 3)
5. --pooling_kernel POOLING_KERNEL
                        Kernel Size for the Pooling Module (default: 2)
6. --layers LAYERS       Layers in Comma Separated Format (default: 64,128)
7. --num_folds NUM_FOLDS
                        Number of Folds (default: 10)


For example, running the following command `python k_fold_train.py --epochs=100 --batch_size=10 --conv_kernel=3 --pooling_kernel=2 --layers="64,128,256" --num_folds=10` will train a model for 100 epochs, with batch size of 10, and three convolutional layers with 64, 128, and 256 hidden neurons respectively and with a 3x3 kernel and 2x2 pooling kernel size and using 10-fold cross validation strategy.

During each fold, 20% of the training data is used as the validation set to find the best performing model of the fold. Both splitting the data into k folds and train and validation split use the stratified splitting technique. Because the random_state variable and torch.manual_seed has been set, the code will produce the same results and splits when you run them multiple times.

After running the code, the training and validation loss of the model will be stored in a folder called ___losses_k_fold___ (k is replaced by the actual value of k) in the same directory as the README. The best performing model (i.e., the model with the lowest validation loss) after the _10_ th epoch and the model trained after the _n_ th epoch are stored in a folder called ___saved_models_k_fold___ (k is replaced by the actual value of k) in the same directory as the README file.

The evaluation file is the ___k_fold_eval.py___ file. You can evaluate any of you trained CNN models using the ___k_fold_eval.py___ file without the need to change the code. The evaluation script accepts the following arguments:

1. -h, --help            show the help message and exit
2. --batch_size BATCH_SIZE
                        Batch Size (default: 10)
3. --conv_kernel CONV_KERNEL
                        Kernel Size for the Conv Module (default: 3)
4. --pooling_kernel POOLING_KERNEL
                        Kernel Size for the Pooling Module (default: 2)
5. --layers LAYERS       Layers in Comma Separated Format (default: 64,128)
6. --num_folds NUM_FOLDS
                        Number of Folds (default: 10)

For example, running the following command `python k_fold_eval.py --batch_size=10 --conv_kernel=3 --pooling_kernel=2 --layers="64,128,256" --num_folds=10` will evaluate the previously trained model using 10-fold cross validation technique with three convolutional layers each with 64, 128, and 256 hidden neurons respectively and with a 3x3 kernel and 2x2 pooling kernel size. The code uses the best and final models stored in the ___saved_models_k_fold___ (k will be replaced by its actual number) directory. The batch size is used to feed the test data in mini-batches to the model.

The evaluation code first plots the training and evaluation losses of the model for each fold to better understand whether the model has converged. Then it will calculate the following for both the best and final models of each fold:
- Confusion Matrix
- Micro Precision
- Micro Recall
- Micro F1 Score
- Micro Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score

Finally, the code will store the confusion matrix, metrics, and the train and validation losses plot in a directory called ___results_k_fold___ (k will be replaced by its actual value) living in the same directory as the README file.

### Evaluating Bias in the Model

The metadata (Gender and Age Group information) for the original dataset should be placed in a directory called __Meta_Data__. The file should be called __original_dataset.csv__ (It can be downloaded from[https://drive.google.com/file/d/1tCh7-plO9SuhTNzDKckQNSkFxaQWw9wS/view?usp=sharing](https://drive.google.com/file/d/1tCh7-plO9SuhTNzDKckQNSkFxaQWw9wS/view?usp=sharing)).

The evaluation file is the ___bias_eval.py___ file. You can evaluate any of you trained CNN models using the ___bias_eval.py___ file without the need to change the code. The evaluation script accepts the following arguments:

1. -h, --help            show the help message and exit
2. --batch_size BATCH_SIZE
                        Batch Size (default: 10)
3. --conv_kernel CONV_KERNEL
                        Kernel Size for the Conv Module (default: 3)
4. --pooling_kernel POOLING_KERNEL
                        Kernel Size for the Pooling Module (default: 2)
5. --layers LAYERS       Layers in Comma Separated Format (default: 64,128)

For example, running the following command `python bias_eval.py --batch_size=10 --conv_kernel=3 --pooling_kernel=2 --layers="64,128,256"` will evaluate the previously trained model with three convolutional layers with 64, 128, and 256 hidden neurons respectively and with a 3x3 kernel and 2x2 pooling kernel size for each metadata categories. The code uses the best and final models stored in the ___saved_models___ directory. The batch size is used to feed the test data in mini-batches to the model.

The evaluation code will calculate the following for both the best and final models for each category:
- Confusion Matrix
- Micro Precision
- Micro Recall
- Micro F1 Score
- Micro Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score

Finally, the code will store the confusion matrix and the metrics in a directory called ___bias_results___ living in the same directory as the README file. Also, it will print out two tables (one for the final model and another for the best model) with the bias evaluation results for each category.

### Bias Mitigation

The model is available in the ___model.py___ file. The dataset for this phase is stored in a directory called __Unbiased_Dataset__ (It can be downloaded from[https://drive.google.com/file/d/1bFQr2gjUMYlRWo2zn5hssffzjNjX3KYu/view?usp=sharing](https://drive.google.com/file/d/1bFQr2gjUMYlRWo2zn5hssffzjNjX3KYu/view?usp=sharing)). The training process is similar to the previous phase. The training file is the ___unbiased_train.py___ file. You can train a CNN model with arbitrary number of convolution layers and any desired hidden neurons using the ___unbiased_train.py___ file without the need to change the code. The training script accepts the following arguments:

1. -h, --help            show the help message and exit
2. --epochs EPOCHS       Number of Epochs (default: 100)
3. --batch_size BATCH_SIZE
                        Batch Size (default: 10)
4. --conv_kernel CONV_KERNEL
                        Kernel Size for the Conv Module (default: 3)
5. --pooling_kernel POOLING_KERNEL
                        Kernel Size for the Pooling Module (default: 2)
6. --layers LAYERS       Layers in Comma Separated Format (default: 64,128)

For example, running the following command `python unbiased_train.py --epochs=100 --batch_size=10 --conv_kernel=3 --pooling_kernel=2 --layers="64,128,256"` will train a model for 100 epochs, with batch size of 10, and three convolutional layers with 64, 128, and 256 hidden neurons respectively and with a 3x3 kernel and 2x2 pooling kernel size.

The training code splits the data into 70% training, 15% validation, and 15% testing data. To split the data, stratified splitting has been used to ensure having the same distribution of groups and labels in the training, validation, and testing sets. Because the random_state variable and torch.manual_seed has been set, the code will produce the same results and splits when you run them multiple times.

After running the code, the training and validation loss of the model will be stored in a folder called ___unbiased_losses___ in the same directory as the README. The best performing model (i.e., the model with the lowest validation loss) after the _10_ th epoch and the model trained after the _n_ th epoch are stored in a folder called ___unbiased_saved_models___ in the same directory as the README file.

The metadata (Gender and Age Group information) for the unbiased dataset should be placed in a directory called __Meta_Data__. The file should be called __unbiased_dataset.csv__ (It can be downloaded from [https://drive.google.com/file/d/1O_uYl1_0vlVM57PeIb85q-1g4SV8fM1A/view?usp=sharing](https://drive.google.com/file/d/1O_uYl1_0vlVM57PeIb85q-1g4SV8fM1A/view?usp=sharing)).

The evaluation file is the ___unbiased_eval.py___ file. You can evaluate any of you trained CNN models using the ___unbiased_eval.py___ file without the need to change the code. The evaluation script accepts the following arguments:

1. -h, --help            show the help message and exit
2. --batch_size BATCH_SIZE
                        Batch Size (default: 10)
3. --conv_kernel CONV_KERNEL
                        Kernel Size for the Conv Module (default: 3)
4. --pooling_kernel POOLING_KERNEL
                        Kernel Size for the Pooling Module (default: 2)
5. --layers LAYERS       Layers in Comma Separated Format (default: 64,128)

For example, running the following command `python unbiased_eval.py --batch_size=10 --conv_kernel=3 --pooling_kernel=2 --layers="64,128,256"` will evaluate the previously trained model with three convolutional layers with 64, 128, and 256 hidden neurons respectively and with a 3x3 kernel and 2x2 pooling kernel size for each metadata categories. The code uses the best and final models stored in the ___unbiased_saved_models___ directory. The batch size is used to feed the test data in mini-batches to the model.

The evaluation code will calculate the following for both the best and final models for each category:
- Confusion Matrix
- Micro Precision
- Micro Recall
- Micro F1 Score
- Micro Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score

Finally, the code will store the confusion matrix and the metrics in a directory called ___unbiased_results___ living in the same directory as the README file. Also, it will print out two tables (one for the final model and another for the best model) with the bias evaluation results for each category.

### K-Fold Cross Validation Training and Evaluation for Unbiased Data

The model is available in the ___model.py___ file. The training file is the ___unbiased_k_fold_train.py___ file. You can train a CNN model with arbitrary number of convolution layers, any desired hidden neurons and for any number of folds using the ___unbiased_k_fold_train.py___ file without the need to change the code. The training script accepts the following arguments:

1. -h, --help            show the help message and exit
2. --epochs EPOCHS       Number of Epochs (default: 100)
3. --batch_size BATCH_SIZE
                        Batch Size (default: 10)
4. --conv_kernel CONV_KERNEL
                        Kernel Size for the Conv Module (default: 3)
5. --pooling_kernel POOLING_KERNEL
                        Kernel Size for the Pooling Module (default: 2)
6. --layers LAYERS       Layers in Comma Separated Format (default: 64,128)
7. --num_folds NUM_FOLDS
                        Number of Folds (default: 10)


For example, running the following command `python unbiased_k_fold_train.py --epochs=100 --batch_size=10 --conv_kernel=3 --pooling_kernel=2 --layers="64,128,256" --num_folds=10` will train a model for 100 epochs, with batch size of 10, and three convolutional layers with 64, 128, and 256 hidden neurons respectively and with a 3x3 kernel and 2x2 pooling kernel size and using 10-fold cross validation strategy on the unbiased dataset.

During each fold, 19% of the training data is used as the validation set to find the best performing model of the fold. Both splitting the data into k folds and train and validation split use the stratified splitting technique using the combination of gender, age group, and label to ensure we have the same distribution of each combination. Because the random_state variable and torch.manual_seed has been set, the code will produce the same results and splits when you run them multiple times.

After running the code, the training and validation loss of the model will be stored in a folder called ___unbiased_losses_k_fold___ (k is replaced by the actual value of k) in the same directory as the README. The best performing model (i.e., the model with the lowest validation loss) after the _10_ th epoch and the model trained after the _n_ th epoch are stored in a folder called ___unbiased_saved_models_k_fold___ (k is replaced by the actual value of k) in the same directory as the README file.

The evaluation file is the ___unbiased_k_fold_eval.py___ file. You can evaluate any of you trained CNN models using the ___unbiased_k_fold_eval.py___ file without the need to change the code. The evaluation script accepts the following arguments:

1. -h, --help            show the help message and exit
2. --batch_size BATCH_SIZE
                        Batch Size (default: 10)
3. --conv_kernel CONV_KERNEL
                        Kernel Size for the Conv Module (default: 3)
4. --pooling_kernel POOLING_KERNEL
                        Kernel Size for the Pooling Module (default: 2)
5. --layers LAYERS       Layers in Comma Separated Format (default: 64,128)
6. --num_folds NUM_FOLDS
                        Number of Folds (default: 10)

For example, running the following command `python unbiased_k_fold_eval.py --batch_size=10 --conv_kernel=3 --pooling_kernel=2 --layers="64,128,256" --num_folds=10` will evaluate the previously trained model using 10-fold cross validation technique with three convolutional layers each with 64, 128, and 256 hidden neurons respectively and with a 3x3 kernel and 2x2 pooling kernel size on the unbiased dataset. The code uses the best and final models stored in the ___unbiased_saved_models_k_fold___ (k will be replaced by its actual number) directory. The batch size is used to feed the test data in mini-batches to the model.

The evaluation code first plots the training and evaluation losses of the model for each fold to better understand whether the model has converged. Then it will calculate the following for both the best and final models of each fold:
- Confusion Matrix
- Micro Precision
- Micro Recall
- Micro F1 Score
- Micro Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score

Finally, the code will store the confusion matrix, metrics, and the train and validation losses plot in a directory called ___unbiased_results_k_fold___ (k will be replaced by its actual value) living in the same directory as the README file.
