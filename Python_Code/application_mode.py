import math
from random import randint
import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from model import EmotionCNN


def calculate_metrics(y_true, y_pred):
    # Calculate confusion matrix
    classes = set(y_true) | set(y_pred)
    num_classes = len(classes)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        conf_matrix[true, pred] += 1

    # Micro metrics
    micro_precision = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    micro_recall = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    micro_accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

    # Macro metrics
    macro_precision = np.mean(
        [conf_matrix[i, i] / np.sum(conf_matrix[:, i]) if np.sum(conf_matrix[:, i]) != 0 else 0 for i in
         range(num_classes)])
    macro_recall = np.mean(
        [conf_matrix[i, i] / np.sum(conf_matrix[i, :]) if np.sum(conf_matrix[i, :]) != 0 else 0 for i in
         range(num_classes)])
    macro_f1 = np.mean(
        [2 * (conf_matrix[i, i] / np.sum(conf_matrix[i, :]) * conf_matrix[i, i] / np.sum(conf_matrix[:, i])) /
         (conf_matrix[i, i] / np.sum(conf_matrix[i, :]) + conf_matrix[i, i] / np.sum(conf_matrix[:, i]))
         if (np.sum(conf_matrix[i, :]) != 0 and np.sum(conf_matrix[:, i]) != 0) else 0 for i in range(num_classes)])

    return {
        "Micro Precision": micro_precision,
        "Micro Recall": micro_recall,
        "Micro F1 Score": micro_f1,
        "Micro Accuracy": micro_accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1 Score": macro_f1,
        "Confusion Matrix": conf_matrix
    }


def main(args):
    # Setting Manual Seed for Reproducibility
    torch.manual_seed(32)
    # Configuration parameters
    num_classes = 4
    batch_size = 10
    conv_kernel_size = args.conv_kernel
    pooling_kernel_size = args.pooling_kernel
    sample = args.data.strip()
    if args.layers.strip() == '':
        raise Exception("--layers cannot be empty!")
    layers = [int(layer.strip()) for layer in args.layers.split(',')]
    Input_directory = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'Dataset'))

    layers_str = '_'.join(map(str, layers))
    model_name = f'best_model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}_{layers_str}'
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'saved_models'))

    if not os.path.exists(os.path.join(model_path, f'{model_name}.pth')):
        raise Exception('A model with the given structure has not been trained!')

    # Define transformations for image data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    print('Creating Application Dataset')
    dataset = datasets.ImageFolder(root=Input_directory, transform=transform)

    total_number_of_images = len(dataset)

    if sample == 'all':
        number_of_images = total_number_of_images
        # Get the indices of the original dataset
        indices = list(range(number_of_images))
        # Get the labels of the dataset
        labels = [dataset[i][1] for i in range(number_of_images)]
    elif sample == 'rand':
        number_of_images = 1
        # Get the indices of the original dataset
        indices = [randint(0, len(dataset) - 1)]
        # Get the labels of the dataset
        labels = [dataset[i][1] for i in indices]
    else:
        number_of_images = math.ceil(float(sample) * total_number_of_images)
        # Get the indices of the original dataset
        indices = list(range(total_number_of_images))
        # Get the labels of the dataset
        labels = [dataset[i][1] for i in range(total_number_of_images)]
        if float(sample) == 0.0 or float(sample) > 1.0:
            number_of_images = 1
            indices = [int(sample)]
            # Get the labels of the dataset
            labels = [dataset[i][1] for i in indices]

    print('Splitting the Dataset...')
    # Use train_test_split to split the dataset into train, validation, and test sets
    if total_number_of_images == number_of_images or number_of_images == 1:
        test_indices = indices
    else:
        _, test_indices = train_test_split(indices, test_size=(number_of_images / total_number_of_images),
                                           stratify=labels, random_state=42)

    # Define datasets and data loaders
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create a DataLoader for your dataset
    print('Creating Application DataLoader...')
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_width = 0
    image_height = 0
    for batch in test_loader:
        first_batch_shape = batch[0].shape
        image_width = first_batch_shape[2]
        image_height = first_batch_shape[3]
        break

    # Create a model instance and define the loss function and optimizer
    model = EmotionCNN(num_classes, image_width, image_height, kernel_size=conv_kernel_size,
                       pooling_kernel=pooling_kernel_size, layers=layers)

    model.to(device)

    print(f'Device: {device}')
    print(f'Total Number of Images: {len(dataset)}')
    print(f'Total Number of Images for the Application: {len(test_dataset)}')
    print(f'Batch Size: {batch_size}')
    print(f'Number of Classes: {num_classes}')
    print(f'Image Width: {image_width}')
    print(f'Image Height: {image_height}')

    print()
    print('===============================================================================')
    print()

    # Calculate Model's Performance on the Application Set for Both the Best and the Final Models
    for model_type in ['final', 'best']:
        print(f'Running the {model_type.title()} Model in Application Mode...')
        # Running the trained model
        model.to(device)
        layers_str = '_'.join(map(str, layers))
        model_name = f'{model_type}_model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}_{layers_str}'
        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'saved_models'))
        model.load_state_dict(torch.load(os.path.join(model_path, f'{model_name}.pth')))
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        true_classes = []
        predicted_classes = []
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f'Running the {model_type.title()} Model in Application Mode',
                                     unit='batch'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                predicted_classes += list(predicted.cpu().numpy())
                true_classes += list(target.cpu().numpy())
                correct += (predicted == target).sum().item()

        y_true = np.array(true_classes)
        y_pred = np.array(predicted_classes)
        if number_of_images > 1:
            metrics = calculate_metrics(y_true, y_pred)
            for metric, value in metrics.items():
                if metric == "Confusion Matrix":
                    print(f"{metric}:\n{value}")
                else:
                    print(f"{metric}: {value:.4f}")
        else:
            print()
            print(f'True Class:', y_true.item())
            print(f'Predicted Class:', y_pred.item())
            print(f'Correctly Classified: {"Yes" if y_true.item() == y_pred.item() else "No"}')
            print()

        print()
        print('===============================================================================')
        print()


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Application Mode.')

    # Define command-line arguments
    parser.add_argument('--conv_kernel', type=int, default=3, help='Kernel Size for the Conv Module (default: 3)')
    parser.add_argument('--pooling_kernel', type=int, default=2, help='Kernel Size for the Pooling Module (default: 2)')
    parser.add_argument('--layers', type=str, default='64,128',
                        help='Layers in Comma Separated Format (default: 64,128)')
    parser.add_argument('--data', type=str, default='rand',
                        help='The Data You Want to Run the Model On. The value can be "all", "rand", any number '
                             'between 0 and 1 representing a split of the dataset, or a single number representing a '
                             'specific datapoint index from the dataset (default: "rand")')

    # Parse the command-line arguments
    args = parser.parse_args()
    main(args)
