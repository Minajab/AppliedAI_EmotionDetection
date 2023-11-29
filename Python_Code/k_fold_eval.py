import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from model import EmotionCNN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate


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
    batch_size = args.batch_size
    num_folds = args.num_folds
    conv_kernel_size = args.conv_kernel
    pooling_kernel_size = args.pooling_kernel
    if args.layers.strip() == '':
        raise Exception("--layers cannot be empty!")
    layers = [int(layer.strip()) for layer in args.layers.split(',')]
    Input_directory = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'Dataset'))

    results_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.exists(os.path.join(results_path, f'results_{num_folds}_fold')):
        os.mkdir(os.path.join(results_path, f'results_{num_folds}_fold'))
    results_path = os.path.join(os.path.join(results_path, f'results_{num_folds}_fold'))

    layers_str = '_'.join(map(str, layers))
    model_name = f'best_model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}_{layers_str}_fold_1'
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', f'saved_models_{num_folds}_fold'))

    if not os.path.exists(os.path.join(model_path, f'{model_name}.pth')):
        raise Exception('A model with the given structure has not been trained!')

    print('Loading the Data...')

    # Define transformations for image data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=Input_directory, transform=transform)

    number_of_images = len(dataset)
    # Get the indices of the original dataset
    all_indices = np.array(list(range(number_of_images)))
    # Get the labels of the dataset
    all_labels = np.array([dataset[i][1] for i in range(number_of_images)])

    print('Splitting the Dataset...')

    final_table = []
    best_table = []
    headers = ['Fold', 'Micro Precision', 'Micro Recall', 'Micro F1 Score', 'Accuracy', 'Macro Precision',
               'Macro Recall', 'Macro F1 Score']
    k_fold = StratifiedKFold(n_splits=num_folds)
    for fold, (train_index, test_index) in enumerate(k_fold.split(all_indices, all_labels), start=1):
        print(f'Fold {fold}:')

        test_indices = all_indices[test_index]

        # Define datasets and data loaders
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        # Create a DataLoader for your dataset
        print('Creating Test DataLoader...')
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        # Set device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_device = 'cuda:0' if torch.cuda.is_available() else device

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
        print(f'Total Number of Testing Images: {len(test_dataset)}')
        print(f'Batch Size: {batch_size}')
        print(f'Number of Classes: {num_classes}')
        print(f'Image Width: {image_width}')
        print(f'Image Height: {image_height}')

        print()
        print('===============================================================================')
        print()

        # Plot Train and Validation Loss
        losses_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', f'losses_{num_folds}_fold'))
        layers_str = '_'.join(map(str, layers))
        losses_name = f'model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}_{layers_str}_fold_{fold}'
        train_losses = pd.read_csv(os.path.join(losses_path, f'{losses_name}_train.csv'), header=None).values.flatten()
        val_losses = pd.read_csv(os.path.join(losses_path, f'{losses_name}_val.csv'), header=None).values.flatten()

        epochs = np.arange(1, len(train_losses) + 1)

        # Find the index of the minimum validation loss after the 10th epoch
        min_val_loss_index = np.argmin(val_losses[9:]) + 9  # Adding 9 to get the correct index

        plt.style.use('default')

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Train Loss', marker='o')
        plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
        plt.axvline(x=min_val_loss_index, color='r', linestyle='--', label='Min Val Loss')

        # Adding labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold} - Model Convergence: Train vs Validation Loss')
        plt.legend()  # Show legend for clarity
        plt.grid(True)  # Add grid for better readability

        # Show the plot
        model_name = f'model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}_{layers_str}_fold_{fold}'
        plt.savefig(os.path.join(results_path, f'{model_name}_train_val_plot.png'), format='png')
        plt.close()

        # Calculate Model's Performance on the Test Set for Both the Best and the Final Models
        for model_type in ['final', 'best']:
            print(f'Testing Using the {model_type.title()} Model...')
            # Test the trained model
            model.to(device)
            layers_str = '_'.join(map(str, layers))
            model_name = f'{model_type}_model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}_{layers_str}_fold_{fold}'
            model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', f'saved_models_{num_folds}_fold'))
            model.load_state_dict(
                torch.load(os.path.join(model_path, f'{model_name}.pth'), map_location=weights_device))
            model.to(device)
            model.eval()
            correct = 0
            total = 0
            true_classes = []
            predicted_classes = []
            with torch.no_grad():
                for data, target in tqdm(test_loader, desc=f'Fold {fold} - Testing {model_type.title()} Model',
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
            metrics = calculate_metrics(y_true, y_pred)
            conf_matrix = metrics['Confusion Matrix']
            for metric, value in metrics.items():
                if metric == "Confusion Matrix":
                    print(f"{metric}:\n{value}")
                else:
                    print(f"{metric}: {value:.4f}")

            # Plot confusion matrix using seaborn
            classes = [str(i) for i in range(max(max(y_true), max(y_pred)) + 1)]
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1.2)
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(results_path, f'{model_name}_conf_matrix.png'), format='png')
            plt.close('all')

            del metrics['Confusion Matrix']
            # Convert metrics dictionary to a pandas DataFrame
            df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

            # Save to CSV
            df.to_csv(os.path.join(results_path, f'{model_name}_metrics.csv'))

            values = [fold] + list(df['Value'].values)
            if model_type == 'final':
                final_table.append(values)
            else:
                best_table.append(values)

            print()
            print('===============================================================================')
            print()

    df = pd.DataFrame(final_table, columns=headers)
    average = ['Average']
    for column in headers:
        if column == 'Fold':
            continue
        average.append(df[column].mean())
    final_table.append(average)
    table = tabulate(final_table, headers, tablefmt="grid")
    print('Final Model Results:')
    print(table)
    df = pd.DataFrame(best_table, columns=headers)
    average = ['Average']
    for column in headers:
        if column == 'Fold':
            continue
        average.append(df[column].mean())
    best_table.append(average)
    table = tabulate(best_table, headers, tablefmt="grid")
    print('Best Model Results:')
    print(table)


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='K-Fold Cross Validation: Test the Emotion Detection Model.')

    # Define command-line arguments
    parser.add_argument('--batch_size', type=int, default=10, help='Batch Size (default: 10)')
    parser.add_argument('--conv_kernel', type=int, default=3, help='Kernel Size for the Conv Module (default: 3)')
    parser.add_argument('--pooling_kernel', type=int, default=2, help='Kernel Size for the Pooling Module (default: 2)')
    parser.add_argument('--layers', type=str, default='64,128',
                        help='Layers in Comma Separated Format (default: 64,128)')
    parser.add_argument('--num_folds', type=int, default=10,
                        help='Number of Folds (default: 10)')

    # Parse the command-line arguments
    args = parser.parse_args()
    main(args)
