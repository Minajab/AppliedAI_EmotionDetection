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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate



def add_column_from_category_and_label_values(row):
    value = row['Age Group'] + ' ' + row['Gender'] + ' ' + row['Emotion']
    value = value.replace(" ", "_")
    value = value.replace("-", "_")
    return value

def calculate_metrics(y_true, y_pred, num_labels):
    conf_matrix = np.zeros((num_labels, num_labels), dtype=int)

    for true, pred in zip(y_true, y_pred):
        conf_matrix[true, pred] += 1

    # Micro metrics
    micro_precision = np.sum(np.diag(conf_matrix)) / max(1, np.sum(conf_matrix))
    micro_recall = np.sum(np.diag(conf_matrix)) / max(1, np.sum(conf_matrix))
    micro_f1 = 2 * (micro_precision * micro_recall) / max(1, micro_precision + micro_recall)
    micro_accuracy = np.sum(np.diag(conf_matrix)) / max(1, np.sum(conf_matrix))

    # Macro metrics
    macro_precision = np.mean(
        [conf_matrix[i, i] / max(1, np.sum(conf_matrix[:, i])) if np.sum(conf_matrix[:, i]) != 0 else 0 for i in
         range(num_labels)])
    macro_recall = np.mean(
        [conf_matrix[i, i] / max(1, np.sum(conf_matrix[i, :])) if np.sum(conf_matrix[i, :]) != 0 else 0 for i in
         range(num_labels)])

    # Avoid division by zero in macro F1
    macro_f1 = 0.0
    for i in range(num_labels):
        precision = conf_matrix[i, i] / max(1, np.sum(conf_matrix[:, i]))
        recall = conf_matrix[i, i] / max(1, np.sum(conf_matrix[i, :]))
        if precision + recall > 0:
            macro_f1 += 2 * precision * recall / (precision + recall)
    macro_f1 /= max(1, num_labels)  # Normalize by the number of classes

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
    conv_kernel_size = args.conv_kernel
    pooling_kernel_size = args.pooling_kernel
    if args.layers.strip() == '':
        raise Exception("--layers cannot be empty!")
    layers = [int(layer.strip()) for layer in args.layers.split(',')]
    Input_directory = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'Unbiased_Dataset'))

    cwd = os.getcwd()

    results_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.exists(os.path.join(results_path, 'unbiased_results')):
        os.mkdir(os.path.join(results_path, 'unbiased_results'))
    results_path = os.path.join(os.path.join(results_path, 'unbiased_results'))

    layers_str = '_'.join(map(str, layers))
    model_name = f'best_model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}_{layers_str}'
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'unbiased_saved_models'))

    if not os.path.exists(os.path.join(model_path, f'{model_name}.pth')):
        raise Exception('A model with the given structure has not been trained!')

    # Define transformations for image data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=Input_directory, transform=transform)
    metadata = pd.read_csv(os.path.join(cwd, '..', 'Meta_Data', 'unbiased_dataset.csv'))
    metadata['Age Group'] = metadata['Age Group'].str.strip().str.lower()
    metadata['Gender'] = metadata['Gender'].str.strip().str.lower()
    metadata['Emotion'] = metadata['Emotion'].str.strip().str.lower()
    metadata['mixed_col'] = metadata.apply(add_column_from_category_and_label_values, axis=1)

    number_of_images = len(dataset)

    print('Attaching Metadata of Each Image')
    images_metadata = []
    valid_indices = []
    for idx, (path_val, class_val) in enumerate(dataset.imgs):
        class_folder, image_name = path_val.split(os.sep)[-2:]
        class_folder = class_folder.lower()
        class_metadata = metadata[metadata['Emotion'] == class_folder]
        image_metadata = class_metadata[class_metadata['Filename'] == image_name]
        if len(image_metadata) > 0:
            valid_indices.append(idx)
            images_metadata.append({
                'Idx': idx,
                'Gender': image_metadata.iloc[0]['Gender'],
                'Age Group': image_metadata.iloc[0]['Age Group'],
                'MixedCol': image_metadata.iloc[0]['mixed_col'],
            })
        if len(class_metadata[class_metadata['Filename'] == image_name]) > 1:
            print(class_metadata[class_metadata['Filename'] == image_name])
            raise Exception('More than one file with same name exists!')

    images_metadata_df = pd.DataFrame(images_metadata)

    indices = list(images_metadata_df['Idx'].values)
    category_class_combinations = list(images_metadata_df['MixedCol'].values)

    category_class_combinations_mapping = {val: category_class_combinations[idx] for idx, val in enumerate(indices)}

    # Get the labels of the dataset
    labels = [dataset[i][1] for i in indices]

    labels_mapping = {val: labels[idx] for idx, val in enumerate(indices)}

    print('Splitting the Dataset...')
    # Use train_test_split to split the dataset into train, validation, and test sets
    _, test_val_idx = train_test_split(indices, test_size=0.3, stratify=category_class_combinations,
                                                   random_state=42)
    _, test_indices = train_test_split(test_val_idx, test_size=0.5,
                                                 stratify=[category_class_combinations_mapping[i] for i in
                                                           test_val_idx],
                                                 random_state=42)
    test_indices = [idx for idx in test_indices if idx in valid_indices]

    # Define datasets and data loaders
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create a DataLoader for your dataset
    print('Creating Test DataLoader...')
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_metadata_df = images_metadata_df[images_metadata_df['Idx'].isin(test_indices)]
    images_metadata_df['Order'] = images_metadata_df['Idx'].map({value: i for i, value in enumerate(test_indices)})
    # Sort the DataFrame based on the temporary 'Order' column
    images_metadata_df = images_metadata_df.sort_values(by='Order')
    # Drop the temporary 'Order' column if needed
    images_metadata_df = images_metadata_df.drop('Order', axis=1).reset_index()

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

    # Calculate Model's Performance on the Test Set for Both the Best and the Final Models
    all_results = {}
    for model_type in ['final', 'best']:
        print(f'Testing Using the {model_type.title()} Model...')
        # Test the trained model
        model.to(device)
        layers_str = '_'.join(map(str, layers))
        model_name = f'{model_type}_model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}_{layers_str}'
        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'unbiased_saved_models'))
        model.load_state_dict(torch.load(os.path.join(model_path, f'{model_name}.pth')))
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        true_classes = []
        predicted_classes = []
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f'Testing {model_type.title()} Model', unit='batch'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(target.cpu().numpy().tolist())

        headers = ['Attribute', 'Group', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        all_results[model_type] = []

        for age_group in images_metadata_df['Age Group'].unique():
            age_mask = images_metadata_df['Age Group'] == age_group
            subgroup_indices = age_mask.index[age_mask].tolist()

            subgroup_labels = [all_labels[i] for i in subgroup_indices]
            subgroup_predictions = [all_predictions[i] for i in subgroup_indices]

            metrics = calculate_metrics(subgroup_labels, subgroup_predictions, num_classes)

            group_results = []
            group_results.append('Age Group')
            group_results.append(age_group.title())
            group_results.append(metrics['Micro Accuracy'])
            group_results.append(metrics['Macro Precision'])
            group_results.append(metrics['Macro Recall'])
            group_results.append(metrics['Macro F1 Score'])
            all_results[model_type].append(group_results)

            conf_matrix = metrics['Confusion Matrix']
            for metric, value in metrics.items():
                if metric == "Confusion Matrix":
                    print(f"{metric}:\n{value}")
                else:
                    print(f"{metric}: {value:.4f}")

            # Plot confusion matrix using seaborn
            classes = [str(i) for i in range(num_classes)]
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1.2)
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            suffix = age_group.lower().split(' ')
            suffix = '_'.join(suffix)
            plt.savefig(os.path.join(results_path, f'{model_name}_age_{suffix}_conf_matrix.png'), format='png')

            del metrics['Confusion Matrix']
            # Convert metrics dictionary to a pandas DataFrame
            df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

            # Save to CSV
            df.to_csv(os.path.join(results_path, f'{model_name}_age_{suffix}_metrics.csv'))

        df = pd.DataFrame(all_results[model_type], columns=headers)
        df = df[df['Attribute'] == 'Age Group']
        average = ['Age Group', 'Average']
        for column in headers:
            if column in ['Attribute', 'Group']:
                continue
            average.append(df[column].mean())
        all_results[model_type].append(average)

        for gender_group in images_metadata_df['Gender'].unique():
            gender_group_mask = images_metadata_df['Gender'] == gender_group
            subgroup_indices = gender_group_mask.index[gender_group_mask].tolist()

            subgroup_labels = [all_labels[i] for i in subgroup_indices]
            subgroup_predictions = [all_predictions[i] for i in subgroup_indices]

            metrics = calculate_metrics(subgroup_labels, subgroup_predictions, num_classes)

            group_results = []
            group_results.append('Gender')
            group_results.append(gender_group.title())
            group_results.append(metrics['Micro Accuracy'])
            group_results.append(metrics['Macro Precision'])
            group_results.append(metrics['Macro Recall'])
            group_results.append(metrics['Macro F1 Score'])
            all_results[model_type].append(group_results)

            conf_matrix = metrics['Confusion Matrix']
            for metric, value in metrics.items():
                if metric == "Confusion Matrix":
                    print(f"{metric}:\n{value}")
                else:
                    print(f"{metric}: {value:.4f}")

            # Plot confusion matrix using seaborn
            classes = [str(i) for i in range(num_classes)]
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1.2)
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            suffix = gender_group.lower().split(' ')
            suffix = '_'.join(suffix)
            plt.savefig(os.path.join(results_path, f'{model_name}_gender_{suffix}_conf_matrix.png'), format='png')

            del metrics['Confusion Matrix']
            # Convert metrics dictionary to a pandas DataFrame
            df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

            # Save to CSV
            df.to_csv(os.path.join(results_path, f'{model_name}_gender_{suffix}_metrics.csv'))

        df = pd.DataFrame(all_results[model_type], columns=headers)
        df = df[df['Attribute'] == 'Gender']
        average = ['Gender', 'Average']
        for column in headers:
            if column in ['Attribute', 'Group']:
                continue
            average.append(df[column].mean())
        all_results[model_type].append(average)

        df = pd.DataFrame(all_results[model_type], columns=headers)
        average = ['Overall System Average', ' ']
        for column in headers:
            if column in ['Attribute', 'Group']:
                continue
            average.append(df[column].mean())
        all_results[model_type].append(average)

        metrics = calculate_metrics(all_labels, all_predictions, num_classes)
        overall = ['Overall System Performance', ' ']
        overall.append(metrics['Micro Accuracy'])
        overall.append(metrics['Macro Precision'])
        overall.append(metrics['Macro Recall'])
        overall.append(metrics['Macro F1 Score'])
        all_results[model_type].append(overall)

        print()
        print('===============================================================================')
        print()


    for model_type in ['final', 'best']:
        print()
        print(f'Bias Results of the {model_type.title()} Model...')
        table = tabulate(all_results[model_type], headers, tablefmt="fancy_grid")
        print(table)


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Test the Bias of the Emotion Detection Model.')

    # Define command-line arguments
    parser.add_argument('--batch_size', type=int, default=10, help='Batch Size (default: 10)')
    parser.add_argument('--conv_kernel', type=int, default=3, help='Kernel Size for the Conv Module (default: 3)')
    parser.add_argument('--pooling_kernel', type=int, default=2, help='Kernel Size for the Pooling Module (default: 2)')
    parser.add_argument('--layers', type=str, default='64,128',
                        help='Layers in Comma Separated Format (default: 64,128)')

    # Parse the command-line arguments
    args = parser.parse_args()
    main(args)
