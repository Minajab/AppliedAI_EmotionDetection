import torch
from torch import nn
import os
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from model import EmotionCNN
import pandas as pd


def add_column_from_category_and_label_values(row):
    value = row['Age Group'] + ' ' + row['Gender'] + ' ' + row['Emotion']
    value = value.replace(" ", "_")
    value = value.replace("-", "_")
    return value


def main(args):
    cwd = os.getcwd()
    # Setting Manual Seed for Reproducibility
    torch.manual_seed(32)
    # Configuration parameters
    num_classes = 4
    learning_rate = 0.001
    num_epochs = args.epochs
    batch_size = args.batch_size
    conv_kernel_size = args.conv_kernel
    pooling_kernel_size = args.pooling_kernel
    if args.layers.strip() == '':
        raise Exception("--layers cannot be empty!")
    layers = [int(layer.strip()) for layer in args.layers.split(',')]
    Input_directory = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'Unbiased_Dataset'))

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
    train_indices, test_val_idx = train_test_split(indices, test_size=0.3, stratify=category_class_combinations,
                                                   random_state=42)
    val_indices, test_indices = train_test_split(test_val_idx, test_size=0.5,
                                                 stratify=[category_class_combinations_mapping[i] for i in
                                                           test_val_idx],
                                                 random_state=42)

    # Define datasets and data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create a DataLoader for your dataset
    print('Creating Train DataLoader...')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('Creating Test DataLoader...')
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    print('Creating Validation DataLoader...')
    val_loader = DataLoader(validation_dataset, batch_size, shuffle=False)

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_width = 0
    image_height = 0
    for batch in train_loader:
        first_batch_shape = batch[0].shape
        image_width = first_batch_shape[2]
        image_height = first_batch_shape[3]
        break

    # Create a model instance and define the loss function and optimizer
    model = EmotionCNN(num_classes, image_width, image_height, kernel_size=conv_kernel_size,
                       pooling_kernel=pooling_kernel_size, layers=layers)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f'Device: {device}')
    print(f'Total Number of Images: {len(images_metadata_df)}')
    print(f'Total Number of Training Images: {len(train_dataset)}')
    print(f'Total Number of Validation Images: {len(validation_dataset)}')
    print(f'Total Number of Testing Images: {len(test_dataset)}')
    print(f'Batch Size: {batch_size}')
    print(f'Number of Epochs: {num_epochs}')
    print(f'Number of Classes: {num_classes}')
    print(f'Image Width: {image_width}')
    print(f'Image Height: {image_height}')

    # Training loop
    print('Training...')
    best_validation_error = np.inf
    train_losses = []
    validation_losses = []
    with tqdm(range(num_epochs), unit='epoch') as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f'Epoch {epoch}')
            model.train()
            running_loss = 0.0
            total_data = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * target.size(0)
                total_data += target.size(0)

            # Validation loop
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_total_data = 0
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item() * target.size(0)
                    val_total_data += target.size(0)

            postfix = {
                'Train Loss': running_loss / total_data,
                'Validation Loss': val_loss / val_total_data,
            }
            train_losses.append(postfix['Train Loss'])
            validation_losses.append(postfix['Validation Loss'])
            tepoch.set_postfix(ordered_dict=postfix)
            # We want the model to be trained for at least 10 epochs. Epoch starts from 0, so it should be at least 9
            if epoch >= 9 and (val_loss / val_total_data) < best_validation_error:
                best_validation_error = val_loss / val_total_data
                model_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), '..'))
                if not os.path.exists(os.path.join(model_path, 'unbiased_saved_models')):
                    os.mkdir(os.path.join(model_path, 'unbiased_saved_models'))
                model_path = os.path.join(model_path, 'unbiased_saved_models')
                layers_str = '_'.join(map(str, layers))
                model_name = f'best_model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}_{layers_str}'
                torch.save(model.state_dict(), os.path.join(model_path, f'{model_name}.pth'))

    print('Training has been Concluded...')

    # Save the trained model to a file
    print('Saving the Trained Model...')
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.exists(os.path.join(model_path, 'unbiased_saved_models')):
        os.mkdir(os.path.join(model_path, 'unbiased_saved_models'))
    model_path = os.path.join(model_path, 'unbiased_saved_models')
    layers_str = '_'.join(map(str, layers))
    model_name = f'final_model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}_{layers_str}'
    torch.save(model.state_dict(), os.path.join(model_path, f'{model_name}.pth'))
    losses_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.exists(os.path.join(losses_path, 'unbiased_losses')):
        os.mkdir(os.path.join(losses_path, 'unbiased_losses'))
    losses_path = os.path.join(losses_path, 'unbiased_losses')
    layers_str = '_'.join(map(str, layers))
    losses_name = f'model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}_{layers_str}'
    np.savetxt(os.path.join(losses_path, f'{losses_name}_train.csv'), train_losses, delimiter=',')
    np.savetxt(os.path.join(losses_path, f'{losses_name}_val.csv'), validation_losses, delimiter=',')

    print('Testing Using the Final Model...')
    # Test the trained model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing', unit='batch'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_accuracy = correct / total
    print(f'Final Model\'s Test Accuracy: {test_accuracy:.4f}')

    print('Testing Using the Best Model...')
    # Test the trained model
    layers_str = '_'.join(map(str, layers))
    best_model_name = f'best_model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}_{layers_str}'
    model.load_state_dict(torch.load(os.path.join(model_path, f'{best_model_name}.pth')))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing', unit='batch'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_accuracy = correct / total
    print(f'Best Model\'s Test Accuracy: {test_accuracy:.4f}')


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Train an Unbiased Emotion Detection Model.')

    # Define command-line arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of Epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch Size (default: 10)')
    parser.add_argument('--conv_kernel', type=int, default=3, help='Kernel Size for the Conv Module (default: 3)')
    parser.add_argument('--pooling_kernel', type=int, default=2, help='Kernel Size for the Pooling Module (default: 2)')
    parser.add_argument('--layers', type=str, default='64,128',
                        help='Layers in Comma Separated Format (default: 64,128)')

    # Parse the command-line arguments
    args = parser.parse_args()
    main(args)
