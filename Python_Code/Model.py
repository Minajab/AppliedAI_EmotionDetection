import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import math
import argparse
from tqdm import tqdm
import numpy as np


def conv_output_size(input_size, kernel_size, stride=1, padding=0):
    return math.floor((input_size + 2 * padding - kernel_size) / stride) + 1


def maxpool_output_size(input_size, kernel_size, stride=1, padding=0):
    return math.floor((input_size + 2 * padding - kernel_size) / stride) + 1


# Define a customizable CNN model for emotion detection
class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int, width: int, height: int, kernel_size=3, pooling_kernel=2):
        super(EmotionCNN, self).__init__()
        self.pooling_kernel = 2
        self.num_classes = num_classes

        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        conv_output_width = conv_output_size(width, kernel_size)
        conv_output_width = maxpool_output_size(conv_output_width, pooling_kernel, stride=pooling_kernel)
        conv_output_height = conv_output_size(height, kernel_size)
        conv_output_height = maxpool_output_size(conv_output_height, pooling_kernel, stride=pooling_kernel)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        conv_output_width = conv_output_size(conv_output_width, kernel_size)
        conv_output_width = maxpool_output_size(conv_output_width, pooling_kernel, stride=pooling_kernel)
        conv_output_height = conv_output_size(conv_output_height, kernel_size)
        conv_output_height = maxpool_output_size(conv_output_height, pooling_kernel, stride=pooling_kernel)

        self.fc1 = nn.Linear(64 * conv_output_width * conv_output_height, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), self.pooling_kernel))
        x = F.relu(F.max_pool2d(self.conv2(x), self.pooling_kernel))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training function
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print('Training finished.')


def main(args):
    # Configuration parameters
    num_classes = 4
    learning_rate = 0.001
    num_epochs = args.epochs
    batch_size = args.batch_size
    conv_kernel_size = args.conv_kernel
    pooling_kernel_size = args.pooling_kernel
    Input_directory = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'Dataset'))

    # Define transformations for image data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    # Create a DataLoader for your dataset
    print('Splitting the Dataset...')
    dataset = datasets.ImageFolder(root=Input_directory, transform=transform)
    random_generator = torch.Generator().manual_seed(42)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15],
                                                                   generator=random_generator)
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
                       pooling_kernel=pooling_kernel_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f'Device: {device}')
    print(f'Batch Size: {batch_size}')
    print(f'Number of Epochs: {num_epochs}')
    print(f'Number of Classes: {num_classes}')
    print(f'Image Width: {image_width}')
    print(f'Image Height: {image_height}')

    # Training loop
    print('Training...')
    best_validation_error = np.inf
    with tqdm(range(num_epochs), unit='epoch') as tepoch:
        for epoch in range(num_epochs):
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
            tepoch.set_postfix(ordered_dict=postfix)
            if (val_loss/val_total_data)<best_validation_error:
                best_validation_error = val_loss/val_total_data
                model_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), '..'))
                if not os.path.exist(os.path.join(model_path, 'saved_models')):
                    os.mkdir(os.path.join(model_path, 'saved_models'))
                model_path = os.path.join(model_path,'saved_models')
                model_name = f'best_model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}'
                torch.save(model.state_dict(), os.path.join(model_path,f'{model_name}.pth'))

    print('Training has been Concluded...')
    print('Testing...')
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
    print(f'Test Accuracy: {test_accuracy:.4f}')

    # Save the trained model to a file
    print('Saving the Trained Model...')
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.exist(os.path.join(model_path, 'saved_models')):
        os.mkdir(os.path.join(model_path, 'saved_models'))
    model_path = os.path.join(model_path, 'saved_models')
    model_name = f'final_model_kernel_{conv_kernel_size}_pooling_kernel_{pooling_kernel_size}'
    torch.save(model.state_dict(), os.path.join(model_path, f'{model_name}.pth'))


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Train an Emotion Detection Model with Two CNN Layers.')

    # Define command-line arguments
    parser.add_argument('--epochs', type=int, default=10, help='Number of Epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch Size (default: 10)')
    parser.add_argument('--conv_kernel', type=int, default=3, help='Kernel Size for the Conv Module (default: 3)')
    parser.add_argument('--pooling_kernel', type=int, default=2, help='Kernel Size for the Pooling Module (default: 2)')

    # Parse the command-line arguments
    args = parser.parse_args()
    main(args)
