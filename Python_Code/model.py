import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(p=0.5)
        conv_output_width = conv_output_size(width, kernel_size)
        conv_output_width = maxpool_output_size(conv_output_width, pooling_kernel, stride=pooling_kernel)
        conv_output_height = conv_output_size(height, kernel_size)
        conv_output_height = maxpool_output_size(conv_output_height, pooling_kernel, stride=pooling_kernel)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout2d(p=0.5)
        conv_output_width = conv_output_size(conv_output_width, kernel_size)
        conv_output_width = maxpool_output_size(conv_output_width, pooling_kernel, stride=pooling_kernel)
        conv_output_height = conv_output_size(conv_output_height, kernel_size)
        conv_output_height = maxpool_output_size(conv_output_height, pooling_kernel, stride=pooling_kernel)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * conv_output_width * conv_output_height, 256)
        self.batch_norm_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.batch_norm_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.5)

        # Output layer
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolution Layers
        x = F.max_pool2d(F.relu(self.batch_norm1(self.conv1(x))), self.pooling_kernel)
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.batch_norm2(self.conv2(x))), self.pooling_kernel)
        x = self.dropout2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.dropout_fc1(F.relu(self.batch_norm_fc1(self.fc1(x))))
        x = self.dropout_fc2(F.relu(self.batch_norm_fc2(self.fc2(x))))

        # Output layer
        x = self.fc_out(x)
        return x