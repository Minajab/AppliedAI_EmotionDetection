import torch.nn as nn
import math


def conv_output_size(input_size, kernel_size, stride=1, padding=0):
    return math.floor((input_size + 2 * padding - kernel_size) / stride) + 1


def maxpool_output_size(input_size, kernel_size, stride=1, padding=0):
    return math.floor((input_size + 2 * padding - kernel_size) / stride) + 1


# Define a customizable CNN model for emotion detection
class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int, width: int, height: int, kernel_size=3, pooling_kernel=2, layers=[64, 128]):
        super(EmotionCNN, self).__init__()

        # Initial input channels
        in_channels = 1

        # Define the convolutional blocks in a loop
        model_layers = []
        conv_output_width = width
        conv_output_height = height
        for i in range(len(layers)):
            out_channels = layers[i]
            model_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size))
            model_layers.append(nn.BatchNorm2d(out_channels))
            model_layers.append(nn.ReLU())
            model_layers.append(nn.MaxPool2d(kernel_size=pooling_kernel))
            model_layers.append(nn.Dropout2d(p=0.5))
            # Update input channels for the next block
            in_channels = layers[i]

            # Calculate output width and height after convolution and pooling
            conv_output_width = conv_output_size(conv_output_width, kernel_size)
            conv_output_width = maxpool_output_size(conv_output_width, pooling_kernel, stride=pooling_kernel)
            conv_output_height = conv_output_size(conv_output_height, kernel_size)
            conv_output_height = maxpool_output_size(conv_output_height, pooling_kernel, stride=pooling_kernel)


        # Flatten before the fully connected layers
        model_layers.append(nn.Flatten())

        # Fully Connected Layer 1
        model_layers.append(nn.Linear(in_channels * conv_output_width * conv_output_height, in_channels * 2))
        model_layers.append(nn.BatchNorm1d(in_channels * 2))
        model_layers.append(nn.ReLU())
        model_layers.append(nn.Dropout(p=0.5))

        # Fully Connected Layer 2
        model_layers.append(nn.Linear(in_channels * 2, in_channels * 4))
        model_layers.append(nn.BatchNorm1d(in_channels * 4))
        model_layers.append(nn.ReLU())
        model_layers.append(nn.Dropout(p=0.5))

        # Output Layer
        model_layers.append(nn.Linear(in_channels * 4, num_classes))

        # Combine all the layers
        self.model = nn.Sequential(*model_layers)

    def forward(self, x):
        # Convolution Layers
        x = self.model(x)
        return x
