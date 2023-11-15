import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import numpy


def Producing_confussion(Evaluation, labels, predicted):
    i = 0
    while i < len(labels):
        if (labels[i] == 0):
            if (predicted[i] == 0):
                Evaluation[0][0] += 1
            else:
                if (predicted[i] == 1):
                    Evaluation[1][1] += 1
                elif (predicted[i] == 2):
                    Evaluation[2][1] += 1
                else:
                    Evaluation[3][1] += 1
        elif (labels[i] == 1):
            if (predicted[i] == 1):
                Evaluation[1][0] += 1
            else:
                if (predicted[i] == 0):
                    Evaluation[0][1] += 1
                elif (predicted[i] == 2):
                    Evaluation[2][1] += 1
                else:
                    Evaluation[3][1] += 1
        elif (labels[i] == 2):
            if (predicted[i] == 2):
                Evaluation[2][0] += 1
            else:
                if (predicted[i] == 0):
                    Evaluation[0][1] += 1
                elif (predicted[i] == 1):
                    Evaluation[1][1] += 1
                else:
                    Evaluation[3][1] += 1
        else:
            if (predicted[i] == 3):
                Evaluation[3][0] += 1
            else:
                if (predicted[i] == 0):
                    Evaluation[0][1] += 1
                elif (predicted[i] == 1):
                    Evaluation[1][1] += 1
                else:
                    Evaluation[2][1] += 1
        i += 1
    return Evaluation


def Micro_evaluation(Evaluation, real_labels):
    Avg_precission = (Evaluation[0][0] + Evaluation[1][0] + Evaluation[2][0] + Evaluation[3][0])/ len(real_labels)
    Avg_recall = Avg_precission
    Avg_f1_score = 2 * (Avg_recall * Avg_precission) / (Avg_recall + Avg_precission)
    print("\nmicro\n")
    print("\nprecission and recall:\n")
    print(Avg_precission)
    print("\nf1_score:\n")
    print(Avg_f1_score)


def Macro_evaluation(Evaluation, labels):
    Avg_precission = ((Evaluation[0][0] / (Evaluation[0][0] + Evaluation[1][0])) + (
            Evaluation[1][0] / (Evaluation[1][0] + Evaluation[1][1])) + (
                              Evaluation[2][0] / (Evaluation[2][0] + Evaluation[2][1])) + (
                              Evaluation[3][0] / (Evaluation[3][0] + Evaluation[3][1]))) / 4
    Avg_recall = (Evaluation[0][0] / labels.count(0) + Evaluation[1][0] / labels.count(1) + Evaluation[2][
        0] / labels.count(2) + Evaluation[3][0] / labels.count(3)) / 4
    Avg_f1_score = ((2 * (
            (Evaluation[0][0] / (Evaluation[0][0] + Evaluation[1][0])) * (Evaluation[0][0] / labels.count(0))) / (
                             (Evaluation[0][0] / (Evaluation[0][0] + Evaluation[1][0])) + (
                             Evaluation[0][0] / labels.count(0))))
                    + ((2 * (Evaluation[1][0] / (Evaluation[1][0] + Evaluation[1][1])) * (
                    Evaluation[1][0] / labels.count(1))) / (
                               (Evaluation[1][0] / (Evaluation[1][0] + Evaluation[1][1])) + (
                               Evaluation[1][0] / labels.count(1))))
                    + ((2 * (Evaluation[2][0] / (Evaluation[2][0] + Evaluation[2][1])) * (Evaluation[2][
                                                                                              0] / labels.count(2))) / (
                               (Evaluation[2][0] / (Evaluation[2][0] + Evaluation[2][1])) + (Evaluation[2][
                                                                                                 0] / labels.count(
                           2))))
                    + ((2 * (Evaluation[3][0] / (Evaluation[3][0] + Evaluation[3][1])) * (
                    Evaluation[3][0] / labels.count(3))) / (
                               (Evaluation[3][0] / (Evaluation[3][0] + Evaluation[3][1])) + (
                               Evaluation[3][0] / labels.count(3))))) / 4
    print("\nmacro\n")
    print("\nprecission:\n")
    print(Avg_precission)
    print("\nrecall:\n")
    print(Avg_recall)
    print("\nf1_score\n")
    print(Avg_f1_score)


if __name__ == '__main__':
    Evaluation = [[0, 0], [0, 0], [0, 0], [0, 0]]
    real_labels = []
    from Trainer import EmotionCNN

    model = EmotionCNN(4, 48, 48, 3)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root='C:\\Users\\mahshad\\Desktop\\Data', transform=transform)
    model.train()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for training and testing
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    learning_rate = 0.001
    num_epochs = 10

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            for label in labels:
                real_labels.append(label)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            print("predicted\n")
            print(predicted)
            print("labels\n")
            print(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            Evaluation = Producing_confussion(Evaluation,
                                              labels, predicted)
    print("\nevaluation\n")
    print(Evaluation)
    Micro_evaluation(Evaluation, real_labels)
    Macro_evaluation(Evaluation, real_labels)
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
