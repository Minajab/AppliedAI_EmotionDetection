import torch
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import EmotionCNN 

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    
    acc = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    precision_micro, recall_micro, fscore_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')

    metrics = {
        'conf_matrix': conf_matrix,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'fscore_micro': fscore_micro
    }

    return metrics

if __name__ == '__main__':
    # Path to the dataset directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'Dataset', 'Cleared_data') 

    # Data transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    _, _, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


    num_classes = 4 
    image_width = 48  
    image_height = 48  

    main_model = EmotionCNN(num_classes=num_classes, width=image_width, height=image_height)
    variant1 = EmotionCNN(num_classes=num_classes, width=image_width, height=image_height)  # Adjust parameters for variant
    variant2 = EmotionCNN(num_classes=num_classes, width=image_width, height=image_height)  # Adjust parameters for variant

  
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_model.load_state_dict(torch.load(os.path.join(script_dir, 'main_model.pth')))
    variant1.load_state_dict(torch.load(os.path.join(script_dir, 'variant1_model.pth')))
    variant2.load_state_dict(torch.load(os.path.join(script_dir, 'variant2_model.pth')))

    
    main_metrics = evaluate_model(main_model, test_loader)
    variant1_metrics = evaluate_model(variant1, test_loader)
    variant2_metrics = evaluate_model(variant2, test_loader)

 
    print("Main Model Metrics:")
    print(main_metrics)

    print("Variant 1 Metrics:")
    print(variant1_metrics)

    print("Variant 2 Metrics:")
    print(variant2_metrics)
