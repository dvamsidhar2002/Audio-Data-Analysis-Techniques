import os
import torch
import torch.nn as nn
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch.nn.functional as F

# Split the dataset into training and validation sets
def split_dataset(dataset, validation_split=0.2):
    num_samples = len(dataset)
    split_idx = int(num_samples * (1 - validation_split))
    indices = torch.randperm(num_samples).tolist()
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    return train_subset, val_subset

# Load the validation set
validation_split = 0.2
train_subset, val_subset = split_dataset(dataset, validation_split)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Load the saved model
model = GenreRNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to('cuda')
model.load_state_dict(torch.load("genre_rnn_model.pth"))
model.eval()

# Evaluate the model
def evaluate_model(model, dataloader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=genres))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# Perform evaluation
print("Evaluating model on validation set...")
evaluate_model(model, val_loader)
