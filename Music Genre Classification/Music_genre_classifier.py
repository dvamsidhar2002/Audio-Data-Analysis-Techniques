import os
import torch
import torch.nn as nn
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time

# Parameters
data_dir = "H:\\Audio Data Analysis\\GTZAN Data\\genres_original"
genres = os.listdir(data_dir)
num_epochs = 20
batch_size = 16
learning_rate = 0.001
sample_rate = 22050
n_mfcc = 20
sequence_length = 130  # Based on 30-second audio at 22050 Hz

# Dataset Class
class MusicGenreDataset(Dataset):
    def __init__(self, data_dir, genres, transform=None):
        print("Initializing Dataset...")
        start_time = time.time()
        self.data_dir = data_dir
        self.genres = genres
        self.transform = transform
        self.filepaths = []
        self.labels = []

        for label, genre in enumerate(self.genres):
            genre_path = os.path.join(data_dir, genre)
            files = [os.path.join(genre_path, f) for f in os.listdir(genre_path) if f.endswith(".wav")]
            self.filepaths.extend(files)
            self.labels.extend([label] * len(files))
        end_time = time.time()
        print(f"Dataset initialized with {len(self.filepaths)} files in {end_time - start_time:.2f} seconds.")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        file_path = self.filepaths[idx]
        label = self.labels[idx]
        try:
            waveform, sr = librosa.load(file_path, sr=sample_rate)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            waveform = np.zeros(sample_rate * 30)  # Fallback to silent audio for 30 seconds
            sr = sample_rate
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < sequence_length:
            padding = sequence_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')
        elif mfcc.shape[1] > sequence_length:
            mfcc = mfcc[:, :sequence_length]
        mfcc = torch.tensor(mfcc, dtype=torch.float32)
        return mfcc.T, label  # Transpose to match RNN input shape (seq_len, features)

# RNN Model
class GenreRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(GenreRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Bidirectional doubles hidden size

    def forward(self, x):
        out, _ = self.rnn(x)  # Shape: (batch, seq_len, hidden_size * 2)
        out = out[:, -1, :]  # Take the last time step
        out = self.fc(out)
        return out

# Prepare Data
print("Loading dataset...")
dataset = MusicGenreDataset(data_dir, genres)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Dataset loaded.")

# Model, Loss, and Optimizer
input_size = n_mfcc
hidden_size = 128
num_classes = len(genres)
model = GenreRNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop with Timing
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    print("Starting training...")
    start_time = time.time()
    model.train()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        print(f"Epoch {epoch + 1}/{num_epochs} started...")
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            batch_start_time = time.time()
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Estimate time for the current batch
            batch_end_time = time.time()
            elapsed_time = batch_end_time - batch_start_time
            remaining_batches = len(dataloader) - (batch_idx + 1)
            print(f"Batch {batch_idx + 1}/{len(dataloader)} processed in {elapsed_time:.2f}s. "
                  f"Estimated time left for this epoch: {remaining_batches * elapsed_time:.2f}s.")

        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_end_time - epoch_start_time:.2f}s. "
              f"Loss: {total_loss / len(dataloader):.4f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

train_model(model, dataloader, criterion, optimizer, num_epochs)

# Save Model
print("Saving model...")
torch.save(model.state_dict(), "genre_rnn_model.pth")
print("Model saved.")
