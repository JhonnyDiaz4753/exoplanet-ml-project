import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

# === Añadir ruido a datos para Data Augmentation ===
def add_noise(X, noise_level=0.02):
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    return np.clip(X_noisy, 0, 1)

# === Bloque ResNet 1D ===
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

# === Self-Attention Manual ===
class ManualSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, V)
        return context.mean(dim=1)

# === Modelo CNN + LSTM + Attention mejorado ===
class ResNetLSTM1DWithManualAttention(nn.Module):
    def __init__(self, lstm_hidden=256, lstm_layers=3):
        super().__init__()
        self.layer1 = BasicBlock1D(1, 16)
        self.layer2 = BasicBlock1D(16, 32, downsample=True)
        self.layer3 = BasicBlock1D(32, 64, downsample=True)

        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            bidirectional=True)

        self.attn = ManualSelfAttention(embed_dim=lstm_hidden * 2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = self.attn(lstm_out)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)

# === Función de entrenamiento mejorada ===
def train_model(X, y, output_path='cnn_lstm_attn_model.pt', epochs=100, batch_size=64, lr=1e-4,
                patience=15, model_name='cnn_lstm_attn', augment_noise=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if augment_noise:
        X = np.concatenate([X, add_noise(X)])
        y = np.concatenate([y, y])

    if model_name == 'cnn_lstm_attn':
        model = ResNetLSTM1DWithManualAttention().to(device)
        print("Training CNN + LSTM + Attention model with augmentation =", augment_noise)
    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)

    train_len = int(0.7 * len(dataset))
    val_len = int(0.15 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            correct += ((pred > 0.5) == yb).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += ((pred > 0.5) == yb).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), output_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

    return model, history, test_set

