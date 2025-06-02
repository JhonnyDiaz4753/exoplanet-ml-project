import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

class CNNLSTMLightCurveClassifier(nn.Module):
    def __init__(self, lstm_hidden=128, lstm_layers=1, dropout=0.5):
        super(CNNLSTMLightCurveClassifier, self).__init__()
        # Bloques Conv1D + BatchNorm + ReLU + MaxPool
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.pool5 = nn.MaxPool1d(2)

        self.dropout_conv = nn.Dropout(dropout)

        # LSTM con entrada: (batch, seq_len, features)
        # Conv1d output shape: (batch, channels, length)
        # Necesitamos permutar a (batch, length, channels) para LSTM
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)

        self.dropout_lstm = nn.Dropout(dropout)

        # Fully connected
        self.fc1 = nn.Linear(lstm_hidden, 128)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 2)  # salida 2 clases

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)

        x = self.dropout_conv(x)

        # Pasar a (batch, seq_len, features) para LSTM
        x = x.permute(0, 2, 1)

        # LSTM
        x, (hn, cn) = self.lstm(x)
        # Tomamos la última salida de secuencia para clasificación
        x = x[:, -1, :]

        x = self.dropout_lstm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, epochs=50, lr=3e-4, weight_decay=1e-5, 
                patience=7, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_val_loss = np.inf
    early_stop_counter = 0
    
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            targets = y_batch.detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets)

        train_loss /= len(train_loader.dataset)
        train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
        train_prec = precision_score(train_targets, train_preds, zero_division=0)
        train_rec = recall_score(train_targets, train_preds, zero_division=0)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0)

        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * x_batch.size(0)

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                targets = y_batch.detach().cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets)

        val_loss /= len(val_loader.dataset)
        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        val_prec = precision_score(val_targets, val_preds, zero_division=0)
        val_rec = recall_score(val_targets, val_preds, zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)

        print(f"Época {epoch}/{epochs} - "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Prec: {train_prec:.4f} Rec: {train_rec:.4f} F1: {train_f1:.4f} - "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Prec: {val_prec:.4f} Rec: {val_rec:.4f} F1: {val_f1:.4f}")

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Guardar modelo con mejor val loss
            torch.save(model.state_dict(), "best_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping activado después de {patience} épocas sin mejora.")
                break

    # Cargar el mejor modelo guardado
    model.load_state_dict(torch.load("best_model.pth"))
    return model