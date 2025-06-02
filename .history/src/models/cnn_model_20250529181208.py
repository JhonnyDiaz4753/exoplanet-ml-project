import torch
import torch.nn as nn
import torch.nn.functional as F

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


def build_model():
    model = CNNLSTMLightCurveClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5, batch_size=32)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    return model, optimizer, criterion, scheduler
