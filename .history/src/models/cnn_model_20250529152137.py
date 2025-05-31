import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLightCurveClassifier(nn.Module):
    def __init__(self):
        super(CNNLightCurveClassifier, self).__init__()
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
        self.dropout_conv = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * 124, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout_conv(x)

        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

def build_model():
    model = CNNLightCurveClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

# Early stopping clase para usar en entrenamiento
class EarlyStopping:
    def __init__(self, patience=15, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Mejor val_loss: {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping contador: {self.counter} de {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


