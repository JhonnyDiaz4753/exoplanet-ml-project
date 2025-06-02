import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLightCurveClassifier(nn.Module):
    def __init__(self):
        super(CNNLightCurveClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool3 = nn.AdaptiveMaxPool1d(50)  # Reduce a tamaño fijo

        self.fc1 = nn.Linear(64 * 50, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)  # salida binaria

    def forward(self, x):
        # x: (batch_size, 1999, 2) → (batch_size, 2, 1999)
        x = x.permute(0, 2, 1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

def build_model():
    model = CNNLightCurveClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()  # binary crossentropy
    return model, optimizer, criterion

