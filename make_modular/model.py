from torch import nn

# model_1
class MNIST_MODEL_1(nn.Module):
    def __init__(self, dropout_rate=.25, in_channel=1, num_classes=10):
        super().__init__()

        self.cnn = nn.Sequential(
            # W_out ((w_in - kernel_size + 2p)/stride) +1
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3), # (b, 1, 28, 28) --> (b, 32, 26, 26)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), # (b, 32, 26, 26) --> (b, 64, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (b, 64, 24, 24) --> (b, 64, 12, 12)
            nn.Dropout(dropout_rate),
            nn.Flatten() # (b, 64, 12, 12) --> (b, 64*12*12)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*12*12, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes)
            )

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out)
        return out
    



# model_2
class MNIST_MODEL_2(nn.Module):
    def __init__(self, dropout_rate=.25, in_channel=1, num_classes=10):
        super().__init__()

        self.cnn = nn.Sequential(
            # W_out ((w_in - kernel_size + 2p)/stride) +1
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3), # (b, 1, 28, 28) --> (b, 32, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), # (b, 32, 26, 26) --> (b, 64, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=1600, out_features=128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=128, out_features=num_classes)
            )

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out)
        return out




# model_3
class MNIST_MODEL_3(nn.Module):
    def __init__(self, dropout_rate=.5, in_channel=1, num_classes=10):
        super().__init__()

        self.cnn = nn.Sequential(
            # W_out ((w_in - kernel_size + 2p)/stride) +1
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3), # (b, 1, 28, 28) --> (b, 32, 26, 26)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), # (b, 32, 26, 26) --> (b, 64, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), 
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*12*12, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes)
            )

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out)
        return out




# model_4
class MNIST_MODEL_4(nn.Module):
    def __init__(self, dropout_rate=.25, in_channel=1, num_classes=10):
        super().__init__()

        self.cnn = nn.Sequential(
            # W_out ((w_in - kernel_size + 2p)/stride) +1
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3), # (b, 1, 28, 28) --> (b, 32, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), # (b, 32, 26, 26) --> (b, 64, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=1600, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes)
            )

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out)
        return out



# model_5
class MNIST_MODEL_5(nn.Module):
    def __init__(self, dropout_rate=.25, in_channel=1, num_classes=10):
        super().__init__()

        self.cnn = nn.Sequential(
            # W_out ((w_in - kernel_size + 2p)/stride) +1
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3), # (b, 1, 28, 28) --> (b, 32, 26, 26)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), # (b, 1, 26, 26) --> (b, 64, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*12*12, out_features=128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=128, out_features=10),
            )

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out)
        return out




class MNIST_MODEL_6(nn.Module):
    def __init__(self, dropout_rate=.25, in_channel=1, num_classes=10):
        super().__init__()

        self.cnn = nn.Sequential(
            # W_out ((w_in - kernel_size + 2p)/stride) +1
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3), # (b, 1, 28, 28) --> (b, 32, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), # (b, 32, 26, 26) --> (b, 64, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=1600, out_features=128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=128, out_features=num_classes)
            )

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out)
        return out



class MNIST_MODEL_7(nn.Module):
    def __init__(self, dropout_rate=.25, in_channel=1, num_classes=10):
        super().__init__()

        self.cnn = nn.Sequential(
            # W_out ((w_in - kernel_size + 2p)/stride) +1
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3), # (b, 1, 28, 28) --> (b, 32, 26, 26)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), # (b, 32, 26, 26) --> (b, 64, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=1600, out_features=128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=128, out_features=num_classes)
            )

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out)
        return out