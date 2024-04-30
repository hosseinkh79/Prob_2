from torch import nn

# model_1
class MNIST_MODEL_1(nn.Module):
    def __init__(self, in_channel=1, num_classes=10) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            # W_out ((w_in - kernel_size + 2p)/stride) +1
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3), # (b, 1, 28, 28) --> (b, 32, 26, 26)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), # (b, 32, 26, 26) --> (b, 64, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (b, 64, 24, 24) --> (b, 64, 12, 12)
            nn.Dropout(.5),
            nn.Flatten() # (b, 64, 12, 12) --> (b, 64*12*12)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*12*12, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
            )

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out)
        return out