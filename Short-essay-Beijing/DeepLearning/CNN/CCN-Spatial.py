import torch
import torch.nn as nn
import torch.nn.functional as F

class PM25CNN(nn.Module):
    def __init__(self, in_channels=7, out_channels=1):
        super(PM25CNN, self).__init__()
        
        # Convolutional Feature Extractor
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # You could add pooling/upsampling as needed (like in U-Net) if you want context/large receptive field
        # For simplicity, we stay at the original resolution.

        # Output Convolution
        self.out_conv = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (batch_size, in_channels, H, W)
        x = self.conv_block1(x)   # (batch_size, 64, H, W)
        x = self.conv_block2(x)   # (batch_size, 128, H, W)
        x = self.conv_block3(x)   # (batch_size, 256, H, W)
        
        # Predict next day PM2.5
        x = self.out_conv(x)      # (batch_size, out_channels, H, W)
        return x

# Example usage
if __name__ == "__main__":
    # Suppose we stack 7 previous days => in_channels=7, predict 1 day => out_channels=1
    model = PM25CNN(in_channels=7, out_channels=1)
    dummy_input = torch.randn((1, 7, 256, 256))  # for example
    output = model(dummy_input)
    print(output.shape)  # (1, 1, 256, 256)
