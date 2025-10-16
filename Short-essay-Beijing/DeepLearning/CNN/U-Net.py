import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Set random seed to ensure reproducible results
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define TIF image dataset class
class TifDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        # Load TIF image using PIL
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        
        # Convert to numpy array for processing
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
        
        # Normalize image
        if image.ndim == 2:  # Grayscale image
            image = image[..., np.newaxis]
        image = image / 255.0
        
        # Process mask labels
        if mask.max() > 1:
            mask = mask / 255.0
        
        # Ensure mask is binary (if segmentation task)
        mask = (mask > 0.5).astype(np.float32)
        
        # Convert to PyTorch tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        mask = torch.from_numpy(mask).unsqueeze(0) if mask.ndim == 2 else torch.from_numpy(mask).permute(2, 0, 1)
        
        if self.transform:
            # Apply transformations
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask

# Double convolution block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

# U-Net model definition
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Downsampling part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Upsampling part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse list
        
        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            # Handle input size mismatch
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)

# Define Dice loss function (commonly used in segmentation tasks)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # Convert predictions to probabilities
        predictions = torch.sigmoid(predictions)
        
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection
        intersection = (predictions * targets).sum()
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

# Training function
def train_fn(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        
        # Forward pass
        predictions = model(data)
        loss = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item())
    
    return total_loss / len(train_loader)

# Validation function
def eval_fn(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            
            predictions = model(data)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

# Calculate IoU metric
def calculate_iou(pred, target):
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    
    intersection = (pred & target).float().sum((1, 2, 3))
    union = (pred | target).float().sum((1, 2, 3))
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def main():
    # Set hyperparameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    IN_CHANNELS = 3  # Adjust according to actual image channels
    OUT_CHANNELS = 1  # Binary segmentation
    
    # Data directories
    TRAIN_IMG_DIR = "path/to/train/images"
    TRAIN_MASK_DIR = "path/to/train/masks"
    VAL_IMG_DIR = "path/to/val/images"
    VAL_MASK_DIR = "path/to/val/masks"
    
    # Set random seed
    set_seed(42)
    
    # Create model
    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(device)
    
    # Define loss function and optimizer
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Define datasets and data loaders
    train_dataset = TifDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    val_dataset = TifDataset(VAL_IMG_DIR, VAL_MASK_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss = eval_fn(model, val_loader, criterion, device)
        
        # Print training information
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet_model.pth")
            print("Saving model checkpoint...")

# Prediction function
def predict(model, image_path, device):
    model.eval()
    
    # Load image
    image = Image.open(image_path)
    image = np.array(image, dtype=np.float32)
    
    # Normalize image
    if image.ndim == 2:  # Grayscale image
        image = image[..., np.newaxis]
    image = image / 255.0
    
    # Convert to PyTorch tensor
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    image = image.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(image)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()
    
    return prediction.squeeze().cpu().numpy()

if __name__ == "__main__":
    main()
