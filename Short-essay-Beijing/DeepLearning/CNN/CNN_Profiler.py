import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

# Set English fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def setup_cudnn():
    """Configure cuDNN for performance optimization"""
    cudnn.enabled = True
    cudnn.benchmark = True
    print("\ncuDNN Configuration:")
    print(f"cuDNN available: {cudnn.is_available()}")
    print(f"cuDNN version: {cudnn.version()}")
    print(f"Benchmark mode: {cudnn.benchmark}")

# Custom dataset class
class TifDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read TIF image
        image = Image.open(self.image_paths[idx])
        # print(f"Image mode: {image.mode}, Image size: {image.size}")  # Debug info
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define improved CNN model
class CNNRegressor(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CNNRegressor, self).__init__()
        
        # Use improved convolutional layer architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.SiLU(inplace=True),  # Replace ReLU with SiLU (Swish) activation
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Add one layer
            nn.BatchNorm2d(256, momentum=0.9),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Calculate feature map size after convolution
        self.feature_size = 256 * 8 * 8  # 128x128 -> 8x8 after 4 downsampling operations
        
        # Optimized fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512, momentum=0.9),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128, momentum=0.9),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),  # Gradually reduce dropout
            
            nn.Linear(128, 1)
        )
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Use Kaiming initialization for SiLU activation function weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Data loading and preprocessing function
def load_data(data_dir):
    image_paths = []
    labels = []
    
    # Traverse data directory to get image paths and labels
    for filename in os.listdir(data_dir):
        if filename.endswith('.tif'):
            image_paths.append(os.path.join(data_dir, filename))
            
            # Extract example part from filename
            example = filename.split('_')[3]
            
            # Determine label type based on length
            if len(example) == 4:
                # Probably year
                label = float(example)
            elif len(example) == 6:
                # Probably month
                label = float(example)
            elif len(example) == 8:
                # Probably date
                label = float(example)
            else:
                raise ValueError(f"Unexpected format in filename: {filename}")
            
            labels.append(label)
    
    # Convert labels to numpy array and normalize
    labels = np.array(labels)
    labels = (labels - np.mean(labels)) / np.std(labels)
    
    return image_paths, labels.tolist()

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Initialize scaler for mixed precision training
    scaler = GradScaler('cuda')
    
    # Ensure model is in training mode
    model.train()
    print("\nConfirm model training mode:", model.training)
    
    # Print CUDA information
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.get_device_name(0)}")
        print(f"Memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    torch.cuda.synchronize()
    
    # Learning rate warmup
    warmup_epochs = 5
    
    # Select epoch to profile
    profile_epoch = 1
    
    # Before training loop starts
    start_time = time.time()
    # Record time for each batch
    batch_times = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Warmup GPU
        if epoch == 0:
            print("Warming up GPU...")
            warmup_tensor = torch.randn(32, 1, 128, 128).to(device)
            with torch.no_grad():
                for _ in range(10):
                    model(warmup_tensor)
            torch.cuda.synchronize()
        
        # Learning rate warmup
        if epoch < warmup_epochs:
            # Linear warmup
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
            print(f"Learning rate warmup: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Only use profiler on specified epoch
        if epoch == profile_epoch:
            # Configure profiler
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=2,  # Skip first 2 steps
                    warmup=2,  # Warmup 2 steps
                    active=6,  # Record 6 steps
                    repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_logs'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                # Training loop
                for batch_idx, (images, labels) in enumerate(train_loader):
                    batch_start = time.time()
                    
                    # Move data to GPU
                    images = images.to(device, non_blocking=True)
                    labels = labels.float().to(device, non_blocking=True)
                    
                    # Use mixed precision training
                    with autocast('cuda'):
                        with record_function("model_forward"):
                            outputs = model(images)
                        with record_function("loss_calculation"):
                            loss = criterion(outputs.squeeze(), labels)
                    
                    # Backward pass
                    with record_function("backward_pass"):
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    
                    # Record loss and update profiler
                    running_loss += loss.item()
                    prof.step()
                    
                    # Only analyze a few batches
                    if batch_idx >= 10:
                        break
                
                # Print analysis results
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                
                # Export Chrome trace format for visualization
                prof.export_chrome_trace("profile_trace.json")
        else:
            # Regular training code
            for batch_idx, (images, labels) in enumerate(train_loader):
                batch_start = time.time()
                
                # Move data to GPU
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                
                # Use mixed precision training
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs.squeeze(), labels)
                
                # Backward pass using scaler
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # Record loss
                running_loss += loss.item()
                
                # Calculate batch time
                torch.cuda.synchronize()
                batch_end = time.time()
                batch_times.append(batch_end - batch_start)
                
                if batch_idx % 5 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                          f'Loss: {loss.item():.4f} '
                          f'Batch Time: {batch_times[-1]:.3f}s '
                          f'GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB')
                
                # Regular cache cleanup
                if batch_idx % 30 == 0:
                    torch.cuda.empty_cache()
        
        # Calculate average training loss
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
        
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # Update learning rate (after warmup period)
        if epoch >= warmup_epochs:
            scheduler.step(epoch_val_loss)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_model.pth')
            print(f"Saved new best model, validation loss: {best_val_loss:.4f}")
        
        # Print epoch statistics
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(batch_times)
        
        print(f'\nEpoch {epoch+1} Statistics:')
        print(f'Training loss: {epoch_train_loss:.4f}')
        print(f'Validation loss: {epoch_val_loss:.4f}')
        print(f'Total time: {epoch_time:.2f}s')
        print(f'Average batch time: {avg_batch_time:.3f}s')
        print(f'Current learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB')
        print(f'GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f}MB')
        
        # Clean GPU cache
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
    
    # Post-training analysis
    print(f"Average batch time: {np.mean(batch_times):.4f}s")
    print(f"Longest batch time: {np.max(batch_times):.4f}s")
    print(f"Total training time: {time.time() - start_time:.4f}s")
    
    return train_losses, val_losses

def main():
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup cuDNN
    setup_cudnn()
    
    # Check if CUDA is available
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current CUDA device:", torch.cuda.get_device_name(0))
        print("CUDA device count:", torch.cuda.device_count())
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA support is required to run this program! Please ensure your GPU is properly configured.")
    
    # Set device
    device = torch.device('cuda')
    print("Using device:", device)
    
    # Enhanced data preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Validation set uses simpler transformations
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load data
    data_dir = r"F:\2000"
    print("Loading data from:", data_dir)
    image_paths, labels = load_data(data_dir)
    print("Number of images found:", len(image_paths))
    print("Label value range:", min(labels), "to", max(labels))
    
    if len(image_paths) == 0:
        raise RuntimeError("No .tif files found! Please check if the data directory path is correct.")
    
    # Split training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    print("Training set size:", len(train_paths))
    print("Validation set size:", len(val_paths))
    
    # Create data loaders
    train_dataset = TifDataset(train_paths, train_labels, transform=transform)
    val_dataset = TifDataset(val_paths, val_labels, transform=val_transform)
    
    # Modify data loader configuration - use better configuration
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
        drop_last=True  # Drop last incomplete batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True
    )
    
    print(f"\nData loader configuration:")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Batch size: 64")
    
    try:
        # Test data loader
        print("\nTesting data loading:")
        test_batch = next(iter(train_loader))
        print(f"Test batch shape: {test_batch[0].shape}")
        print(f"Test label shape: {test_batch[1].shape}")
        
        # Initialize improved model
        model = CNNRegressor(dropout_rate=0.5).to(device)
        print("\nModel structure:")
        print(model)
        
        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal model parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Use improved loss function - Huber loss combined with MSE
        criterion = nn.HuberLoss(delta=0.1)
        
        # Use AdamW optimizer - better weight decay handling
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=0.01,  # Lower initial learning rate
            weight_decay=1e-3,  # Increase weight decay
            betas=(0.9, 0.999),  # Use AdamW default parameters
            eps=1e-8  # Increase epsilon to prevent division by zero
        )
        
        # Save initial learning rate for warmup
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']
        
        # Improved learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, 
            min_lr=1e-6, verbose=True
        )
        
        # Train model
        num_epochs = 5  # Increase training epochs
        print("\nStarting training...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
        )
        
        # Plot loss curves
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.show()
        
        # Load best model for testing
        print("\nLoading best model for testing...")
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint['epoch']
        best_loss = checkpoint['val_loss']
        print(f"Best model from epoch {best_epoch+1}, validation loss: {best_loss:.4f}")
        
        # Final evaluation on validation set
        model.eval()
        val_loss = 0.0
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                
                # Collect predictions and true values
                predictions.extend(outputs.squeeze().cpu().numpy())
                true_values.extend(labels.cpu().numpy())
        
        final_val_loss = val_loss / len(val_loader)
        print(f"Final validation loss: {final_val_loss:.4f}")
        
        # Plot prediction vs true value scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(true_values, predictions, alpha=0.5)
        plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title('Predicted vs True Values')
        plt.grid(True)
        plt.savefig('predictions.png')
        plt.show()
        
    except Exception as e:
        print(f"\nError occurred during training:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise

if __name__ == '__main__':
    main()
