import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import seaborn as sns
from matplotlib.animation import FuncAnimation
from cartopy import crs as ccrs
import cartopy.feature as cfeature
import warnings
import time
warnings.filterwarnings('ignore')

# Set random seed to ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Custom dataset class
class PM25Dataset(Dataset):
    def __init__(self, data, seq_length, pred_length, transform=None):
        """
        Initialize PM2.5 dataset
        
        Args:
            data: PM2.5 data containing time and space dimensions, shape [time, latitude, longitude]
            seq_length: Historical sequence length for prediction
            pred_length: Future time step length for prediction
            transform: Data transformation/normalization function
        """
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.transform = transform
        self.len = len(data) - seq_length - pred_length + 1
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # Extract input sequence
        x = self.data[idx:idx+self.seq_length]
        # Extract target sequence
        y = self.data[idx+self.seq_length:idx+self.seq_length+self.pred_length]
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
            
        # Return tensor, x shape: [seq_length, height, width]
        # y shape: [pred_length, height, width]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# Define CNN-LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, seq_length, input_channels, hidden_dim, num_layers, output_length, 
                 kernel_size=3, padding=1):
        super(CNNLSTM, self).__init__()
        
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # CNN part - extract spatial features
        self.conv1 = nn.Conv2d(1, 16, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Calculate feature map size after CNN (assuming input is [batch, channels, height, width])
        # After 3 max pooling operations, size becomes 1/8 of original
        self.cnn_flat_dim = self._get_conv_output_size(input_channels)
        
        # LSTM part - extract temporal features
        self.lstm = nn.LSTM(
            input_size=self.cnn_flat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully connected layer - generate predictions
        self.fc = nn.Linear(hidden_dim, self.cnn_flat_dim)
        
        # Deconvolution part - restore features to original size
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(16, output_length, kernel_size=2, stride=2)
        
    def _get_conv_output_size(self, shape):
        # Helper function: calculate CNN output flattened size
        bs = 1
        x = torch.rand(bs, 1, *shape)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        return x.numel() // bs
        
    def forward(self, x):
        """
        Forward propagation
        
        Args:
            x: Input data, shape [batch_size, seq_length, height, width]
        
        Returns:
            Output prediction, shape [batch_size, pred_length, height, width]
        """
        batch_size, seq_len, height, width = x.size()
        
        # CNN processes each time step
        cnn_output = []
        for t in range(seq_len):
            # [batch_size, 1, height, width]
            xt = x[:, t, :, :].unsqueeze(1)
            
            # CNN forward pass
            xt = self.relu(self.conv1(xt))
            xt = self.pool(xt)
            xt = self.relu(self.conv2(xt))
            xt = self.pool(xt)
            xt = self.relu(self.conv3(xt))
            xt = self.pool(xt)
            
            # Flatten CNN output
            xt = xt.view(batch_size, -1)
            cnn_output.append(xt)
        
        # Stack CNN outputs from all time steps
        # [batch_size, seq_length, cnn_flat_dim]
        cnn_output = torch.stack(cnn_output, dim=1)
        
        # LSTM processes time series
        lstm_out, _ = self.lstm(cnn_output)
        # Only use output from last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layer
        fc_out = self.fc(lstm_out)
        
        # Reshape to convolutional feature map shape
        # Assuming after 3 pooling operations, feature map size is 1/8 of original
        small_h, small_w = height // 8, width // 8
        fc_out = fc_out.view(batch_size, 64, small_h, small_w)
        
        # Deconvolution to restore size
        out = self.relu(self.deconv1(fc_out))
        out = self.relu(self.deconv2(out))
        out = self.deconv3(out)
        
        return out

# Data loading and preprocessing function
def load_and_preprocess_data(data_path, start_year=2000, end_year=2023):
    """
    Load and preprocess PM2.5 data
    
    Args:
        data_path: Data file path
        start_year: Start year
        end_year: End year
    
    Returns:
        Preprocessed data, shape [time, latitude, longitude]
    """
    # In actual application, this needs to be adjusted based on your data format
    # Here assume data is stored in CSV files by year
    
    print(f"Loading PM2.5 data from {start_year} to {end_year}...")
    
    # Example code, please adjust according to actual data format
    data_frames = []
    for year in range(start_year, end_year + 1):
        try:
            file_path = os.path.join(data_path, f"pm25_{year}.csv")
            df = pd.read_csv(file_path)
            # Assume data contains date, longitude, latitude and PM2.5 values
            df['date'] = pd.to_datetime(df['date'])
            data_frames.append(df)
            print(f"Successfully loaded {year} data")
        except Exception as e:
            print(f"Unable to load {year} data: {e}")
    
    if not data_frames:
        raise ValueError("Failed to load any data")
    
    # Merge data from all years
    all_data = pd.concat(data_frames)
    
    # Assume we need to reshape data to [time, latitude, longitude] 3D grid
    # This part needs adjustment based on actual data structure
    print("Reorganizing data into 3D grid...")
    
    # Create time index
    time_index = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='D')
    
    # Get unique latitude and longitude values
    lats = sorted(all_data['latitude'].unique())
    lons = sorted(all_data['longitude'].unique())
    
    # Create an empty 3D array
    grid_data = np.zeros((len(time_index), len(lats), len(lons)))
    
    # Fill 3D grid
    for i, date in enumerate(time_index):
        day_data = all_data[all_data['date'].dt.date == date.date()]
        for _, row in day_data.iterrows():
            lat_idx = lats.index(row['latitude'])
            lon_idx = lons.index(row['longitude'])
            grid_data[i, lat_idx, lon_idx] = row['pm25']
    
    print(f"Data preprocessing complete. Final data shape: {grid_data.shape}")
    
    # Data normalization
    scaler = MinMaxScaler()
    grid_data_flat = grid_data.reshape(-1, grid_data.shape[1] * grid_data.shape[2])
    grid_data_scaled = scaler.fit_transform(grid_data_flat)
    grid_data = grid_data_scaled.reshape(grid_data.shape)
    
    return grid_data, time_index, lats, lons, scaler

# Training function
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """
    Train CNN-LSTM model
    
    Args:
        model: CNN-LSTM model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Training device (CPU/GPU)
    
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Ensure model is in training mode
    model.train()
    print("\nConfirm model training mode:", model.training)
    
    # Print CUDA information
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.get_device_name(0)}")
        print(f"Memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    torch.cuda.synchronize()  # Ensure GPU synchronization
    
    print("Starting model training...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        batch_times = []
        
        # Warmup GPU
        if epoch == 0:
            print("Warming up GPU...")
            warmup_tensor = torch.randn(32, train_loader.dataset[0][0].shape[0], 
                                      train_loader.dataset[0][0].shape[1], 
                                      train_loader.dataset[0][0].shape[2]).to(device)
            with torch.no_grad():
                for _ in range(10):
                    model(warmup_tensor)
            torch.cuda.synchronize()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient gradient clearing
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate batch time
            torch.cuda.synchronize()  # Ensure GPU operations complete
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            if batch_idx % 5 == 0:  # Print progress more frequently
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Batch Time: {batch_time:.3f}s '
                      f'GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB')
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_cnnlstm_model.pth')
        
        # Print epoch statistics
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        
        print(f'\nEpoch {epoch+1} Statistics:')
        print(f'Training loss: {train_loss:.4f}')
        print(f'Validation loss: {val_loss:.4f}')
        print(f'Total time: {epoch_time:.2f}s')
        print(f'Average batch time: {avg_batch_time:.3f}s')
        print(f'Current learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB')
        print(f'GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f}MB')
        
        # Clean GPU cache
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    return model, history

# Visualization function
def visualize_predictions(model, test_loader, time_index, lats, lons, scaler, device, output_dir='results'):
    """
    Visualize prediction results
    
    Args:
        model: Trained CNN-LSTM model
        test_loader: Test data loader
        time_index: Time index
        lats: List of latitude values
        lons: List of longitude values
        scaler: Scaler for denormalization
        device: Device (CPU/GPU)
        output_dir: Output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()
    
    # Select a sample for visualization
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Convert to NumPy arrays on CPU
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            
            # Denormalization
            def inverse_transform(data):
                shape = data.shape
                flat_data = data.reshape(-1, shape[-2] * shape[-1])
                flat_data = scaler.inverse_transform(flat_data)
                return flat_data.reshape(shape)
            
            inputs_np = inverse_transform(inputs_np)
            targets_np = inverse_transform(targets_np)
            outputs_np = inverse_transform(outputs_np)
            
            # Create custom colormap
            colors = [(0.0, 'green'), (0.3, 'yellow'), (0.6, 'orange'), (1.0, 'red')]
            cmap = LinearSegmentedColormap.from_list('pm25_cmap', colors)
            
            # Create visualization for each time step
            for i in range(outputs_np.shape[1]):  # For each predicted time step
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Plot last input time step
                ax = axes[0]
                ax.set_extent([min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS)
                
                lons_grid, lats_grid = np.meshgrid(lons, lats)
                cs = ax.pcolormesh(lons_grid, lats_grid, inputs_np[0, -1], 
                                  cmap=cmap, vmin=0, vmax=300, transform=ccrs.PlateCarree())
                ax.set_title(f'Last Input (Historical Data)')
                
                # Plot target
                ax = axes[1]
                ax.set_extent([min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS)
                
                cs = ax.pcolormesh(lons_grid, lats_grid, targets_np[0, i], 
                                  cmap=cmap, vmin=0, vmax=300, transform=ccrs.PlateCarree())
                ax.set_title(f'Actual Value (True Future)')
                
                # Plot prediction
                ax = axes[2]
                ax.set_extent([min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS)
                
                cs = ax.pcolormesh(lons_grid, lats_grid, outputs_np[0, i], 
                                  cmap=cmap, vmin=0, vmax=300, transform=ccrs.PlateCarree())
                ax.set_title(f'Predicted Value (Model Prediction)')
                
                # Add colorbar
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(cs, cax=cbar_ax)
                cbar.set_label('PM2.5 (μg/m³)')
                
                plt.suptitle(f'PM2.5 Prediction - Future Time Step {i+1}', fontsize=16)
                plt.tight_layout(rect=[0, 0, 0.9, 0.95])
                plt.savefig(os.path.join(output_dir, f'prediction_timestep_{i+1}.png'), dpi=300)
                plt.close()
            
            # Only process first batch for demonstration
            break
    
    # Create time series trend plot
    create_trend_analysis(inputs_np, targets_np, outputs_np, time_index, output_dir)
    
    # Create animation showing spatiotemporal changes
    create_spatiotemporal_animation(inputs_np, targets_np, outputs_np, lats, lons, output_dir)

# Create time trend analysis plot
def create_trend_analysis(inputs, targets, outputs, time_index, output_dir):
    """
    Create PM2.5 time trend analysis plot
    
    Args:
        inputs: Input data
        targets: Target data
        outputs: Prediction data
        time_index: Time index
        output_dir: Output directory
    """
    # Calculate average PM2.5 value over entire China region
    input_mean = np.mean(inputs, axis=(2, 3))  # Average all spatial points
    target_mean = np.mean(targets, axis=(2, 3))
    output_mean = np.mean(outputs, axis=(2, 3))
    
    # Plot time series trend
    plt.figure(figsize=(15, 8))
    
    # Get prediction length of last sample
    pred_length = outputs.shape[1]
    seq_length = inputs.shape[1]
    
    # Create time index corresponding to last input sequence
    end_idx = len(time_index) - pred_length - 1
    input_time = time_index[end_idx-seq_length+1:end_idx+1]
    future_time = time_index[end_idx+1:end_idx+1+pred_length]
    
    # Plot historical data and prediction data
    plt.plot(input_time, input_mean[0], 'b-', label='Historical Data')
    plt.plot(future_time, target_mean[0], 'g-', label='Actual Future Data')
    plt.plot(future_time, output_mean[0], 'r--', label='Model Prediction')
    
    plt.title('China PM2.5 Concentration Time Trend (2000-2023)', fontsize=16)
    plt.xlabel('Time')
    plt.ylabel('PM2.5 (μg/m³)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pm25_time_trend.png'), dpi=300)
    plt.close()
    
    # Create annual trend plot
    create_annual_trend(time_index, inputs, targets, outputs, output_dir)

# Create annual trend plot
def create_annual_trend(time_index, inputs, targets, outputs, output_dir):
    """
    Create annual PM2.5 change trend plot
    
    Args:
        time_index: Time index
        inputs: Input data
        targets: Target data
        outputs: Prediction data
        output_dir: Output directory
    """
    # Assume we have complete 2000-2023 historical data
    # Calculate annual average PM2.5
    
    # Simplified processing here, actual application needs adjustment based on real data
    years = range(2000, 2024)
    annual_pm25 = np.random.normal(50, 15, len(years))
    annual_pm25[10:] *= 0.8  # Assume decline after 2010
    annual_pm25[15:] *= 0.9  # Assume further decline after 2015
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(years, annual_pm25, alpha=0.7)
    
    # Add trend line
    z = np.polyfit(range(len(years)), annual_pm25, 1)
    p = np.poly1d(z)
    plt.plot(years, p(range(len(years))), "r--", linewidth=2)
    
    plt.title('China PM2.5 Annual Average Concentration Change (2000-2023)', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('PM2.5 (μg/m³)')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.xticks(years[::2], rotation=45)  # Display every 2 years
    
    # Add data labels
    for i, bar in enumerate(bars):
        if i % 2 == 0:  # Show value for every 2 bars
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                    f'{annual_pm25[i]:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pm25_annual_trend.png'), dpi=300)
    plt.close()

# Create spatiotemporal change animation
def create_spatiotemporal_animation(inputs, targets, outputs, lats, lons, output_dir):
    """
    Create PM2.5 spatiotemporal change animation
    
    Args:
        inputs: Input data
        targets: Target data
        outputs: Prediction data
        lats: List of latitude values
        lons: List of longitude values
        output_dir: Output directory
    """
    # Merge historical and prediction data
    seq_length = inputs.shape[1]
    pred_length = outputs.shape[1]
    
    # Select first batch sample
    combined_data = np.concatenate((inputs[0], outputs[0]), axis=0)
    
    # Create custom colormap
    colors = [(0.0, 'green'), (0.3, 'yellow'), (0.6, 'orange'), (1.0, 'red')]
    cmap = LinearSegmentedColormap.from_list('pm25_cmap', colors)
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        ax.set_extent([min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        
        lons_grid, lats_grid = np.meshgrid(lons, lats)
        cs = ax.pcolormesh(lons_grid, lats_grid, combined_data[frame], cmap=cmap, vmin=0, vmax=300, transform=ccrs.PlateCarree())
        
        if frame < seq_length:
            ax.set_title(f'Historical PM2.5 Data - Time Step {frame+1}/{seq_length}')
        else:
            pred_step = frame - seq_length + 1
            ax.set_title(f'Predicted PM2.5 Data - Future Time Step {pred_step}/{pred_length}')
        
        return [cs]
    
    ani = FuncAnimation(fig, update, frames=range(seq_length + pred_length),
                        blit=False, repeat=True)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax)
    cbar.set_label('PM2.5 (μg/m³)')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    ani.save(os.path.join(output_dir, 'pm25_spatiotemporal.gif'), writer='pillow', fps=2, dpi=150)
    plt.close()

# Main function
def main():
    # Set parameters
    data_path = 'data/pm25'  # Data path
    seq_length = 30  # Historical sequence length (e.g. 30 days)
    pred_length = 7  # Prediction sequence length (e.g. 7 days)
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001
    hidden_dim = 128
    num_layers = 2
    
    # Check if CUDA is available
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current CUDA device:", torch.cuda.get_device_name(0))
        print("CUDA device count:", torch.cuda.device_count())
        device = torch.device("cuda")
    else:
        print("Warning: CUDA device not detected, will use CPU for training (training will be very slow)")
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load and preprocess data
    try:
        data, time_index, lats, lons, scaler = load_and_preprocess_data(data_path)
        print(f"Data shape: {data.shape}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        # Generate simulated data for demonstration
        print("Generating simulated data for demonstration...")
        # Generate daily data from 2000 to 2023
        start_date = '2000-01-01'
        end_date = '2023-12-31'
        time_index = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Assume China's latitude and longitude range (simplified)
        lats = np.linspace(18, 54, 36)  # 18°N to 54°N
        lons = np.linspace(73, 135, 62)  # 73°E to 135°E
        
        # Create simulated PM2.5 data, shape [time, latitude, longitude]
        num_days = len(time_index)
        data = np.zeros((num_days, len(lats), len(lons)))
        
        # Add time trends (annual periodicity and long-term decline)
        for i in range(num_days):
            # Base value
            base = 50
            
            # Seasonal variation (high in winter, low in summer)
            day_of_year = time_index[i].dayofyear
            seasonal = 30 * np.sin(2 * np.pi * (day_of_year - 15) / 365)
            
            # Long-term trend (gradual decline after 2010)
            year = time_index[i].year
            trend = 0
            if year > 2010:
                trend = -10 * (year - 2010) / 13  # Annual decline after 2010
            
            # Northern regions have higher PM2.5 values
            for lat_idx, lat in enumerate(lats):
                for lon_idx, lon in enumerate(lons):
                    # Northern regions have higher PM2.5 in winter
                    north_factor = (lat - lats.min()) / (lats.max() - lats.min())
                    winter_boost = 0
                    if day_of_year < 80 or day_of_year > 330:  # Winter
                        winter_boost = 50 * north_factor
                    
                    # Eastern industrial areas have higher PM2.5
                    east_factor = (lon - lons.min()) / (lons.max() - lons.min())
                    industry_factor = east_factor * (1 - abs(lat - 35) / 20)
                    
                    # Combine all factors
                    pm25_value = base + seasonal + trend + winter_boost + 20 * industry_factor
                    
                    # Add some random noise
                    noise = np.random.normal(0, 5)
                    
                    # Ensure positive values
                    data[i, lat_idx, lon_idx] = max(0, pm25_value + noise)
        
        # Create scaler and normalize data
        scaler = MinMaxScaler()
        data_flat = data.reshape(-1, data.shape[1] * data.shape[2])
        data_scaled = scaler.fit_transform(data_flat)
        data = data_scaled.reshape(data.shape)
    
    # Create dataset and data loaders
    dataset = PM25Dataset(data, seq_length, pred_length)
    
    # Split into training, validation and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)  # Use batch_size=1 during testing for visualization
    
    # Create model
    input_channels = (len(lats), len(lons))
    model = CNNLSTM(seq_length, input_channels, hidden_dim, num_layers, pred_length)
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    model, history = train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)
    
    # Save model
    torch.save(model.state_dict(), 'pm25_cnnlstm_model.pth')
    
    # Visualize prediction results
    visualize_predictions(model, test_loader, time_index, lats, lons, scaler, device)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png', dpi=300)
    plt.close()
    
    print("Model training and evaluation complete!")

if __name__ == "__main__":
    main()
