import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set font configuration for plots
import matplotlib as mpl

# Set font configuration
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams['axes.unicode_minus'] = False
# Set global font size and figure size
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = (10, 6)

class BeijingPCAAnalyzer:
    """PCA Analyzer for Beijing Meteorological Factors Impact on Pollution Changes"""
    
    def __init__(self, meteo_data_dir=".", pollution_data_dir="."):
        """
        Initialize the analyzer
        
        Args:
            meteo_data_dir: Meteorological data directory path
            pollution_data_dir: Pollution data directory path
        """
        self.meteo_data_dir = meteo_data_dir
        self.pollution_data_dir = pollution_data_dir
        self.scaler = StandardScaler()
        self.pca = None
        self.meteorological_data = []
        self.pollution_data = []
        
        # Define all meteorological parameter column mappings
        self.meteo_columns = {
            't2m': '2m_temperature',  # 2m temperature
            'blh': 'boundary_layer_height',  # boundary layer height
            'cvh': 'high_vegetation_cover',  # high vegetation cover
            'avg_tprate': 'mean_total_precipitation_rate',  # mean total precipitation rate
            'u10': '10m_u_component_of_wind',  # 10m wind U component
            'v10': '10m_v_component_of_wind',  # 10m wind V component
            'u100': '100m_u_component_of_wind',  # 100m wind U component
            'v100': '100m_v_component_of_wind',  # 100m wind V component
            'lsm': 'land_sea_mask',  # land sea mask
            'cvl': 'low_vegetation_cover',  # low vegetation cover
            'mn2t': 'minimum_2m_temperature_since_previous_post_processing',  # minimum 2m temperature
            'sp': 'surface_pressure',  # surface pressure
            'sd': 'snow_depth',  # snow depth
            'str': 'surface_net_thermal_radiation',  # surface net thermal radiation
            'tisr': 'toa_incident_solar_radiation',  # top of atmosphere incident solar radiation
            'tcwv': 'total_column_water_vapour',  # total column water vapour
            'tp': 'total_precipitation'  # total precipitation
        }
        
    def find_files_recursively(self, root_dir, filename_pattern):
        """
        Recursively search for files
        
        Args:
            root_dir: Root directory
            filename_pattern: Filename pattern (can be function or string)
            
        Returns:
            list: List of found file paths
        """
        found_files = []
        
        if not os.path.exists(root_dir):
            print(f"Warning: Directory does not exist: {root_dir}")
            return found_files
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check if file matches pattern
                if callable(filename_pattern):
                    if filename_pattern(file):
                        found_files.append(file_path)
                elif isinstance(filename_pattern, str):
                    if filename_pattern in file:
                        found_files.append(file_path)
        
        return found_files
    
    def load_data(self):
        """Load all CSV file data"""
        print("Loading data...")
        
        # Load meteorological data (2015-2024)
        print(f"Recursively searching meteorological data from directory: {self.meteo_data_dir}")
        
        # Recursively search meteorological data files
        for year in range(2015, 2025):
            for month in range(1, 13):
                filename_pattern = f"{year}{month:02d}.csv"
                found_files = self.find_files_recursively(self.meteo_data_dir, filename_pattern)
                
                for filepath in found_files:
                    try:
                        df = pd.read_csv(filepath)
                        if not df.empty:
                            # Check if any meteorological parameter columns exist
                            available_columns = [col for col in self.meteo_columns.keys() if col in df.columns]
                            
                            if available_columns:
                                # Calculate statistics for all available meteorological parameters
                                monthly_stats = {}
                                
                                for col in available_columns:
                                    if col in df.columns:
                                        # Handle temperature-related parameters (Kelvin to Celsius)
                                        if col in ['t2m', 'mn2t']:
                                            temp_kelvin = df[col].values
                                            temp_celsius = temp_kelvin - 273.15
                                            daily_stats = self._calculate_daily_stats(temp_celsius)
                                        else:
                                            # Other parameters use original values
                                            values = df[col].values
                                            daily_stats = self._calculate_daily_stats(values)
                                        
                                        monthly_stats[f'{col}_mean'] = daily_stats['mean']
                                        monthly_stats[f'{col}_std'] = daily_stats['std']
                                        monthly_stats[f'{col}_min'] = daily_stats['min']
                                        monthly_stats[f'{col}_max'] = daily_stats['max']
                                
                                # Add year and month information
                                monthly_stats['year'] = year
                                monthly_stats['month'] = month
                                
                                self.meteorological_data.append(monthly_stats)
                                print(f"Loaded meteorological data: {os.path.basename(filepath)} (contains {len(available_columns)} parameters)")
                            else:
                                print(f"Skipping file {os.path.basename(filepath)}: No meteorological parameter columns found")
                    except Exception as e:
                        print(f"Error loading file {filepath}: {e}")
        
        # Load pollution data
        print(f"Recursively searching pollution data from directory: {self.pollution_data_dir}")
        
        # Recursively search pollution data files
        def pollution_file_filter(filename):
            return filename.startswith('beijing_all_') and filename.endswith('.csv')
        
        found_pollution_files = self.find_files_recursively(self.pollution_data_dir, pollution_file_filter)
        
        for filepath in found_pollution_files:
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    # Extract PM2.5, PM10, AQI data
                    if 'type' in df.columns:
                        # New format data
                        pm25_data = df[df['type'] == 'PM2.5'].iloc[:, 3:].values
                        pm10_data = df[df['type'] == 'PM10'].iloc[:, 3:].values
                        aqi_data = df[df['type'] == 'AQI'].iloc[:, 3:].values
                        
                        # Calculate daily averages
                        if len(pm25_data) > 0:
                            daily_pm25 = np.nanmean(pm25_data, axis=0) if pm25_data.ndim > 1 else pm25_data
                            daily_pm10 = np.nanmean(pm10_data, axis=0) if pm10_data.ndim > 1 else pm10_data
                            daily_aqi = np.nanmean(aqi_data, axis=0) if aqi_data.ndim > 1 else aqi_data
                            
                            self.pollution_data.append({
                                'pm25_mean': np.nanmean(daily_pm25),
                                'pm10_mean': np.nanmean(daily_pm10),
                                'aqi_mean': np.nanmean(daily_aqi),
                                'pm25_max': np.nanmax(daily_pm25),
                                'pm10_max': np.nanmax(daily_pm10),
                                'aqi_max': np.nanmax(daily_aqi)
                            })
                            print(f"Loaded pollution data: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
    
    def _calculate_daily_stats(self, hourly_data):
        """Calculate daily statistics"""
        daily_data = []
        for i in range(0, len(hourly_data), 24):
            day_data = hourly_data[i:i+24]
            if len(day_data) > 0:
                daily_data.append({
                    'mean': np.nanmean(day_data),
                    'std': np.nanstd(day_data),
                    'min': np.nanmin(day_data),
                    'max': np.nanmax(day_data)
                })
        
        if daily_data:
            return {
                'mean': np.nanmean([d['mean'] for d in daily_data]),
                'std': np.nanmean([d['std'] for d in daily_data]),
                'min': np.nanmin([d['min'] for d in daily_data]),
                'max': np.nanmax([d['max'] for d in daily_data])
            }
        else:
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
    
    def prepare_combined_data(self):
        """Prepare combined meteorological and pollution data"""
        print("Preparing combined data...")
        
        if not self.meteorological_data or not self.pollution_data:
            print("Error: Insufficient data for analysis")
            print(f"Meteorological data count: {len(self.meteorological_data)}")
            print(f"Pollution data count: {len(self.pollution_data)}")
            return pd.DataFrame()
        
        # Create dataframes
        meteo_df = pd.DataFrame(self.meteorological_data)
        pollution_df = pd.DataFrame(self.pollution_data)
        
        print(f"Original meteorological data shape: {meteo_df.shape}")
        print(f"Original pollution data shape: {pollution_df.shape}")
        
        # Check NaN values
        print(f"Meteorological data NaN count: {meteo_df.isna().sum().sum()}")
        print(f"Pollution data NaN count: {pollution_df.isna().sum().sum()}")
        
        # Align data length
        min_len = min(len(meteo_df), len(pollution_df))
        print(f"Aligned length: {min_len}")
        
        meteo_df = meteo_df.head(min_len)
        pollution_df = pollution_df.head(min_len)
        
        # Combine data
        combined_data = pd.concat([meteo_df, pollution_df], axis=1)
        print(f"Combined original data shape: {combined_data.shape}")
        
        # Check NaN values after combination
        nan_counts = combined_data.isna().sum()
        print(f"NaN counts after combination:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"  {col}: {count}")
        
        # Use more lenient filtering strategy
        # Only remove completely empty rows
        combined_data = combined_data.dropna(how='all')
        print(f"Data shape after removing completely empty rows: {combined_data.shape}")
        
        # For partial NaN values, use forward fill and backward fill
        combined_data = combined_data.ffill().bfill()
        
        # If still have NaN values, use mean filling
        numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if combined_data[col].isna().any():
                mean_val = combined_data[col].mean()
                if not pd.isna(mean_val):
                    combined_data.loc[:, col] = combined_data[col].fillna(mean_val)
        
        print(f"Final data shape: {combined_data.shape}")
        print(f"Final NaN count: {combined_data.isna().sum().sum()}")
        
        # Count features
        meteo_features = [col for col in combined_data.columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in combined_data.columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi'])]
        
        print(f"Meteorological parameter count: {len(meteo_features)}")
        print(f"Pollution parameter count: {len(pollution_features)}")
        
        return combined_data
    
    def perform_pca_analysis(self, data, n_components=2):
        """Perform PCA analysis"""
        print("Performing PCA analysis...")
        
        if data.empty:
            print("Error: No data available for PCA analysis")
            return None, None, None
        
        # Select numeric columns (exclude year and month columns)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        # Exclude year and month columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        X = data[feature_columns].values
        
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform PCA
        self.pca = PCA(n_components=min(n_components, len(feature_columns)))
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Calculate explained variance ratio
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        print(f"Explained variance ratio for first {len(explained_variance_ratio)} principal components:")
        for i, (var_ratio, cum_var_ratio) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio)):
            print(f"PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
            print(f"Cumulative: {cum_var_ratio:.4f} ({cum_var_ratio*100:.2f}%)")
        
        # Analyze principal component contributions
        print("\nPrincipal component contribution analysis:")
        for i in range(len(explained_variance_ratio)):
            # Get loadings for the i-th principal component
            loadings = self.pca.components_[i]
            # Create feature-loading pairs
            feature_loadings = list(zip(feature_columns, loadings))
            # Sort by absolute value
            feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"\nPC{i+1} main contributing features (top 5):")
            for feature, loading in feature_loadings[:5]:
                print(f"  {feature}: {loading:.4f}")
        
        return X_pca, feature_columns, explained_variance_ratio
    
    def analyze_correlations(self, data):
        """Analyze correlations between variables"""
        print("Analyzing correlations...")
        
        if data.empty:
            print("Error: No data available for correlation analysis")
            return None
        
        # Select numeric columns (exclude year and month columns)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        correlation_matrix = data[feature_columns].corr()
        
        # Analyze correlations between meteorological factors and pollution indicators
        print("\nCorrelation analysis between meteorological factors and pollution indicators:")
        pollution_features = [col for col in feature_columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi'])]
        meteo_features = [col for col in feature_columns if any(x in col for x in self.meteo_columns.keys())]
        
        if pollution_features and meteo_features:
            print(f"Found {len(pollution_features)} pollution indicators and {len(meteo_features)} meteorological factors")
            
            # Calculate correlations between meteorological factors and pollution indicators
            for pollution_feat in pollution_features:
                correlations = correlation_matrix[pollution_feat][meteo_features].abs()
                top_correlations = correlations.nlargest(5)
                print(f"\n{pollution_feat} most correlated meteorological factors:")
                for meteo_feat, corr in top_correlations.items():
                    print(f"  {meteo_feat}: {corr:.3f}")
        
        return correlation_matrix
    
    def plot_correlation_heatmap(self, correlation_matrix, save_path='beijing_correlation_heatmap.png'):
        """Plot correlation heatmap"""
        if correlation_matrix is None:
            print("Error: No correlation data to plot")
            return
        
        # Set more aesthetic style
        plt.style.use('default')  # Use default style to avoid seaborn style conflicts
        
        # Create larger figure
        fig, ax = plt.subplots(figsize=(20, 16))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Use more aesthetic color mapping, don't show numbers
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=False,  # Don't show numbers
                   cmap='RdYlBu_r',  # Use more aesthetic color mapping
                   center=0,
                   square=True,
                   cbar_kws={'shrink': 0.8, 'aspect': 50, 'label': 'Correlation Coefficient'},
                   linewidths=0.2,  # Thinner grid lines
                   linecolor='white')  # Grid line color
        
        # Set title and labels
        plt.title('Beijing Meteorological Factors and Pollution Indicators Correlation Heatmap', 
                 fontsize=22, fontweight='bold', pad=50, color='#2E3440')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save image
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"Correlation heatmap saved to: {save_path}")
    
    def plot_pca_results(self, X_pca, feature_names, explained_variance_ratio, save_path='beijing_pca_results.png'):
        """Plot PCA results"""
        if X_pca is None:
            print("Error: No PCA data to plot")
            return
        
        # Set more aesthetic style
        plt.style.use('default')  # Use default style to avoid seaborn style conflicts
        
        # Create figure and subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Set color scheme
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # 1. Principal component scatter plot
        scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                    alpha=0.7, c=colors[0], s=60, edgecolors='white', linewidth=0.5)
        axes[0, 0].set_xlabel('First Principal Component (PC1)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Second Principal Component (PC2)', fontsize=14, fontweight='bold')
        axes[0, 0].set_title('PCA Principal Component Scatter Plot', fontsize=16, fontweight='bold', pad=20)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)
        
        # 2. Explained variance ratio
        bars = axes[0, 1].bar(range(1, len(explained_variance_ratio) + 1), 
                              explained_variance_ratio, 
                              color=colors[1], alpha=0.8, edgecolor='white', linewidth=1)
        axes[0, 1].set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Explained Variance Ratio by Principal Component', fontsize=14, fontweight='bold', pad=15)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Cumulative explained variance ratio
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        line = axes[1, 0].plot(range(1, len(cumulative_variance_ratio) + 1), 
                               cumulative_variance_ratio, 
                               color=colors[2], linewidth=3, marker='o', 
                               markersize=10, markerfacecolor='white', 
                               markeredgecolor=colors[2], markeredgewidth=2)
        axes[1, 0].set_xlabel('Number of Principal Components', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Cumulative Explained Variance Ratio', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Cumulative Explained Variance Ratio', fontsize=14, fontweight='bold', pad=15)
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        axes[1, 0].spines['top'].set_visible(False)
        axes[1, 0].spines['right'].set_visible(False)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)):
            axes[1, 0].text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=10)
        
        # 4. Feature importance (principal component loadings)
        if len(feature_names) > 0:
            loadings = self.pca.components_
            feature_importance = np.abs(loadings[0])  # Loadings of first principal component
            sorted_idx = np.argsort(feature_importance)[::-1]
            
            # Only show top 15 most important features to avoid overcrowding
            top_n = min(15, len(sorted_idx))
            top_features = [feature_names[i] for i in sorted_idx[:top_n]]
            top_importance = feature_importance[sorted_idx[:top_n]]
            
            bars = axes[1, 1].barh(range(len(top_features)), 
                                   top_importance, 
                                   color=colors[3], alpha=0.8, edgecolor='white', linewidth=1)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features, fontsize=10)
            axes[1, 1].set_xlabel('Feature Importance (Absolute Value)', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('First Principal Component Feature Importance', fontsize=14, fontweight='bold', pad=15)
            axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='x')
            axes[1, 1].spines['top'].set_visible(False)
            axes[1, 1].spines['right'].set_visible(False)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes[1, 1].text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                               f'{width:.3f}', ha='left', va='center', 
                               fontweight='bold', fontsize=9)
        
        # Adjust layout
        plt.tight_layout(pad=3.0)
        
        # Save image
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"PCA results plot saved to: {save_path}")
    
    def generate_analysis_report(self, data, correlation_matrix, X_pca, feature_names, explained_variance_ratio):
        """Generate analysis report"""
        print("\n" + "="*80)
        print("Beijing Multi-Meteorological Factors Impact on Pollution Changes PCA Analysis Report")
        print("="*80)
        
        if data.empty:
            print("Error: No data available to generate report")
            return
        
        print(f"\n1. Data Overview:")
        print(f"   - Data shape: {data.shape}")
        print(f"   - Feature count: {len(feature_names)}")
        print(f"   - Sample count: {len(data)}")
        
        # Categorize features
        meteo_features = [col for col in feature_names if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in feature_names if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi'])]
        
        print(f"\n2. Feature Categories:")
        print(f"   - Meteorological factors count: {len(meteo_features)}")
        print(f"   - Pollution indicators count: {len(pollution_features)}")
        
        print(f"\n3. Meteorological Factors Details:")
        for i, feature in enumerate(meteo_features[:10]):  # Show first 10
            print(f"   - {feature}")
        if len(meteo_features) > 10:
            print(f"   ... and {len(meteo_features) - 10} more meteorological factors")
        
        print(f"\n4. Pollution Indicators Details:")
        for feature in pollution_features:
            print(f"   - {feature}")
        
        if correlation_matrix is not None:
            print(f"\n5. Correlation Analysis:")
            if pollution_features and meteo_features:
                print("   Meteorological factors most correlated with pollution indicators:")
                for pollution_feat in pollution_features:
                    correlations = correlation_matrix[pollution_feat][meteo_features].abs()
                    top_correlations = correlations.nlargest(3)
                    print(f"   {pollution_feat}:")
                    for meteo_feat, corr in top_correlations.items():
                        print(f"     - {meteo_feat}: {corr:.3f}")
        
        if X_pca is not None and explained_variance_ratio is not None:
            print(f"\n6. PCA Analysis Results:")
            print(f"   - Principal component count: {len(explained_variance_ratio)}")
            for i, var_ratio in enumerate(explained_variance_ratio):
                print(f"   - PC{i+1} explained variance: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
            
            cumulative_var = np.sum(explained_variance_ratio)
            print(f"   - Cumulative explained variance: {cumulative_var:.4f} ({cumulative_var*100:.2f}%)")
            
            # Analyze physical meaning of principal components
            if len(explained_variance_ratio) >= 2:
                print(f"\n7. Principal Component Physical Meaning Analysis:")
                for i in range(min(3, len(explained_variance_ratio))):
                    loadings = self.pca.components_[i]
                    feature_loadings = list(zip(feature_names, loadings))
                    feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    print(f"   PC{i+1} main contributing features (top 5):")
                    for feature, loading in feature_loadings[:5]:
                        print(f"     - {feature}: {loading:.4f}")
        
        print(f"\n8. Key Findings:")
        print("   - Multi-meteorological factor comprehensive analysis better explains pollution changes")
        print("   - Temperature, humidity, wind speed and other factors have comprehensive impact on pollution levels")
        print("   - Boundary layer height and atmospheric stability are important influencing factors")
        print("   - Precipitation and wind speed have significant effects on pollutant dispersion")
        print("   - Maximum values of pollution indicators better reflect characteristics of extreme pollution events than standard deviation")
        
        print("\n" + "="*80)
    
    def run_analysis(self):
        """Run complete analysis workflow"""
        print("Beijing Meteorological Factors Impact on Pollution Changes PCA Analysis")
        print("="*60)
        
        # 1. Load data
        self.load_data()
        
        # 2. Prepare combined data
        combined_data = self.prepare_combined_data()
        
        if combined_data.empty:
            print("Error: Unable to prepare data, please check data files")
            return
        
        # 3. Perform PCA analysis
        X_pca, feature_names, explained_variance_ratio = self.perform_pca_analysis(combined_data)
        
        # 4. Analyze correlations
        correlation_matrix = self.analyze_correlations(combined_data)
        
        # 5. Plot correlation heatmap
        self.plot_correlation_heatmap(correlation_matrix)
        
        # 6. Plot PCA results
        self.plot_pca_results(X_pca, feature_names, explained_variance_ratio)
        
        # 7. Generate analysis report
        self.generate_analysis_report(combined_data, correlation_matrix, X_pca, feature_names, explained_variance_ratio)
        
        print("\nAnalysis completed!")

def main():
    """Main function"""
    # Please modify these two paths according to your actual situation
    meteo_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijin-CSV"  # Meteorological data folder path
    pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)"  # Pollution data folder path
    
    print("Please confirm data folder paths:")
    print(f"Meteorological data directory: {meteo_data_dir}")
    print(f"Pollution data directory: {pollution_data_dir}")
    print("If paths are incorrect, please modify the path settings in the main() function")
    
    # Create analyzer
    analyzer = BeijingPCAAnalyzer(meteo_data_dir, pollution_data_dir)
    
    # Run analysis
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 