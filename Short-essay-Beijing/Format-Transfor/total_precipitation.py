import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NCtoCSVConverter:
    def __init__(self, input_folder, output_folder, num_processes=None, min_precipitation_mm=0.001):
        """
        初始化转换器
        
        Parameters:
        - input_folder: 输入NC文件文件夹路径
        - output_folder: 输出CSV文件文件夹路径
        - num_processes: 进程数，默认为CPU核心数
        - min_precipitation_mm: 最小降水阈值（毫米），小于此值的记录将被过滤掉
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_processes = num_processes or cpu_count()
        self.min_precipitation_mm = min_precipitation_mm
        
        # 创建输出文件夹
        os.makedirs(self.output_folder, exist_ok=True)
        
        # 关键变量映射（根据您的文件结构调整）
        self.key_variables = {
            'time': ['valid_time', 'time'],
            'latitude': ['latitude', 'lat'],
            'longitude': ['longitude', 'lon'],
            'precipitation': ['tp', 'precipitation', 'pr'],
            'temperature': ['t2m', 'temperature', 'tmp']  # 用于温度转换示例
        }
    
    def find_variable_name(self, ds, possible_names):
        """在数据集中查找变量名"""
        for name in possible_names:
            if name in ds.variables:
                return name
        return None
    
    def kelvin_to_celsius(self, temp_kelvin):
        """开尔文温度转摄氏度"""
        if temp_kelvin is None:
            return None
        return temp_kelvin - 273.15
    
    def process_time_dimension(self, time_data, time_units):
        """处理时间维度，转换为可读格式"""
        try:
            if 'since' in time_units:
                # 处理NetCDF标准时间格式
                time_df = pd.to_datetime(time_data, unit='s', origin='unix') \
                    if 'seconds' in time_units else pd.to_datetime(time_data)
            else:
                time_df = pd.to_datetime(time_data)
            return time_df
        except:
            # 如果转换失败，返回原始时间数据
            return time_data
    
    def process_single_file(self, nc_file_path):
        """处理单个NC文件"""
        try:
            print(f"正在处理文件: {os.path.basename(nc_file_path)}")
            
            # 读取NetCDF文件
            ds = xr.open_dataset(nc_file_path)
            
            # 查找关键变量
            time_var = self.find_variable_name(ds, self.key_variables['time'])
            lat_var = self.find_variable_name(ds, self.key_variables['latitude'])
            lon_var = self.find_variable_name(ds, self.key_variables['longitude'])
            precip_var = self.find_variable_name(ds, self.key_variables['precipitation'])
            temp_var = self.find_variable_name(ds, self.key_variables['temperature'])
            
            if not all([time_var, lat_var, lon_var]):
                print(f"警告: 文件 {nc_file_path} 缺少必要的维度变量")
                ds.close()
                return False
            
            # 提取维度数据
            time_data = ds[time_var].values
            lat_data = ds[lat_var].values
            lon_data = ds[lon_var].values
            
            # 处理时间维度
            time_units = getattr(ds[time_var], 'units', '')
            time_readable = self.process_time_dimension(time_data, time_units)
            
            # 准备数据列表
            data_records = []
            
            # 处理每个时间点的数据
            for t_idx, time_val in enumerate(time_readable):
                time_str = str(time_val)
                
                # 处理每个经纬度点
                for lat_idx, lat_val in enumerate(lat_data):
                    for lon_idx, lon_val in enumerate(lon_data):
                        record = {
                            'time': time_str,
                            'latitude': float(lat_val),
                            'longitude': float(lon_val)
                        }
                        
                        # 添加降水数据（支持含step维度，并转换单位为mm）
                        if precip_var and precip_var in ds.variables:
                            precip_da = ds[precip_var]
                            try:
                                sel_indexers = {}
                                # 按变量实际维度名进行安全索引
                                if time_var in precip_da.dims:
                                    sel_indexers[time_var] = t_idx
                                if lat_var in precip_da.dims:
                                    sel_indexers[lat_var] = lat_idx
                                if lon_var in precip_da.dims:
                                    sel_indexers[lon_var] = lon_idx

                                selected = precip_da.isel(**sel_indexers)
                                # 如存在额外维度（step/expver/number），进行汇总或选择
                                if 'step' in selected.dims:
                                    selected = selected.sum(dim='step', skipna=True)
                                if 'expver' in selected.dims:
                                    # ERA5 常见 expver=[1,5]，通常可取最大或求和；这里取和以保守整合
                                    selected = selected.sum(dim='expver', skipna=True)
                                if 'number' in selected.dims:
                                    # 若包含集合成员维度，取平均值更稳健
                                    selected = selected.mean(dim='number', skipna=True)

                                precip_val = float(selected.values)
                                # 处理缺失/填充值
                                fill_value = getattr(precip_da, '_FillValue', None)
                                if (isinstance(precip_val, float) and np.isnan(precip_val)) or (fill_value is not None and precip_val == fill_value):
                                    precip_val = np.nan

                                # 单位转换：m -> mm（ERA5的tp通常是以m表示的累计降水）
                                units = str(getattr(precip_da, 'units', '')).lower()
                                if 'm' in units and 'mm' not in units:
                                    # 转换为毫米并保持足够精度
                                    precip_mm = precip_val * 1000.0
                                    # 只有当值大于阈值时才记录
                                    if precip_mm >= self.min_precipitation_mm:
                                        record[f'{precip_var}_mm'] = round(float(precip_mm), 4)
                                    else:
                                        record[f'{precip_var}_mm'] = 0.0
                                else:
                                    # 保持原始单位，但确保精度
                                    if precip_val > 0:
                                        record[precip_var] = round(float(precip_val), 8)
                                    else:
                                        record[precip_var] = 0.0
                            except Exception as e:
                                print(f"    Warning: Error in advanced processing, using fallback: {e}")
                                # 回退到原始索引顺序假设 (time, lat, lon)
                                precip_data = precip_da.values
                                if precip_data.ndim >= 3:
                                    # 若包含step维度，尝试将其合并
                                    if precip_data.ndim == 4:
                                        # 假定顺序为 (time, step, lat, lon)
                                        precip_slice = precip_data[t_idx, :, lat_idx, lon_idx]
                                        precip_val = float(np.nansum(precip_slice))
                                    else:
                                        precip_val = float(precip_data[t_idx, lat_idx, lon_idx])
                                    
                                    # 应用相同的单位转换和精度控制
                                    units = str(getattr(precip_da, 'units', '')).lower()
                                    if 'm' in units and 'mm' not in units:
                                        precip_mm = precip_val * 1000.0
                                        if precip_mm > 0.001:
                                            record[f'{precip_var}_mm'] = round(float(precip_mm), 4)
                                        else:
                                            record[f'{precip_var}_mm'] = 0.0
                                    else:
                                        if precip_val > 0:
                                            record[precip_var] = round(float(precip_val), 8)
                                        else:
                                            record[precip_var] = 0.0
                        
                        # 添加温度数据并转换（如果存在）
                        if temp_var and temp_var in ds.variables:
                            temp_data = ds[temp_var].values
                            temp_units = getattr(ds[temp_var], 'units', '')
                            
                            if temp_data.ndim == 3:  # (time, lat, lon)
                                temp_val = temp_data[t_idx, lat_idx, lon_idx]
                                
                                # 处理缺失值
                                if hasattr(ds[temp_var], '_FillValue'):
                                    fill_value = ds[temp_var]._FillValue
                                    if np.ma.is_masked(temp_val) or temp_val == fill_value:
                                        temp_val = np.nan
                                
                                # 温度单位转换
                                if 'kelvin' in temp_units.lower() or 'K' in temp_units:
                                    temp_val = self.kelvin_to_celsius(temp_val)
                                    record[f'{temp_var}_celsius'] = float(temp_val)
                                else:
                                    record[temp_var] = float(temp_val)
                        
                        # 添加其他关键变量（根据您的需求扩展）
                        if 'number' in ds.variables:
                            record['ensemble_number'] = int(ds['number'].values)
                        
                        if 'expver' in ds.variables:
                            expver_data = ds['expver'].values
                            if len(expver_data) > t_idx:
                                record['expver'] = str(expver_data[t_idx])
                        
                        # 只有当记录包含有效降水数据时才添加
                        has_precipitation = False
                        if precip_var and precip_var in ds.variables:
                            # 检查是否有降水数据
                            precip_key = f'{precip_var}_mm' if f'{precip_var}_mm' in record else precip_var
                            if precip_key in record and record[precip_key] > 0:
                                has_precipitation = True
                        
                        # 如果设置了过滤阈值，只保留有降水的记录
                        if self.min_precipitation_mm > 0:
                            if has_precipitation:
                                data_records.append(record)
                        else:
                            data_records.append(record)
            
            # 创建DataFrame
            df = pd.DataFrame(data_records)
            
            # 关闭数据集
            ds.close()
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(nc_file_path))[0]
            output_file = os.path.join(self.output_folder, f"{base_name}.csv")
            
            # 保存为CSV
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            print(f"成功转换: {os.path.basename(nc_file_path)} -> {base_name}.csv")
            print(f"输出数据形状: {df.shape}")
            print(f"数据列: {list(df.columns)}")
            
            return True
            
        except Exception as e:
            print(f"处理文件 {nc_file_path} 时出错: {str(e)}")
            return False
    
    def batch_convert(self):
        """批量转换NC文件为CSV格式"""
        # 查找所有NC文件
        nc_pattern = os.path.join(self.input_folder, "*.nc")
        nc_files = glob.glob(nc_pattern)
        
        if not nc_files:
            print(f"在文件夹 {self.input_folder} 中未找到NC文件")
            return
        
        print(f"找到 {len(nc_files)} 个NC文件")
        print(f"使用 {self.num_processes} 个进程进行并行处理")
        
        # 使用多进程处理
        with Pool(processes=self.num_processes) as pool:
            results = pool.map(self.process_single_file, nc_files)
        
        # 统计结果
        successful = sum(results)
        failed = len(nc_files) - successful
        
        print(f"\n转换完成!")
        print(f"成功: {successful} 个文件")
        print(f"失败: {failed} 个文件")
        print(f"输出目录: {self.output_folder}")

def main():
    """主函数 - 在这里设置您的输入输出路径"""
    
    # ================================
    # 设置您的文件夹路径
    # ================================
    INPUT_FOLDER = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC\total_precipitation"    # 替换为您的NC文件输入路径
    OUTPUT_FOLDER = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\total_precipitation"  # 替换为您的CSV文件输出路径
    
    # 进程数 (None表示自动使用所有CPU核心)
    NUM_PROCESSES = None
    
    # 最小降水阈值（毫米），小于此值的记录将被过滤掉
    # 设置为0.001表示只保留大于0.001毫米的降水记录
    # 设置为0表示保留所有记录（包括0值）
    MIN_PRECIPITATION_MM = 0.001
    
    # ================================
    # 执行转换
    # ================================
    
    # 创建转换器实例
    converter = NCtoCSVConverter(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        num_processes=NUM_PROCESSES,
        min_precipitation_mm=MIN_PRECIPITATION_MM
    )
    
    # 执行批量转换
    converter.batch_convert()

if __name__ == "__main__":
    main()