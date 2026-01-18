import os
import pandas as pd
from pathlib import Path

def xls_to_csv_advanced(folder_path, output_folder=None, encoding='utf-8', recursive=False):
    """
    高级版本：支持子文件夹递归和多Sheet处理
    """
    source_dir = Path(r"C:\Users\IU\Desktop\大论文图\五大城市群的具体人口与占地面积")
    output_dir = Path(r"C:\Users\IU\Desktop\1") if output_folder else source_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    error_count = 0
    
    # 根据recursive参数决定是否递归
    pattern = "**/*.xlsx" if recursive else "*.xlsx"
    
    for xls_file in source_dir.glob(pattern):
        # 如果递归，保持相对目录结构
        relative_path = xls_file.relative_to(source_dir).parent
        current_output_dir = output_dir / relative_path
        current_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 首先尝试作为Excel文件读取
            excel_success = False

            # 尝试xlrd引擎（适用于旧版Excel文件）
            try:
                xls = pd.ExcelFile(xls_file, engine='xlrd')
                engine_used = 'xlrd'
                excel_success = True
            except Exception:
                # 如果xlrd失败，尝试openpyxl引擎（适用于新版.xlsx文件）
                try:
                    xls = pd.ExcelFile(xls_file, engine='openpyxl')
                    engine_used = 'openpyxl'
                    excel_success = True
                except Exception:
                    excel_success = False

            if excel_success:
                # 如果成功读取，则是真正的Excel文件
                if len(xls.sheet_names) == 1:
                    # 单Sheet情况
                    df = pd.read_excel(xls_file, engine=engine_used)
                    csv_path = current_output_dir / f"{xls_file.stem}.csv"
                    df.to_csv(csv_path, index=False, encoding=encoding)
                    print(f"[OK] Excel single sheet converted successfully")
                else:
                    # 多Sheet情况，每个Sheet单独保存
                    for sheet_name in xls.sheet_names:
                        df = pd.read_excel(xls_file, sheet_name=sheet_name, engine=engine_used)
                        safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in ('_','-'))
                        csv_path = current_output_dir / f"{xls_file.stem}_{safe_sheet_name}.csv"
                        df.to_csv(csv_path, index=False, encoding=encoding)
                    print(f"[OK] Excel multi-sheet converted successfully ({len(xls.sheet_names)} sheets)")

                converted_count += 1
                continue

            # 尝试作为CSV文件读取
            try:
                # 检查文件开头是否是UTF-8 BOM
                with open(xls_file, 'rb') as f:
                    first_bytes = f.read(3)
                    if first_bytes == b'\xef\xbb\xbf':
                        # 是UTF-8 BOM的CSV文件
                        df = pd.read_csv(xls_file, encoding='utf-8')
                    else:
                        # 尝试其他编码
                        encodings_to_try = ['gbk', 'utf-8', 'cp1252']
                        df = None
                        for enc in encodings_to_try:
                            try:
                                df = pd.read_csv(xls_file, encoding=enc)
                                break  # 如果成功读取，跳出循环
                            except (UnicodeDecodeError, pd.errors.ParserError):
                                continue  # 尝试下一个编码

                        if df is None:
                            raise ValueError("Could not decode file with any of the attempted encodings")

                csv_path = current_output_dir / f"{xls_file.stem}.csv"
                df.to_csv(csv_path, index=False, encoding=encoding)
                print(f"[OK] CSV file converted successfully")
                converted_count += 1

            except Exception as csv_error:
                print(f"[ERROR] Failed to convert CSV file: {str(csv_error)[:100]}...")
                error_count += 1

        except Exception as e:
            print(f"[ERROR] Unexpected error: {str(e)[:100]}...")
            error_count += 1
    
    print("\n" + "="*50)
    print(f"Conversion completed! Total: {converted_count} files successful, {error_count} files failed")
    print("="*50)

# 使用示例
if __name__ == "__main__":
    # 递归处理所有子文件夹
    xls_to_csv_advanced(".", encoding='utf-8', recursive=True)