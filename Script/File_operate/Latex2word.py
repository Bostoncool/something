"""
批量将文件夹内所有 TXT 文件转换为 Word 格式，支持 LaTeX 公式转换。
"""
import pypandoc
import os

# 若未检测到 pandoc，则自动下载
try:
    pypandoc.get_pandoc_path()
except OSError:
    pypandoc.download_pandoc()

# ============ 可配置参数 ============
# 输入文件夹路径（默认使用脚本所在目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = r"C:\Users\IU\Desktop\New folder"

# 输出文件夹（None 表示与输入同目录；也可指定如 'output_docx' 子文件夹）
output_folder = "output_docx"

# ============ 批量转换逻辑 ============
if output_folder is not None:
    out_dir = os.path.join(input_folder, output_folder)
    os.makedirs(out_dir, exist_ok=True)
else:
    out_dir = input_folder

# 显式指定输入格式为 markdown（.txt 扩展名会被 pandoc 误判，需强制指定）
# tex_math_dollars: $...$ 行内公式, $$...$$ 块级公式
# tex_math_double_backslash: \[...\] \(...\) 公式
input_format = 'markdown+tex_math_dollars+tex_math_double_backslash'
extra_args = ['--standalone']

txt_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.txt')]

if not txt_files:
    print(f'在 {input_folder} 中未找到 .txt 文件')
else:
    success, fail = 0, 0
    for txt_file in txt_files:
        input_path = os.path.join(input_folder, txt_file)
        output_name = os.path.splitext(txt_file)[0] + '.docx'
        output_path = os.path.join(out_dir, output_name)
        try:
            pypandoc.convert_file(
                input_path,
                'docx',
                format=input_format,
                outputfile=output_path,
                extra_args=extra_args,
            )
            print(f'✓ 已转换: {txt_file} -> {output_name}')
            success += 1
        except Exception as e:
            print(f'✗ 转换失败 {txt_file}: {e}')
            fail += 1
    print(f'\n完成: 成功 {success} 个, 失败 {fail} 个')