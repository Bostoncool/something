import pypandoc
import os

# 若未检测到 pandoc，则自动下载
try:
    pypandoc.get_pandoc_path()
except OSError:
    pypandoc.download_pandoc()

# 将包含 LaTeX 的 Markdown 转为 Word
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'input.md')
output_path = os.path.join(script_dir, 'output.docx')
pypandoc.convert_file(input_path, 'docx', outputfile=output_path)