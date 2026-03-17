"""
批量将文件夹内所有 TXT 文件转换为 Word 格式，支持 LaTeX 公式转换。
自动识别文档中可转为公式的部分（如 x^2、x_i、\\frac{}{}、希腊字母等），并包装为 LaTeX 格式。
"""
import pypandoc
import os
import re

# 若未检测到 pandoc，则自动下载
try:
    pypandoc.get_pandoc_path()
except OSError:
    pypandoc.download_pandoc()


def _fix_common_latex_typos(content: str) -> str:
    """
    修正公式中常见的 LaTeX 拼写错误（缺少反斜杠的命令）。
    仅在 $...$ 公式区域内替换，避免误改正文。
    """
    def fix_inside_math(match: re.Match) -> str:
        inner = match.group(1)
        # 常见需反斜杠的命令：bar, overline, hat, vec, dot, ddot, tilde 等
        fixes = [
            (r"(?<!\\)bar\b", r"\\bar"),      # bar{ -> \bar{
            (r"(?<!\\)overline\b", r"\\overline"),
            (r"(?<!\\)hat\b", r"\\hat"),
            (r"(?<!\\)vec\b", r"\\vec"),
            (r"(?<!\\)dot\b", r"\\dot"),
            (r"(?<!\\)ddot\b", r"\\ddot"),
            (r"(?<!\\)tilde\b", r"\\tilde"),
            (r"(?<!\\)widehat\b", r"\\widehat"),
            (r"(?<!\\)widetilde\b", r"\\widetilde"),
        ]
        for pattern, repl in fixes:
            inner = re.sub(pattern, repl, inner)
        return f"${inner}$"

    # 只对 $...$ 内的内容做修正（不处理 $$）
    return re.sub(r"(?<!\$)\$([^$]+)\$(?!\$)", fix_inside_math, content)


def auto_detect_and_wrap_math(content: str) -> str:
    """
    自动识别文档中可转为公式的部分，并包装为 $...$ 行内公式。
    识别模式包括：
    - LaTeX 命令：\\frac{}{}、\\sqrt{}、\\sum、\\int、\\alpha 等
    - 上下标：x^2、x_i、x^{n+1}、a_{ij}
    - Unicode 数学符号：α、β、θ、π、²、₃ 等
    - 常见数学运算符组合
    """
    # 简单 LaTeX 命令（无参数）
    simple_cmd = r"\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|sigma|phi|omega|infty|partial|nabla|times|div|pm|mp|leq|geq|neq|approx|equiv|in|forall|exists)"

    def wrap_match(m: re.Match) -> str:
        s = m.group(0).strip()
        if not s or s in ("$", "$$"):
            return m.group(0)
        # 避免重复包装
        if s.startswith("$") and s.endswith("$"):
            return m.group(0)
        if s.startswith("\\(") and s.endswith("\\)"):
            return m.group(0)
        if s.startswith("\\[") and s.endswith("\\]"):
            return m.group(0)
        return f"${s}$"

    result = content
    patterns = [
        # 先匹配 LaTeX 命令块（\frac{a}{b} 等）
        (r"\\frac\s*\{[^{}]*\}\s*\{[^{}]*\}", wrap_match),
        (r"\\sqrt\s*\{[^{}]*\}", wrap_match),
        (r"\\sum(?:\s*_\s*\{[^{}]*\})?(?:\s*\^\s*\{[^{}]*\})?", wrap_match),
        (r"\\int(?:\s*_\s*\{[^{}]*\})?(?:\s*\^\s*\{[^{}]*\})?", wrap_match),
        (r"\\prod(?:\s*_\s*\{[^{}]*\})?(?:\s*\^\s*\{[^{}]*\})?", wrap_match),
        (r"\\lim(?:\s*_\s*\{[^{}]*\})?", wrap_match),
        # 简单 LaTeX 命令
        (simple_cmd, wrap_match),
        # 上下标形式：变量^指数 或 变量_下标
        (r"[a-zA-Z]\s*_\s*\{[^{}]*\}(?:\s*[+\-*/=]?\s*[a-zA-Z0-9_\^\{\}\s+\-*/=])*", wrap_match),
        (r"[a-zA-Z]\s*\^\s*\{[^{}]*\}(?:\s*[+\-*/=]?\s*[a-zA-Z0-9_\^\{\}\s+\-*/=])*", wrap_match),
        (r"[a-zA-Z]\s*_\s*[a-zA-Z0-9]+(?:\s*[+\-*/=]?\s*[a-zA-Z0-9_\^\s+\-*/=])*", wrap_match),
        (r"[a-zA-Z]\s*\^\s*[a-zA-Z0-9+-]+(?:\s*[+\-*/=]?\s*[a-zA-Z0-9_\^\s+\-*/=])*", wrap_match),
        # Unicode 数学符号（至少2个连续或与字母数字混合）
        (r"(?:[a-zA-Z0-9]*[\u03B1-\u03C9\u0391-\u03A9\u00B2\u00B3\u2080-\u2089\u2070-\u2079][a-zA-Z0-9]*)(?:\s*[+\-*/=]\s*(?:[a-zA-Z0-9]*[\u03B1-\u03C9\u0391-\u03A9\u00B2\u00B3\u2080-\u2089\u2070-\u2079][a-zA-Z0-9]*))*", wrap_match),
    ]

    for pattern, repl in patterns:
        # 只包装非公式区域内的匹配：排除已在 $...$、\(...\)、\[...\] 内的内容
        def replacer(m, _res):
            full = m.group(0)
            start = m.start()
            prefix = _res[:start]
            # 已在 $...$ 内（奇数个单 $）
            if len(re.findall(r"(?<!\$)\$(?!\$)", prefix)) % 2 == 1:
                return full
            # 已在 \(...\) 或 \[...\] 内
            if prefix.count("\\(") > prefix.count("\\)") or prefix.count("\\[") > prefix.count("\\]"):
                return full
            return repl(m)

        result = re.sub(pattern, lambda m: replacer(m, result), result)

    return result


def preprocess_file_for_math(input_path: str) -> str:
    """读取文件并对其中的可识别公式进行自动包装，返回处理后的内容"""
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    content = auto_detect_and_wrap_math(content)
    content = _fix_common_latex_typos(content)
    return content


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

# 显式指定输入格式为 markdown，并启用 LaTeX 数学扩展
# tex_math_dollars: $...$ 行内公式, $$...$$ 块级公式
# tex_math_double_backslash: \[...\] \(...\) 公式
input_format = "markdown+tex_math_dollars+tex_math_double_backslash"
extra_args = ["--standalone"]

# 同时扫描 .txt 与 .md 文件
valid_exts = (".txt", ".md")
text_files = [
    f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)
]

if not text_files:
    print(f"在 {input_folder} 中未找到 .txt 或 .md 文件")
else:
    success, fail = 0, 0
    for text_file in text_files:
        input_path = os.path.join(input_folder, text_file)
        output_name = os.path.splitext(text_file)[0] + ".docx"
        output_path = os.path.join(out_dir, output_name)
        try:
            # 预处理：自动识别并包装可转为公式的部分
            preprocessed = preprocess_file_for_math(input_path)
            pypandoc.convert_text(
                preprocessed,
                "docx",
                format=input_format,
                outputfile=output_path,
                extra_args=extra_args,
            )
            print(f"✓ 已转换: {text_file} -> {output_name}")
            success += 1
        except Exception as e:
            print(f"✗ 转换失败 {text_file}: {e}")
            fail += 1
    print(f"\n完成: 成功 {success} 个, 失败 {fail} 个")