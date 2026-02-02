---
name: latex-compiler
description: 整合生成完整的LaTeX文档并编译。使用mcmthesis官方模板，支持XeLaTeX编译，自动处理图表和引用。
---

# LaTeX编译器 (LaTeX Compiler)

## 功能概述

将所有内容整合为完整的LaTeX文档并编译生成PDF。

## 模板配置

### mcmthesis模板

```latex
\documentclass[12pt]{mcmthesis}
\mcmsetup{
    tcn = {XXXXX},        % 控制号
    problem = {A},         % 选题
    sheet = true,          % 摘要页
    titlepage = false
}

\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}

\begin{document}

\begin{abstract}
% 摘要内容
\end{abstract}

\maketitle

% 正文内容

\bibliography{references}
\bibliographystyle{plain}

\end{document}
```

## 编译流程

```bash
# 编译流程
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

## 自动化脚本

```python
import subprocess

def compile_latex(main_file, output_dir='output'):
    commands = [
        ['xelatex', '-output-directory', output_dir, main_file],
        ['biber', f'{output_dir}/main'],
        ['xelatex', '-output-directory', output_dir, main_file],
        ['xelatex', '-output-directory', output_dir, main_file],
    ]
    
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            return False, result.stderr.decode()
    
    return True, None
```

## 输出格式

```json
{
  "compilation": {
    "success": true,
    "pdf_file": "output/papers/XXXXX.pdf",
    "pages": 24,
    "compile_time": 15.3,
    "warnings": [],
    "errors": []
  }
}
```

## 相关技能

- `compilation-error-handler` - 错误处理
- `citation-manager` - 引用管理
