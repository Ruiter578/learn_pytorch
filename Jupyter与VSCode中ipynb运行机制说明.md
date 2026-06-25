# Jupyter 与 VS Code 中 ipynb 运行机制说明

本文以 `01-一、Pytorch的建模流程/1-1,结构化数据建模流程范例.ipynb` 为例，说明 `.ipynb` 文件、Jupyter、VS Code Jupyter 插件、Python kernel、pip 安装和依赖环境之间的关系。

## 1. `.ipynb` 文件是什么

`.ipynb` 是 Jupyter Notebook 文件，本质是一个 JSON 文档，不是普通 `.py` 脚本。

它通常包含：

- `cells`：单元格列表。
- `cell_type`：单元格类型，常见是 `markdown` 或 `code`。
- `source`：单元格内容。
- `outputs`：代码单元格上一次运行后的输出结果。
- `execution_count`：代码单元格上一次运行编号。
- `metadata`：notebook、kernel、语言、插件等元数据。

当前这个 notebook 的关键元数据是：

```json
{
  "kernelspec": {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3"
  },
  "language_info": {
    "name": "python",
    "version": "3.10.14"
  }
}
```

这表示它希望用一个名为 `python3` 的 Python kernel 运行，原作者当时的 Python 版本是 3.10.14。

## 2. Jupyter 的工作机制

Jupyter 不是单纯的文本编辑器，它由几层组成：

- Notebook 前端：负责显示 `.ipynb`、编辑单元格、显示图表和输出。
- Jupyter Server：负责管理 notebook 会话、文件、kernel 连接等。
- Kernel：真正执行代码的进程。Python notebook 通常使用 `ipykernel`。
- 通信协议：前端和 kernel 之间通过 Jupyter messaging protocol 通信。

运行一个代码单元格时，大致流程是：

1. 前端把当前代码单元格发送给 Jupyter Server 或 VS Code 的 notebook 控制层。
2. Server/控制层把代码发给对应 kernel。
3. Python kernel 在自己的 Python 进程里执行代码。
4. kernel 把 stdout、stderr、异常、图像、HTML、DataFrame 等结果发回前端。
5. 前端把结果写回界面，并可保存到 `.ipynb` 的 `outputs` 字段中。

所以 `.ipynb` 文件只是记录内容和结果；真正运行代码的是 kernel。

## 3. VS Code 是否必须安装 Jupyter 软件

通常不需要单独打开传统 Jupyter Notebook 网页版。

如果 VS Code 已安装 Python/Jupyter 相关插件，可以直接在 VS Code 中完成：

- 打开 `.ipynb`
- 选择 kernel
- 运行单元格
- 查看图表、DataFrame、Markdown、输出
- 保存 notebook

但这不等于完全不需要 Jupyter 生态组件。VS Code 仍然需要能启动一个 kernel。对 Python 来说，通常需要当前解释器环境中安装：

```bash
python -m pip install ipykernel
```

也就是说，VS Code 可以替代 Jupyter Notebook 的网页前端，但不能替代 Python kernel 和相关依赖。

## 4. VS Code 中运行 ipynb 是否依赖终端

不依赖“当前打开的终端”。

VS Code 运行 notebook 单元格时，通常不是把代码粘贴到你当前打开的 integrated terminal 里执行，而是：

- 由 VS Code Jupyter 插件启动或连接一个 kernel 进程。
- 代码在该 kernel 对应的 Python 解释器环境中运行。
- 输出回传到 notebook 单元格下方。

因此，关键不是当前终端是什么，而是 notebook 右上角选择的 kernel 是什么。

你可以在 VS Code notebook 右上角选择 kernel。常见选择包括：

- base conda 环境
- 某个 conda env
- 某个 venv
- 系统 Python
- 远程服务器上的 Python 环境

在远程 VS Code 场景中，kernel 运行在远程服务器上，不是在你的本地 Windows/WSL 机器上。

## 5. `!pip install torch` 到底在哪里执行

当前 notebook 第一格是：

```python
!pip install torch
!pip install 'torchkeras>=4.0.0'
```

`!` 是 IPython 的 shell escape。它会让 kernel 通过系统 shell 启动一个子进程执行命令。这个命令不会在你当前可见的 VS Code 终端面板中运行，而是在 kernel 进程派生的 shell 子进程中运行。

它使用的 `pip` 通常来自 kernel 所在环境的 `PATH`，但为了避免装错环境，更推荐写成：

```python
%pip install torch
%pip install "torchkeras>=4.0.0"
```

或者：

```python
import sys
!{sys.executable} -m pip install torch
!{sys.executable} -m pip install "torchkeras>=4.0.0"
```

这两种写法都比裸 `!pip install ...` 更稳，因为它们明确把包安装到当前 kernel 对应的 Python 环境中。

## 6. `import numpy/pandas/torch` 依赖什么

`import` 不依赖当前终端，而依赖当前 kernel 对应 Python 环境里是否安装了这些包。

例如 notebook 中有：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchkeras
```

这些 import 是否成功，取决于当前 kernel 环境里有没有：

- `numpy`
- `pandas`
- `matplotlib`
- `torch`
- `torchkeras`
- `tqdm`

当前服务器默认 `python3` 指向 `/home/linyichen/anaconda3/bin/python3`，但检查结果显示该环境目前没有：

- `ipykernel`
- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `torchkeras`

因此第一次在 VS Code 运行 notebook 前，应先选择或创建一个合适的 Python 环境，并在该环境中安装依赖。

## 7. 建议的环境准备方式

建议为这个教程单独建一个 conda 环境，不要污染 base：

```bash
conda create -n eat-pytorch python=3.10 -y
conda activate eat-pytorch
python -m pip install ipykernel numpy pandas matplotlib tqdm scikit-learn
python -m pip install torch torchkeras
python -m ipykernel install --user --name eat-pytorch --display-name "Python (eat-pytorch)"
```

然后在 VS Code 中打开 notebook，右上角选择：

```text
Python (eat-pytorch)
```

之后 notebook 中所有代码都会在这个 kernel 环境里运行。

如果你不想新建环境，也可以在当前环境里安装，但不推荐长期这样做。

## 8. 为什么 notebook 只安装 torch 和 torchkeras

这个 notebook 只显式安装：

```python
!pip install torch
!pip install 'torchkeras>=4.0.0'
```

原因可能是原作者假设运行环境已经内置了常见数据科学生态包，例如 `numpy`、`pandas`、`matplotlib`、`tqdm` 等。很多云 notebook、Anaconda 环境、Kaggle/Colab 环境默认已经带这些包。

此外，安装 `torch` 时，pip 下载的本来就是 PyPI 上发布的 wheel 包；用户通常不需要手动“安装轮子”。除非你在离线环境或特殊 CUDA 版本环境中，才需要手动指定 wheel URL 或下载 `.whl` 文件。

## 9. 是否需要补装 torchvision

对 `01-一、Pytorch的建模流程/1-1,结构化数据建模流程范例.ipynb` 来说，不需要。

这个 notebook 是结构化数据建模示例，主要用：

- pandas
- numpy
- matplotlib
- torch
- torchkeras
- tqdm

`torchvision` 主要用于图像任务，包括图像数据集、图像 transforms、预训练视觉模型等。它更可能在 `01-一、Pytorch的建模流程/1-2,图片数据建模流程范例.ipynb` 或图像相关章节中用到。

因此建议：

- 学 `1-1` 时不必补 `torchvision`。
- 学图片章节前再安装：

```bash
python -m pip install torchvision
```

如果你希望一次性准备完整教程环境，可以安装：

```bash
python -m pip install torch torchvision torchkeras numpy pandas matplotlib tqdm scikit-learn tensorboard
```

后续章节如果出现 `ModuleNotFoundError`，再按报错补装对应库。

## 10. VS Code 中判断当前 notebook 使用哪个 Python

在 notebook 新建一个单元格运行：

```python
import sys
print(sys.executable)
print(sys.version)
```

再运行：

```python
import site
print(site.getsitepackages())
```

这样可以确认当前 kernel 到底使用哪个 Python，以及包会安装到哪里。

安装包时，最稳写法是：

```python
%pip install package_name
```

安装后如果 VS Code 提示 restart kernel，应重启 kernel 再继续运行。

## 11. 推荐学习流程

1. 为教程创建独立 conda 环境。
2. 安装 `ipykernel` 和基础依赖。
3. 在 VS Code 打开 `.ipynb`。
4. 右上角选择正确 kernel。
5. 第一格安装命令建议改成 `%pip` 或在终端提前安装。
6. 每次遇到 `ModuleNotFoundError`，确认当前 kernel 后再安装缺失包。
7. 不要根据当前终端判断 notebook 环境，以 notebook 右上角 kernel 为准。

