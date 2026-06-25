# PyTorch 项目学习顺序与 ipynb 使用指南

本文档基于 `/home/cyrus/workspace/learn_pytorch` 当前项目文件整理，目标是帮你知道：先学哪些文件、每个阶段怎么学、`.ipynb` 文件有什么特点，以及 notebook 里的代码到底在哪里运行。

## 1. 项目整体结构

这个项目是一个以 Jupyter Notebook 为主体的 PyTorch 教程仓库，内容来自《20 天吃掉那只 PyTorch》一类的章节式学习材料。

根目录主要包含几类文件：

- `README.md`：项目总说明，包含原作者给出的 20 天学习计划、扩展章节和依赖背景。
- `01-一、Pytorch的建模流程/00-一、Pytorch的建模流程.ipynb` 到 `07-七、Pytorch与广告推荐/00-七、Pytorch与广告推荐.ipynb`：章节导航 notebook，多数是目录型内容。
- `1-1,...ipynb` 到 `6-3,...ipynb`：PyTorch 入门到高阶 API 的主线学习内容。
- `7-1,...ipynb` 到 `7-8,...ipynb`：广告推荐方向的进阶内容，包括 FM、DeepFM、DIN、DIEN 等。
- `A-1,...ipynb` 到 `A-7,...ipynb`：周边工具章节，例如 Kaggle、Streamlit、Optuna、Gradio、wandb。
- `environment-eat-pytorch*.yml`、`requirements-eat-pytorch.txt`：环境复刻文件。
- `Jupyter与VSCode中ipynb运行机制说明.md`：已有的 notebook 运行机制说明。
- `迁移到另一台服务器与复刻环境流程.md`：服务器迁移和环境复刻步骤。

当前项目已经有 `AI_docs/` 目录，本文档就放在这个目录中。

## 2. 推荐学习顺序

建议不要一上来从张量 API 细节开始啃，而是先跑通一个完整建模流程。这样你会先建立“数据、模型、训练、评估、保存”的整体感，再回头理解底层概念。

### 阶段 0：环境和运行方式

先读：

1. `README.md`
2. `Jupyter与VSCode中ipynb运行机制说明.md`
3. `迁移到另一台服务器与复刻环境流程.md`

这个阶段的目标不是学 PyTorch 细节，而是确认你能打开 `.ipynb`、选择正确 kernel、运行一个代码单元格，并知道包装到哪里。

建议先在 notebook 里运行：

```python
import sys
print(sys.executable)
print(sys.version)

import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

当前本机命令行检查到的 Python 是：

```text
/home/cyrus/anaconda3/envs/eat-pytorch/bin/python
Python 3.10.20
```

当前可用 Jupyter kernels 包括：

```text
python3
eat-pytorch
```

打开 notebook 时，优先选择 `eat-pytorch` 这个 kernel。

### 阶段 1：先建立完整建模流程

推荐顺序：

1. `01-一、Pytorch的建模流程/00-一、Pytorch的建模流程.ipynb`
2. `01-一、Pytorch的建模流程/1-1,结构化数据建模流程范例.ipynb`
3. `01-一、Pytorch的建模流程/1-2,图片数据建模流程范例.ipynb`
4. `01-一、Pytorch的建模流程/1-3,文本数据建模流程范例.ipynb`
5. `01-一、Pytorch的建模流程/1-4,时间序列数据建模流程范例.ipynb`

这几个 notebook 的共同结构大致是：

1. 准备数据
2. 定义模型
3. 训练模型
4. 评估模型
5. 使用模型
6. 保存模型

学习方法：

- 第一遍只要求能从上到下跑通，不要在每个 API 上停太久。
- 每个 notebook 跑完后，用自己的话写 6 行总结：数据是什么、输入张量形状是什么、模型是什么、loss 是什么、optimizer 是什么、最后如何预测。
- 优先精读 `1-1`，因为结构化数据示例最容易看清完整训练套路。
- `1-2`、`1-3`、`1-4` 可以重点比较数据处理方式的变化，而不是重复纠结训练循环。

### 阶段 2：补 PyTorch 核心概念

推荐顺序：

1. `02-二、Pytorch的核心概念/00-二、Pytorch的核心概念.ipynb`
2. `02-二、Pytorch的核心概念/2-1,张量数据结构.ipynb`
3. `02-二、Pytorch的核心概念/2-2,自动微分机制.ipynb`
4. `02-二、Pytorch的核心概念/2-3,动态计算图.ipynb`

这一阶段要抓住三个关键词：

- Tensor：数据容器，类似更适合深度学习的多维数组。
- Autograd：自动求导机制，反向传播的基础。
- Dynamic graph：PyTorch 的计算图是边运行边构建的，所以它调试起来比较像普通 Python。

学习方法：

- 每看到一个 tensor 操作，都主动问自己：shape 是什么、dtype 是什么、device 在哪里。
- 学 `autograd` 时，重点理解 `requires_grad`、`grad`、`backward()`、计算图释放。
- 学动态计算图时，重点理解“Python 控制流可以直接参与模型计算”这件事。

### 阶段 3：理解 PyTorch 的三层 API

推荐顺序：

1. `03-三、Pytorch的层次结构/00-三、Pytorch的层次结构.ipynb`
2. `03-三、Pytorch的层次结构/3-1,低阶API示范.ipynb`
3. `03-三、Pytorch的层次结构/3-2,中阶API示范.ipynb`
4. `03-三、Pytorch的层次结构/3-3,高阶API示范.ipynb`

这一阶段建议建立一个分层地图：

- 低阶 API：直接操作 tensor、loss、梯度、参数更新。
- 中阶 API：使用 `nn.Module`、`Dataset`、`DataLoader` 等组织模型和数据。
- 高阶 API：借助 `torchkeras` 这类封装简化训练、评估、回调和可视化。

学习方法：

- 同一个任务分别看低阶、中阶、高阶写法，比较“哪些代码被框架封装掉了”。
- 不要只记高阶 API。真正排错时，仍然要能回到 tensor、loss、backward、optimizer 这些低层机制。

### 阶段 4：系统学习低阶 API

推荐顺序：

1. `04-四、Pytorch的低阶API/00-四、Pytorch的低阶API.ipynb`
2. `04-四、Pytorch的低阶API/4-1,张量的结构操作.ipynb`
3. `04-四、Pytorch的低阶API/4-2,张量的数学运算.ipynb`
4. `04-四、Pytorch的低阶API/4-3,nn.functional和nn.Module.ipynb`

这部分 API 多、细节多，适合当作手册反复查。

学习方法：

- 不建议一次性背完所有 tensor 操作。
- 建议按使用场景整理笔记：改 shape、拼接切分、索引选择、矩阵运算、统计运算、激活函数。
- `nn.functional` 和 `nn.Module` 要重点区分：前者更像函数，后者更像带参数和状态的层。

### 阶段 5：掌握中阶 API

推荐顺序：

1. `05-五、Pytorch的中阶API/00-五、Pytorch的中阶API.ipynb`
2. `05-五、Pytorch的中阶API/5-1,Dataset和DataLoader.ipynb`
3. `05-五、Pytorch的中阶API/5-2,模型层.ipynb`
4. `05-五、Pytorch的中阶API/5-3,损失函数.ipynb`
5. `05-五、Pytorch的中阶API/5-4,TensorBoard可视化.ipynb`

这是从“能跑 demo”走向“能组织项目”的关键阶段。

学习方法：

- `Dataset` 和 `DataLoader` 一定要动手改：改 batch size、shuffle、num_workers、返回字段。
- `模型层` 章节重点看常用层的输入输出形状。
- `损失函数` 不要只看名字，要知道它适合分类、回归、多标签还是排序任务。
- `TensorBoard` 重点学会记录 loss、metric、图像或模型结构。

### 阶段 6：学习高阶 API 和工程化训练

推荐顺序：

1. `06-六、Pytorch的高阶API/00-六、Pytorch的高阶API.ipynb`
2. `06-六、Pytorch的高阶API/6-1,构建模型的3种方法.ipynb`
3. `06-六、Pytorch的高阶API/6-2,训练模型的3种方法.ipynb`
4. `06-六、Pytorch的高阶API/6-3,使用GPU训练模型.ipynb`

这一阶段关注“如何更舒服地训练模型”。

学习方法：

- 对比 Sequential、继承 `nn.Module`、自定义复杂模型三种方式。
- 对比手写训练循环、封装训练函数、`torchkeras.KerasModel` 等训练方式。
- 学 GPU 时重点掌握 `device`、`.to(device)`、数据和模型必须在同一设备上。

### 阶段 7：广告推荐进阶

如果你的目标是推荐系统、广告点击率预估、排序模型，再学这一部分。

推荐顺序：

1. `07-七、Pytorch与广告推荐/00-七、Pytorch与广告推荐.ipynb`
2. `07-七、Pytorch与广告推荐/7-1,推荐算法业务.ipynb`
3. `07-七、Pytorch与广告推荐/7-2,广告算法业务.ipynb`
4. `07-七、Pytorch与广告推荐/7-3,FM模型.ipynb`
5. `07-七、Pytorch与广告推荐/7-4,DeepFM模型.ipynb`
6. `07-七、Pytorch与广告推荐/7-5,FiBiNET模型.ipynb`
7. `07-七、Pytorch与广告推荐/7-6,DeepCross模型.ipynb`
8. `07-七、Pytorch与广告推荐/7-7,DIN网络.ipynb`
9. `07-七、Pytorch与广告推荐/7-8,DIEN网络.ipynb`

学习方法：

- `7-1` 和 `7-2` 是业务背景，主要读懂推荐、广告、CTR、特征交叉、用户行为序列。
- `7-3` 到 `7-8` 建议按模型演进来学：FM 解决二阶交叉，DeepFM 结合深度网络，DIN/DIEN 引入用户兴趣和序列建模。
- 每学一个模型，画出：输入特征、embedding、特征交互层、MLP、输出、loss。

### 阶段 8：周边工具按需学习

推荐顺序取决于你的目标：

- 想找免费 GPU：`08-彩蛋：Pytorch周边工具/A-1,Kaggle免费GPU使用攻略.ipynb`
- 想做 Web demo：`08-彩蛋：Pytorch周边工具/A-2,Streamlit构建机器学习应用.ipynb`
- 使用 Mac M 系列芯片：`08-彩蛋：Pytorch周边工具/A-3,使用MacM1芯片加速pytorch.ipynb`
- 想自动调参：`08-彩蛋：Pytorch周边工具/A-4,optuna可视化调参魔法指南.ipynb`
- 想快速包装模型交互界面：`08-彩蛋：Pytorch周边工具/A-5,Gradio让你的机器学习模型性感起来.ipynb`
- 想做实验追踪和可视化：`08-彩蛋：Pytorch周边工具/A-6,30分钟吃掉wandb可视化模型分析.ipynb`
- 想做 wandb 自动调参：`08-彩蛋：Pytorch周边工具/A-7,30分钟吃掉wandb可视化自动调参.ipynb`

这些不是主线必修，建议在完成前 6 阶段后按需查阅。

## 3. 一套实用学习方法

### 第一遍：跑通

目标是让每个 notebook 至少完整运行一次。

做法：

- 从上到下运行全部单元格。
- 遇到 `ModuleNotFoundError`，先确认当前 kernel，再用 `%pip install 包名` 安装。
- 遇到数据文件缺失，先读该 notebook 的“准备数据”部分，看是否需要下载额外数据集。

不要在第一遍追求完全理解。先让代码活起来。

### 第二遍：拆解

目标是知道每段代码承担什么角色。

建议把每个完整案例拆成：

1. 数据读取与预处理
2. Dataset/DataLoader
3. 模型定义
4. loss 与 optimizer
5. 训练循环
6. 评估指标
7. 预测与保存

每个模块都用一句话解释它的输入和输出。

### 第三遍：改造

目标是从“看懂”变成“能用”。

可以做这些小改动：

- 改 batch size，观察训练速度和 loss 曲线。
- 改模型层数或隐藏单元数。
- 换 optimizer，例如从 SGD 换成 Adam。
- 增加一个 metric。
- 保存模型后重启 kernel，再加载模型做预测。
- 把 CPU 训练改成 GPU 训练。

### 第四遍：复现到自己的小任务

选一个最接近你目标的 notebook，把数据替换成自己的数据。

建议优先复制结构，而不是复制所有代码：

- 结构化表格任务：参考 `1-1`
- 图像任务：参考 `1-2`
- 文本任务：参考 `1-3`
- 时间序列任务：参考 `1-4`
- 推荐/广告任务：参考 `7-3` 以后章节

## 4. `.ipynb` 文件的特点

`.ipynb` 是 Jupyter Notebook 文件，本质上是一个 JSON 文档，不是普通 Python 脚本。

它通常包含：

- `cells`：单元格列表。
- `cell_type`：单元格类型，常见是 `markdown` 或 `code`。
- `source`：单元格里的源内容。
- `outputs`：代码单元格上一次运行后的输出。
- `execution_count`：代码单元格上一次运行编号。
- `metadata`：kernel、语言、插件等元数据。

本项目的 notebook 多数使用 Python 3.10 系列 kernel。当前抽查结果显示：

- `01-一、Pytorch的建模流程/1-1,结构化数据建模流程范例.ipynb`：kernel 显示名是 `eat-pytorch`，Python 版本是 `3.10.20`。
- 多数其他 notebook：kernel 显示名是 `Python 3 (ipykernel)`，Python 版本记录多为 `3.10.14`。

这说明文件里保存的是“建议使用的 kernel 信息”，但你实际运行时仍然可以在 VS Code 或 Jupyter 里重新选择当前机器上的 kernel。

### notebook 的优点

- 适合边写说明、边运行代码、边看图表。
- 每个代码块可以单独运行，方便实验和调试。
- 输出结果可以保存在文件里，别人打开时能看到上一次运行结果。
- 对教学、可视化、数据分析很友好。

### notebook 的风险

- 单元格可以不按顺序运行，导致“看起来代码在上面，实际变量来自之前某次运行”。
- kernel 里有状态，变量不会因为你编辑了上面的单元格就自动清空。
- 输出保存在 `.ipynb` 里，文件可能变大。
- `!pip install`、`!wget` 这类 shell 命令容易让环境和代码混在一起。

建议养成一个习惯：重要 notebook 改完后，用 `Restart Kernel and Run All` 从头跑一遍。

## 5. notebook 里的代码块到底在哪里运行

简短回答：代码不是在 `.ipynb` 文件里运行，也通常不是在你当前打开的 shell 里运行，而是在 notebook 选择的 Jupyter kernel 进程里运行。

更准确地说：

1. `.ipynb` 文件只负责保存文字、代码、输出和元数据。
2. VS Code/Jupyter 前端负责显示 notebook，并把代码发送出去。
3. Jupyter kernel 是真正执行 Python 代码的进程。
4. kernel 执行完后，把 stdout、stderr、异常、图片、HTML、DataFrame 等结果返回前端。
5. 前端把结果显示在单元格下方，并可以保存到 `.ipynb` 的 `outputs` 字段。

所以，关键不是“当前 shell 是什么”，而是 notebook 右上角选择了哪个 kernel。

### 它是在当前 shell 里面吗

通常不是。

在 VS Code 里打开一个 notebook 并运行代码时，VS Code 不会把代码粘贴到你当前打开的 integrated terminal 里执行。它会启动或连接一个 kernel 进程，然后把单元格代码发给这个 kernel。

当前 shell 的作用通常是：

- 你用它启动 `jupyter lab` 或 `jupyter notebook`。
- 你用它安装依赖、创建 conda 环境。
- 你用它检查文件、运行命令。

但 notebook 单元格本身运行在哪个 Python，取决于 kernel。

### 当前 shell 和 kernel 可能一样，也可能不一样

例如当前终端里：

```bash
which python
python --version
```

显示的是：

```text
/home/cyrus/anaconda3/envs/eat-pytorch/bin/python
Python 3.10.20
```

如果 notebook 右上角也选择了 `eat-pytorch` kernel，那么当前 shell 的 Python 和 notebook kernel 很可能来自同一个 conda 环境。

但如果 notebook 选择了别的 kernel，例如系统 Python、base conda、远程 kernel，那么它们就不是同一个运行环境。

### 如何在 notebook 里确认代码跑在哪里

在任意 notebook 新建一个单元格运行：

```python
import sys
import os

print("python:", sys.executable)
print("version:", sys.version)
print("cwd:", os.getcwd())
```

再检查 PyTorch：

```python
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

`sys.executable` 是最重要的，它告诉你当前 kernel 使用的 Python 解释器路径。

## 6. `!命令`、`%pip` 和 shell 的区别

在 notebook 代码单元格里，经常会看到：

```python
!pip install torch
!ls
!nvidia-smi
```

这里的 `!` 是 IPython 的 shell escape，意思是让当前 kernel 临时启动一个 shell 子进程去执行命令。

注意：这仍然不是“你当前可见的终端面板”。它是 kernel 派生出来的子进程，输出会回到 notebook 单元格下面。

安装包时更推荐：

```python
%pip install package_name
```

或者：

```python
import sys
!{sys.executable} -m pip install package_name
```

原因是这两种写法更能保证包装进当前 notebook kernel 对应的 Python 环境里，避免 `pip` 指向另一个环境。

## 7. 单元格运行顺序和状态

notebook 的每个代码单元格共享同一个 kernel 状态。

例如你先运行：

```python
x = 10
```

再运行另一个单元格：

```python
print(x)
```

第二个单元格能打印 `10`，因为 `x` 保存在 kernel 内存里。

如果你删除第一个单元格，但没有重启 kernel，`x` 仍然可能存在。这也是 notebook 最容易让初学者困惑的地方。

建议：

- 学习时可以逐格运行，方便观察。
- 交作业、提交代码、保存重要实验前，执行 `Restart Kernel and Run All`。
- 如果出现“不知道变量从哪来”的情况，先重启 kernel。

## 8. 工作目录和文件路径

notebook 代码里的相对路径通常相对于当前工作目录 `os.getcwd()`。

在 VS Code 中打开本项目 notebook 时，工作目录通常是项目根目录：

```text
/home/cyrus/workspace/learn_pytorch
```

但这不是绝对保证。可以用下面代码检查：

```python
import os
print(os.getcwd())
print(os.listdir(".")[:10])
```

如果数据读取报错，先确认：

- 当前工作目录是不是项目根目录。
- 数据文件是否已经下载。
- notebook 中写的相对路径是否和你的目录一致。

## 9. 环境和依赖建议

当前项目已经提供了：

- `environment-eat-pytorch-portable.yml`
- `requirements-eat-pytorch.txt`
- `迁移到另一台服务器与复刻环境流程.md`

如果重新创建环境，推荐优先看迁移文档里的“稳妥复刻方案”。核心思路是：

```bash
conda create -n eat-pytorch python=3.10 -y
conda activate eat-pytorch
python -m pip install ipykernel numpy pandas matplotlib tqdm scikit-learn seaborn pillow tensorboard jupyterlab notebook
python -m pip install torch torchvision torchaudio
python -m pip install "torchkeras>=4.0.0" tabm
python -m ipykernel install --user --name eat-pytorch --display-name "Python (eat-pytorch)"
```

如果需要指定 CUDA 版本，应使用 PyTorch 官方对应 wheel 源。例如迁移文档里给出了 CUDA 12.8、12.6、11.8 的安装思路。

在 notebook 内安装依赖时，优先用：

```python
%pip install package_name
```

安装完成后，如果 VS Code 提示重启 kernel，就重启再继续。

## 10. 常见问题排查

### import 失败

现象：

```text
ModuleNotFoundError: No module named 'xxx'
```

处理：

1. 在 notebook 里运行 `import sys; print(sys.executable)`。
2. 确认右上角 kernel 是否是 `eat-pytorch`。
3. 用 `%pip install xxx` 安装到当前 kernel。
4. 重启 kernel 后再运行。

### GPU 不可用

现象：

```python
torch.cuda.is_available()
```

返回 `False`。

处理：

- 先在 shell 运行 `nvidia-smi`，确认机器有没有 NVIDIA GPU 和驱动。
- 确认安装的是支持 CUDA 的 PyTorch，而不是 CPU-only 版本。
- 确认远程 VS Code 连接的是带 GPU 的服务器。

### 变量明明没定义却能用

原因通常是 kernel 里残留了之前运行过的变量。

处理：

- `Restart Kernel`
- 再 `Run All`

### pip 安装了但 import 还是失败

通常是 pip 装到了另一个 Python 环境。

处理：

```python
import sys
print(sys.executable)
%pip install package_name
```

不要只在外部 shell 里盲目 `pip install`。

### 相对路径找不到文件

处理：

```python
import os
print(os.getcwd())
```

确认当前工作目录后，再修正相对路径或切换工作目录。

## 11. 建议的笔记模板

每学完一个 notebook，可以按这个模板写 5 到 10 行笔记：

```text
文件：
主题：
输入数据：
核心模型：
关键 API：
训练方式：
评估指标：
我改动了什么：
遇到的问题：
下一步：
```

这样学完整个项目后，你得到的不是一堆跑过的 notebook，而是一套可以迁移到自己项目里的 PyTorch 方法库。

## 12. 最推荐的最短路径

如果你的时间有限，先学这一条路径：

1. `01-一、Pytorch的建模流程/1-1,结构化数据建模流程范例.ipynb`
2. `02-二、Pytorch的核心概念/2-1,张量数据结构.ipynb`
3. `02-二、Pytorch的核心概念/2-2,自动微分机制.ipynb`
4. `04-四、Pytorch的低阶API/4-3,nn.functional和nn.Module.ipynb`
5. `05-五、Pytorch的中阶API/5-1,Dataset和DataLoader.ipynb`
6. `06-六、Pytorch的高阶API/6-2,训练模型的3种方法.ipynb`
7. `06-六、Pytorch的高阶API/6-3,使用GPU训练模型.ipynb`

这条路径能最快建立 PyTorch 的主干能力：张量、求导、模型、数据管道、训练循环、GPU。

然后再按你的方向补：

- 做视觉：补 `1-2`
- 做 NLP：补 `1-3`
- 做时序：补 `1-4`
- 做推荐广告：补第 7 章
- 做实验管理和展示：补 A 系列工具章节

