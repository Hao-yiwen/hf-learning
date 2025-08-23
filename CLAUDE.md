# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 HuggingFace 学习项目，用于练习和实验 Transformers、Diffusers 等 HuggingFace 生态系统的功能。项目按月份组织学习内容，每个月份文件夹包含相关的 Jupyter Notebook 和 Python 脚本。

## 代码组织结构

- **按月份组织**: 代码存放在 `YYYYMM/` 格式的文件夹中（如 `202508/`）
- **文件类型**: 
  - `.ipynb` - Jupyter Notebook 实验代码
  - `.py` - Python 测试脚本

## 常用开发命令

### Jupyter Notebook 操作
```bash
# 启动 Jupyter Notebook
jupyter notebook

# 启动 JupyterLab（如果已安装）
jupyter lab
```

### 环境检查
```bash
# 检查 Python 环境和包版本
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Git 提交规范
提交信息使用中文，格式为 `<类型>: <描述>`
- 类型包括：新增、修复、优化、更新、文档、测试、重构、样式

## 技术栈和依赖

主要使用的库：
- **transformers**: HuggingFace 核心库，用于各种 NLP 任务
- **torch**: PyTorch 深度学习框架
- **diffusers**: 用于扩散模型（如文本生成图像）
- **accelerate**: 加速模型加载和训练

## 开发注意事项

1. **GPU 支持**: 代码通常检查 CUDA 可用性并自动选择设备（cuda/cpu）
2. **模型加载**: 使用 pipeline API 时注意指定模型名称，避免使用默认模型
3. **环境问题**: 项目中存在一些 PyTorch 环境配置问题（torch.hub 模块缺失），需要重新配置环境
4. **实验性代码**: Notebook 中的代码是实验性的，可能包含失败的尝试和错误输出

## 当前已知问题

1. **PyTorch 环境问题**: 
   - `torch.hub` 模块缺失
   - torchvision 导入失败
   - 建议重新安装 PyTorch 和相关依赖

2. **任务支持**:
   - `text-to-image` 不是 transformers pipeline 的标准任务
   - 图像生成应使用 diffusers 库的 DiffusionPipeline

## 修复环境建议

如遇到环境问题，建议运行：
```bash
# 重新安装 PyTorch（根据 CUDA 版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install transformers diffusers accelerate
```