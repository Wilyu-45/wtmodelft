# Writing Skill - 个人写作风格模型微调项目

基于 LoRA 技术微调大语言模型，让模型学习并模仿用户个人的写作风格。

## 项目简介

本项目使用 QLoRA（4-bit 量化 + LoRA）技术对基础大语言模型进行微调，使其能够模仿特定作者的写作风格、用词习惯和句式结构。

## 功能特性

- **数据处理**：自动读取文本文件，清理、切片、生成训练数据
- **模型训练**：基于 Unsloth + LoRA 进行高效微调，支持 4-bit 量化
- **推理服务**：提供 Gradio Web UI 界面，便于交互测试
- **工具集**：
  - `split_txt.py` - txt 文件分割与批量导出工具
  - `process_data.py` - 训练数据预处理
  - `train_model.py` - 模型训练脚本
  - `inference.py` / `inference_ui.py` - 模型推理与界面

## 项目结构

```
writingskill/
├── data/                       # 数据目录
│   ├── raw/                    # 原始文本数据（.gitignore）
│   └── processed/              # 处理后的训练/测试数据
│       ├── training_data.jsonl
│       └── test_data.jsonl
├── models/                     # 模型目录
│   ├── base_model_huff/        # 基础模型（.gitignore 大文件）
│   ├── my_style_model/         # 合并后的最终模型（.gitignore）
│   └── lora_adapter/           # LoRA 权重（.gitignore）
├── output/                     # 训练输出
│   └── checkpoint-*/           # 训练检查点（.gitignore）
├── scripts/                    # 脚本目录
│   ├── config.py               # 全局配置
│   ├── check_env.py            # 环境检查
│   ├── process_data.py         # 数据处理
│   ├── train_model.py          # 模型训练
│   ├── inference.py            # 模型推理
│   ├── inference_ui.py         # Gradio UI
│   └── split_txt.py            # txt 分割/批量导出工具
├── torch-gpu/                  # Python 虚拟环境（.gitignore）
├── requirements.txt            # Python 依赖
└── .gitignore
```

## 环境要求

- Python 3.10+
- CUDA 支持的 GPU（建议 8GB+ 显存）
- Windows / Linux

## 安装

```bash
# 克隆仓库
git clone https://github.com/Wilyu-45/wtmodelft.git
cd wtmodelft

# 创建虚拟环境（可选）
python -m venv torch-gpu
torch-gpu\Scripts\activate   # Windows
# source torch-gpu/bin/activate  # Linux

# 安装依赖
pip install -r requirements.txt
```

## 使用流程

### 1. 准备原始数据

将待学习的文本文件（.txt 格式）放入 `data/raw/` 目录。

### 2. 准备基础模型

将基础模型放入 `models/base_model_huff/` 目录，或在 `scripts/config.py` 中修改 `base_model_name` 指向其他模型。

> 注：基础模型文件较大（>100MB），未包含在本仓库中，需要单独下载。

### 3. 数据预处理

```bash
python scripts/process_data.py
```

会生成 `data/processed/training_data.jsonl` 和 `test_data.jsonl`。

### 4. 模型训练

```bash
python scripts/train_model.py
```

训练完成后会保存 LoRA 权重到 `output/checkpoint-*/`。

### 5. 启动推理 UI

```bash
python scripts/inference_ui.py
```

打开浏览器访问 Gradio 提供的本地地址即可与模型交互。

## 工具脚本说明

### `scripts/split_txt.py`

GUI 工具，集成三大功能：

1. **txt 分割**：选择文本文件，手动添加行范围，按指定文件名输出到 `input/` 目录
2. **批量导出 audiotemp.list**：解析 `output_path|folder|language|text` 格式的列表文件，将 text 内容输出为 txt 文件
3. **处理 input 纯文本**：将 `input/` 目录下的 ASS/SSA 字幕文件转换为纯文本（去除 `{\blur3}` 等标签）

## 配置说明

主要配置在 `scripts/config.py`：

| 配置项 | 说明 |
|--------|------|
| `MODEL_CONFIG` | 基础模型路径与序列长度 |
| `LORA_CONFIG` | LoRA rank、alpha、dropout 等 |
| `TRAINING_CONFIG` | 训练超参数（学习率、batch size 等） |
| `DATA_CONFIG` | 数据处理参数（切片大小、重叠等） |
| `INFERENCE_CONFIG` | 推理参数（temperature、top_p 等） |
| `DATA_FORMAT` | 数据格式：alpaca / chatml / plain |
| `SYSTEM_PROMPT` | 系统提示词 |

## 训练数据格式

项目使用 ChatML 格式：

```json
{"messages": [
  {"role": "system", "content": "你是一个模仿用户写作风格的助手。..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

## 注意事项

- 基础模型与训练输出未上传到 GitHub（单文件 >100MB）
- 训练需要 GPU 资源，CPU 训练会非常慢
- 建议显存 8GB 以上，4-bit 量化后可适配 6GB
- HF_ENDPOINT 已设置为国内镜像（`https://hf-mirror.com`）

