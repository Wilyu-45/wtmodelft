"""
Personal Style Model Training Configuration
个人风格模型训练配置
"""

import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MODELSCOPE_SDK_DEBUG"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# 网络超时配置（秒）
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

# 数据文件路径
TRAINING_DATA_FILE = PROCESSED_DATA_DIR / "training_data.jsonl"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test_data.jsonl"

# 模型配置
MODEL_CONFIG = {
    "base_model_name": str(MODEL_DIR / "base_model_huff"),
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "dtype": None,
}

# LoRA配置
LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "bias": "none",
}

# 训练配置
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 50,
    "num_train_epochs": 5,
    "learning_rate": 1e-4,
    "fp16": True,
    "bf16": False,
    "logging_steps": 10,
    "optim": "paged_adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "seed": 3407,
    "output_dir": str(OUTPUT_DIR),
    "report_to": "none",
    "max_grad_norm": 0.3,
    "save_steps": 100,
}

# 数据处理配置
DATA_CONFIG = {
    "min_article_length": 100,
    "max_article_length": 10000,
    "chunk_size": 512,
    "overlap": 50,
    "train_ratio": 0.9,
}

# 推理配置
INFERENCE_CONFIG = {
    "max_new_tokens": 600,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
    "repetition_penalty": 1.2,
}

# 系统提示词（用于定义模型角色）
SYSTEM_PROMPT = """你是一个模仿用户写作风格的助手。请严格按照用户的用词习惯、句式结构和表达方式来生成文本。"""

# 数据格式化类型：["alpaca", "chatml", "plain"]
DATA_FORMAT = "chatml"

def ensure_dirs():
    """确保所有必要的目录存在"""
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

def get_model_path():
    """获取最终模型保存路径"""
    return MODEL_DIR / "my_style_model"

def get_lora_path():
    """获取LoRA权重保存路径"""
    return MODEL_DIR / "lora_adapter"