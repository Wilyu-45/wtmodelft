"""
项目环境验证脚本
检查所有依赖和配置是否正确
"""

import sys
import subprocess
from pathlib import Path

def print_status(message, status):
    """打印状态信息"""
    symbols = {"ok": "✓", "error": "✗", "info": "○"}
    symbol = symbols.get(status, "-")
    print(f"  [{symbol}] {message}")

def check_python():
    """检查Python版本"""
    print("\n检查Python环境...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro}", "ok")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor} (需要 3.8+)", "error")
        return False

def check_cuda():
    """检查CUDA"""
    print("\n检查CUDA环境...")
    try:
        import torch
        if torch.cuda.is_available():
            print_status(f"CUDA可用: {torch.cuda.get_device_name(0)}", "ok")
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print_status(f"显存: {mem:.1f} GB", "info")
            print_status(f"PyTorch版本: {torch.__version__}", "info")
            return True
        else:
            print_status("CUDA不可用", "error")
            return False
    except ImportError:
        print_status("PyTorch未安装", "error")
        return False

def check_packages():
    """检查必要的包"""
    print("\n检查Python包...")
    required = [
        "torch",
        "transformers",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "modelscope"
    ]

    all_ok = True
    for package in required:
        try:
            __import__(package)
            print_status(f"{package}", "ok")
        except ImportError:
            print_status(f"{package} (需要安装)", "error")
            all_ok = False

    return all_ok

def check_directories():
    """检查目录结构"""
    print("\n检查目录结构...")
    project_root = Path(__file__).parent.parent

    dirs = {
        "data/raw": "存放原始文章",
        "data/processed": "存放处理后数据",
        "scripts": "脚本目录",
        "output": "训练输出目录",
        "models": "模型保存目录"
    }

    all_ok = True
    for dir_name, desc in dirs.items():
        dir_path = project_root / dir_name
        if dir_path.exists():
            files = list(dir_path.glob('*'))
            print_status(f"{dir_name} ({len(files)} 个文件) - {desc}", "ok")
        else:
            print_status(f"{dir_name} (需要创建) - {desc}", "error")
            all_ok = False

    return all_ok

def check_data():
    """检查数据"""
    print("\n检查训练数据...")
    project_root = Path(__file__).parent.parent
    data_file = project_root / "data" / "processed" / "training_data.jsonl"

    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print_status(f"训练数据: {data_file} ({len(lines)} 条)", "ok")
        return True
    else:
        print_status("训练数据不存在，请先运行数据处理", "info")
        return False

def check_model():
    """检查模型"""
    print("\n检查模型文件...")
    project_root = Path(__file__).parent.parent

    lora_path = project_root / "models" / "lora_adapter"
    full_path = project_root / "models" / "my_style_model"

    if full_path.exists():
        print_status(f"完整模型: {full_path}", "ok")
        return True
    elif lora_path.exists():
        print_status(f"LoRA权重: {lora_path}", "ok")
        return True
    else:
        print_status("模型文件不存在，请先运行训练", "info")
        return False

def print_intro():
    """打印介绍"""
    print("=" * 60)
    print("个人风格模型 - 环境验证")
    print("=" * 60)

def print_guide():
    """打印使用指南"""
    print("\n" + "=" * 60)
    print("使用指南")
    print("=" * 60)
    print("""
步骤1: 准备数据
  将您的文章(.txt, .md)放入: data/raw/
  运行: python scripts/process_data.py

步骤2: 训练模型
  运行: python scripts/train_model.py
  或指定参数: python scripts/train_model.py --epochs 5

步骤3: 批量分析
  将待分析文本放入: input/
  运行: python scripts/batch_inference.py --task "吐槽"
  支持长内容分段处理 --max_input_tokens 1500
    """)

def main():
    """主函数"""
    print_intro()

    check_python()
    check_cuda()
    check_packages()
    check_directories()
    check_data()
    check_model()

    print_guide()

    print("\n验证完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()