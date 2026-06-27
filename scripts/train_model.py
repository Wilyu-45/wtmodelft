"""
QLoRA训练脚本
使用标准的 transformers + peft + bitsandbytes
支持4bit量化，大幅降低显存需求
"""

import sys
import torch
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import Dataset

def check_cuda():
    """检查CUDA环境"""
    if not torch.cuda.is_available():
        print("[错误] 未检测到CUDA环境，无法进行训练")
        return False

    print(f"[信息] CUDA设备: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[信息] 显存总量: {mem:.1f} GB")
    return True

def load_dataset(data_file: Path):
    """加载训练数据集"""
    if not data_file.exists():
        print(f"[错误] 数据文件不存在: {data_file}")
        return None

    dataset = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                dataset.append(data)
            except json.JSONDecodeError:
                continue

    print(f"[信息] 成功加载 {len(dataset)} 个训练样本")
    return dataset

def prepare_dataset_for_training(dataset, tokenizer, data_format: str, max_length: int):
    """准备适合训练的数据集格式"""

    def format_chatml_to_text(item, tokenizer, max_length):
        """将ChatML格式转换为纯文本"""
        if "messages" in item:
            text = ""
            for msg in item["messages"]:
                if msg["role"] == "system":
                    text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
                elif msg["role"] == "user":
                    text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                elif msg["role"] == "assistant":
                    text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            text += "<|im_end|>"
        else:
            text = item.get("text", str(item))

        encodings = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings

    print("[信息] 准备训练数据...")
    processed_data = []
    for item in dataset:
        if data_format == "chatml":
            formatted = format_chatml_to_text(item, tokenizer, max_length)
            processed_data.append(formatted)
        elif data_format == "alpaca":
            if "instruction" in item and "output" in item:
                text = f"### Instruction:\n{item.get('instruction', '')}\n\n### Response:\n{item.get('output', '')}"
            else:
                text = str(item)
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
            encodings["labels"] = encodings["input_ids"].copy()
            processed_data.append(encodings)
        else:
            text = item.get("text", str(item))
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
            encodings["labels"] = encodings["input_ids"].copy()
            processed_data.append(encodings)

    return Dataset.from_list(processed_data)

def create_model_and_trainer(
    dataset,
    base_model: str = None,
    max_seq_length: int = None,
    lora_r: int = None,
    lora_alpha: int = None,
    per_device_batch_size: int = None,
    num_epochs: int = None,
    learning_rate: float = None,
    output_dir: str = None
):
    """创建模型和训练器"""

    if base_model is None:
        base_model = config.MODEL_CONFIG["base_model_name"]
    if max_seq_length is None:
        max_seq_length = config.MODEL_CONFIG["max_seq_length"]
    if lora_r is None:
        lora_r = config.LORA_CONFIG["r"]
    if lora_alpha is None:
        lora_alpha = config.LORA_CONFIG["lora_alpha"]
    if per_device_batch_size is None:
        per_device_batch_size = config.TRAINING_CONFIG["per_device_train_batch_size"]
    if num_epochs is None:
        num_epochs = config.TRAINING_CONFIG["num_train_epochs"]
    if learning_rate is None:
        learning_rate = config.TRAINING_CONFIG["learning_rate"]
    if output_dir is None:
        output_dir = config.TRAINING_CONFIG["output_dir"]

    print("=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"基座模型: {base_model}")
    print(f"最大序列长度: {max_seq_length}")
    print(f"LoRA Rank: {lora_r}")
    print(f"LoRA Alpha: {lora_alpha}")
    print(f"批次大小: {per_device_batch_size}")
    print(f"训练轮数: {num_epochs}")
    print(f"学习率: {learning_rate}")
    print(f"输出目录: {output_dir}")
    print()

    print("[1/6] 加载分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=False
        )
    except Exception as e:
        print(f"[警告] 从 HuggingFace 加载失败，尝试 ModelScope...")
        try:
            from modelscope import AutoTokenizer as MsAutoTokenizer
            tokenizer = MsAutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True
            )
        except:
            print(f"[错误] ModelScope 也加载失败，请手动下载模型到本地")
            raise e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("[完成] 分词器加载成功")

    print("[2/6] 配置4bit量化...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True
    )
    print("[完成] 量化配置完成")

    print("[3/6] 加载基座模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"[警告] 量化加载失败，尝试不用量化...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
        except:
            print(f"[错误] 模型加载失败: {e}")
            raise e

    model.config.use_cache = False
    print("[完成] 基座模型加载成功")

    print("[4/6] 准备模型用于kbit训练...")
    model = prepare_model_for_kbit_training(model)
    print("[完成] 模型准备完成")

    print("[5/6] 配置LoRA适配器...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=config.LORA_CONFIG["target_modules"],
        lora_dropout=config.LORA_CONFIG["lora_dropout"],
        bias=config.LORA_CONFIG["bias"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("[完成] LoRA配置成功")

    print("[6/6] 准备训练数据集...")
    processed_dataset = prepare_dataset_for_training(
        dataset, tokenizer, config.DATA_FORMAT, max_seq_length
    )
    print(f"[完成] 数据集准备完成，共 {len(processed_dataset)} 个样本")

    print()
    print("[7/7] 创建训练器...")

    training_arguments = TrainingArguments(
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=config.TRAINING_CONFIG["gradient_accumulation_steps"],
        warmup_steps=config.TRAINING_CONFIG["warmup_steps"],
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        bf16=False,
        logging_steps=config.TRAINING_CONFIG["logging_steps"],
        optim="paged_adamw_8bit",
        weight_decay=config.TRAINING_CONFIG["weight_decay"],
        lr_scheduler_type=config.TRAINING_CONFIG["lr_scheduler_type"],
        seed=config.TRAINING_CONFIG["seed"],
        output_dir=output_dir,
        report_to=config.TRAINING_CONFIG["report_to"],
        max_grad_norm=config.TRAINING_CONFIG["max_grad_norm"],
        save_steps=config.TRAINING_CONFIG["save_steps"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )
    print("[完成] 训练器创建成功")

    print()
    print("[开始训练]...")
    print("-" * 60)
    trainer.train()
    print("-" * 60)
    print("[完成] 训练完成!")

    return model, tokenizer, trainer

def save_model(model, tokenizer, save_lora: bool = True, save_full: bool = False):
    """保存模型"""
    config.ensure_dirs()

    if save_lora:
        lora_path = config.get_lora_path()
        model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)
        print(f"[信息] LoRA权重已保存至: {lora_path}")

    if save_full:
        full_model_path = config.get_model_path()
        print("[信息] 保存完整模型...")
        model.save_pretrained(full_model_path)
        tokenizer.save_pretrained(full_model_path)
        print(f"[信息] 完整模型已保存至: {full_model_path}")

def train(
    data_file: Path = None,
    base_model: str = None,
    num_epochs: int = 3,
    save_lora: bool = True,
    save_full: bool = True
):
    """完整的训练流程"""
    if not check_cuda():
        return None

    if data_file is None:
        data_file = config.TRAINING_DATA_FILE

    if not data_file.exists():
        print(f"[错误] 训练数据文件不存在: {data_file}")
        print("[提示] 请先运行 process_data.py 处理您的文章数据")
        return None

    print()
    print("个人风格模型 - 训练工具")
    print("=" * 60)

    print("[信息] 加载训练数据...")
    dataset = load_dataset(data_file)
    if not dataset:
        return None

    print()
    model, tokenizer, trainer = create_model_and_trainer(
        dataset=dataset,
        base_model=base_model,
        num_epochs=num_epochs
    )

    print()
    print("[信息] 保存模型...")
    save_model(model, tokenizer, save_lora=save_lora, save_full=save_full)

    print()
    print("=" * 60)
    print("训练完成!")
    print("=" * 60)

    return {
        "lora_path": str(config.get_lora_path()) if save_lora else None,
        "full_model_path": str(config.get_model_path()) if save_full else None
    }

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="个人风格模型训练")
    parser.add_argument("--data", type=str, default=None,
                        help="训练数据文件路径")
    parser.add_argument("--model", type=str, default=None,
                        help="基座模型名称")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--save_lora", action="store_true", default=True,
                        help="保存LoRA权重")
    parser.add_argument("--save_full", action="store_true", default=True,
                        help="保存完整模型")
    parser.add_argument("--no_full", action="store_true",
                        help="不保存完整模型")

    args = parser.parse_args()

    data_file = Path(args.data) if args.data else None
    base_model = args.model
    num_epochs = args.epochs
    save_full = not args.no_full

    result = train(
        data_file=data_file,
        base_model=base_model,
        num_epochs=num_epochs,
        save_lora=args.save_lora,
        save_full=save_full
    )

    if result:
        print("\n训练结果:")
        if result["lora_path"]:
            print(f"  LoRA权重: {result['lora_path']}")
        if result["full_model_path"]:
            print(f"  完整模型: {result['full_model_path']}")

if __name__ == "__main__":
    main()