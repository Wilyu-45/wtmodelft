"""
数据处理脚本
将原始文章转换为训练数据集
支持多种数据格式：Alpaca, ChatML, Plain
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import config

class ArticleProcessor:
    def __init__(self):
        self.min_length = config.DATA_CONFIG["min_article_length"]
        self.max_length = config.DATA_CONFIG["max_article_length"]
        self.chunk_size = config.DATA_CONFIG["chunk_size"]
        self.overlap = config.DATA_CONFIG["overlap"]

    def clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text or not isinstance(text, str):
            return ""

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        text = text.strip()

        return text

    def split_into_chunks(self, text: str) -> List[str]:
        """将长文本分割成块（使用滑动窗口）"""
        if len(text) <= self.min_length:
            return [text] if len(text) >= self.min_length else []

        step = self.chunk_size - self.overlap
        if step <= 0:
            step = self.chunk_size // 2

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()

            if len(chunk) >= self.min_length:
                chunks.append(chunk)

            if end >= len(text):
                break

            start += step

        return chunks

    def format_alpaca(self, instruction: str, input_text: str, output: str) -> Dict:
        """Alpaca格式"""
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }

    def format_chatml(self, user_msg: str, assistant_msg: str,
                     system_prompt: Optional[str] = None) -> Dict:
        """ChatML格式 (Qwen原生)"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

        return {"messages": messages}

    def format_plain(self, text: str) -> Dict:
        """Plain格式（直接续写）"""
        return {"text": text}

    def create_training_sample(self, chunk: str, data_format: str) -> Dict:
        """根据指定格式创建训练样本"""
        import random

        user_instructions = [
            "请模仿我的写作风格，写一段关于{topic}的文字。",
            "用我之前的风格，续写一段关于{topic}的内容。",
            "严格按照我的行文习惯，写一段{topic}的描述。",
            "模仿我的语言风格，创作一段关于{topic}的文字。",
            "请用我上面的写作方式，来写一段{topic}。",
            "参考之前的文字风格，写一段关于{topic}的内容。",
            "用我惯用的表达方式，来描写{topic}。",
            "按我上面的文风，写一段{topic}相关的文字。",
            "我的写作风格是这样的，请按这个风格写{topic}。",
            "根据前文的语气和用词，创作一段关于{topic}的文字。",
        ]

        topics = [
            "动画评论", "新番吐槽", "季度总结", "观后感",
            "个人感想", "生活感悟", "动漫推荐", "作品分析",
            "业界观察", "文化讨论"
        ]

        random.seed()
        user_msg = random.choice(user_instructions).format(topic=random.choice(topics))

        if data_format == "alpaca":
            instruction = "请模仿以下文本的写作风格和表达方式，续写或创作新内容。"
            return self.format_alpaca(instruction, "", chunk)

        elif data_format == "chatml":
            return self.format_chatml(user_msg, chunk, config.SYSTEM_PROMPT)

        elif data_format == "plain":
            return self.format_plain(chunk)

        else:
            raise ValueError(f"不支持的数据格式: {data_format}")

    def process_file(self, file_path: Path) -> List[Dict]:
        """处理单个文件，返回训练样本列表"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            cleaned = self.clean_text(content)

            if len(cleaned) < self.min_length:
                print(f"  [警告] 文件 {file_path.name} 内容过短，跳过")
                return []

            chunks = self.split_into_chunks(cleaned)
            if not chunks:
                chunks = [cleaned]

            samples = []
            for i, chunk in enumerate(chunks):
                sample = self.create_training_sample(chunk, config.DATA_FORMAT)
                samples.append(sample)

            print(f"  [成功] {file_path.name}: 生成了 {len(samples)} 个训练样本")
            return samples

        except Exception as e:
            print(f"  [错误] 处理文件 {file_path.name} 时出错: {str(e)}")
            return []

def load_raw_articles(data_dir: Path) -> List[Path]:
    """加载所有原始文章文件（自动去重）"""
    supported_formats = {'.txt', '.md', '.text'}

    files_dict = {}
    for file_path in data_dir.iterdir():
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in supported_formats:
                key = file_path.name.lower()
                if key not in files_dict:
                    files_dict[key] = file_path

    return sorted(files_dict.values())

def prepare_dataset(
    raw_data_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    test_split: float = 0.1,
    data_format: str = "chatml"
):
    """
    准备训练数据集

    Args:
        raw_data_dir: 原始数据目录
        output_file: 输出文件路径
        test_split: 测试集比例
        data_format: 数据格式 ("alpaca", "chatml", "plain")
    """
    if raw_data_dir is None:
        raw_data_dir = config.RAW_DATA_DIR
    if output_file is None:
        output_file = config.TRAINING_DATA_FILE

    config.DATA_FORMAT = data_format

    print("=" * 60)
    print("开始处理数据")
    print("=" * 60)
    print(f"原始数据目录: {raw_data_dir}")
    print(f"输出文件: {output_file}")
    print(f"数据格式: {data_format}")
    print()

    raw_files = load_raw_articles(raw_data_dir)

    if not raw_files:
        print(f"[错误] 在 {raw_data_dir} 中未找到任何文章文件")
        print("支持的格式: .txt, .md, .text")
        return None

    print(f"找到 {len(raw_files)} 个文章文件")
    print()

    processor = ArticleProcessor()
    all_samples = []

    for i, file_path in enumerate(raw_files, 1):
        print(f"处理文件 [{i}/{len(raw_files)}]:")
        samples = processor.process_file(file_path)
        all_samples.extend(samples)

    if not all_samples:
        print("[错误] 没有生成任何有效的训练样本")
        return None

    print()
    print(f"共生成 {len(all_samples)} 个训练样本")

    import random
    random.seed(42)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * (1 - test_split))
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]

    print(f"训练集: {len(train_samples)} 个样本")
    print(f"测试集: {len(test_samples)} 个样本")

    config.ensure_dirs()

    train_output = config.TRAINING_DATA_FILE
    test_output = config.TEST_DATA_FILE

    with open(train_output, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    with open(test_output, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print()
    print(f"训练数据已保存至: {train_output}")
    print(f"测试数据已保存至: {test_output}")
    print("=" * 60)

    return {
        "train_file": train_output,
        "test_file": test_output,
        "train_count": len(train_samples),
        "test_count": len(test_samples)
    }

def main():
    """主函数"""
    print()
    print("个人风格模型 - 数据处理工具")
    print("=" * 60)
    print()

    config.ensure_dirs()

    raw_files = load_raw_articles(config.RAW_DATA_DIR)

    if not raw_files:
        print("[提示] 原始数据目录为空")
        print(f"请将您的文章放入目录: {config.RAW_DATA_DIR}")
        print("支持的格式: .txt, .md, .text")
        print()

        example_usage = """
示例用法:
---------

# 方式1: 编程调用
from process_data import prepare_dataset

result = prepare_dataset(
    raw_data_dir='data/raw',
    output_file='data/processed/training_data.jsonl',
    test_split=0.1,
    data_format='chatml'
)

# 方式2: 命令行
python process_data.py

# 方式3: 直接运行后手动添加数据
# 将文章放入 data/raw 目录后，再次运行此脚本
        """
        print(example_usage)
    else:
        print(f"在 {config.RAW_DATA_DIR} 中找到 {len(raw_files)} 个文件")
        print()

        prepare_dataset(
            raw_data_dir=config.RAW_DATA_DIR,
            test_split=1 - config.DATA_CONFIG.get("train_ratio", 0.9),
            data_format=config.DATA_FORMAT
        )

if __name__ == "__main__":
    main()