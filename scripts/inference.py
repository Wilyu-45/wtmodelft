"""
单文件风格分析脚本 v8
- 处理单个长文本文件
- 保持滑动窗口 + 温度衰减 + 风格锚点
- 仅支持 HuggingFace 模型格式
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class StyleAnalyzer:
    """风格分析器"""

    def __init__(self, model_path=None, lora_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not torch.cuda.is_available():
            print("[警告] 未检测到CUDA，将使用CPU运行")

        if model_path and Path(model_path).exists():
            print(f"[信息] 加载 HuggingFace 模型: {model_path}")
            self.model, self.tokenizer = self._load_model(model_path)
        elif lora_path and Path(lora_path).exists():
            print(f"[信息] 加载 LoRA 权重: {lora_path}")
            self.model, self.tokenizer = self._load_lora(lora_path)
        else:
            raise FileNotFoundError("未找到模型文件")

        self.MAX_SEQ_LENGTH = config.MODEL_CONFIG.get("max_seq_length", 2048)
        print(f"[信息] 模型加载完成，运行设备: {self.device}")
        print(f"[信息] 最大序列长度: {self.MAX_SEQ_LENGTH} tokens")

    def _load_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        return model, tokenizer

    def _load_lora(self, lora_path):
        base_model = config.MODEL_CONFIG["base_model_name"]

        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

        return model, tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _tokens_to_chars(self, tokens: int) -> int:
        return int(tokens * 1.5)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 600,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)

        prompt_length = inputs.input_ids.shape[1]

        available_tokens = self.MAX_SEQ_LENGTH - prompt_length
        actual_max_tokens = min(max_new_tokens, available_tokens)

        if actual_max_tokens <= 0:
            print(f"[警告] Prompt 过长({prompt_length} tokens)，无可用生成空间")
            return ""

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=actual_max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[-1].strip()

        return response

    def _build_system_prompt(self, task: str, style_anchors: str = None) -> str:
        task_descriptions = {
            "杂谈": "对内容进行轻松随意的聊天式评论",
            "吐槽": "用调侃的方式指出内容的槽点",
            "总结": "简洁明了地总结内容的核心要点",
            "扩写": "在保持风格的基础上丰富扩展内容",
            "风格重写": "用你的写作风格重新表达内容"
        }

        task_desc = task_descriptions.get(task, f"对内容进行{task}")

        few_shot_example = ""

        if task == "杂谈":
            few_shot_example = """
【正确范例】
用户：请用杂谈的方式聊聊这段内容：主角在路上捡到100元。
助手：诶，这个设定还挺有意思的。捡钱这事吧，谁没遇到过呢？但问题是，捡到钱之后高兴一整天，是不是有点过了？"""

        elif task == "吐槽":
            few_shot_example = """
【正确范例】
用户：请吐槽这个剧情：主角在路上捡到100元，然后高兴了一天。
助手：讲真，这剧情也太敷衍了吧？捡个钱都能水一整天，作者是不是没东西写了？格局呢？"""

        elif task == "总结":
            few_shot_example = """
【正确范例】
用户：请总结这篇文章的核心观点。
助手：这篇文章主要说了三件事：第一，xxx；第二，yyy；第三，zzz。"""

        elif task == "风格重写":
            few_shot_example = """
【正确范例】
用户：请用我的风格重写这段：今天天气很好，我出门散步。
助手：今天这天气，不出门溜达简直对不起自己。得，出门逛逛去。"""

        base = f"""你是一个使用特定写作风格的{task}专家。
你的核心任务是：基于用户提供的【待处理内容】，运用你熟悉的写作风格，进行{task}。
{task_desc}。
你必须严格遵守以下【工作流程】：
1. **仔细阅读**【待处理内容】本身
2. **只基于**内容本身进行{task}，不要引入外部知识
3. **保持风格**与用户一致，输出专业结果"""

        if style_anchors:
            base = base + f"\n\n用户风格特征：{style_anchors}"

        base = base + few_shot_example
        return base

    def _build_first_segment_prompt(
        self,
        segment: str,
        task: str,
        total_segments: int,
        style_anchors: str = None
    ) -> str:
        system = self._build_system_prompt(task, style_anchors)

        prompt = f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
【待处理内容】（请只针对以下内容进行{task}，不要引入任何外部知识）：

{segment}

请开始你的{task}（严格基于以上内容）：

<|im_end|>
<|im_start|>assistant
"""
        return prompt

    def _build_continuation_prompt(
        self,
        segment: str,
        previous_result: str,
        context_tokens: int,
        task: str,
        segment_idx: int,
        total_segments: int,
        style_anchors: str = None
    ) -> str:
        system = self._build_system_prompt(task, style_anchors)

        context_chars = self._tokens_to_chars(context_tokens)
        previous_excerpt = previous_result[-context_chars:] if len(previous_result) > context_chars else previous_result

        prompt = f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
【前文{task}结果】（必须保持完全相同的写作风格）：

{previous_excerpt}

【待处理内容】（这是第{segment_idx+1}/{total_segments}部分，请严格基于以下内容进行{task}）：

{segment}

请继续{task}（必须与前文风格完全一致，承接逻辑）：

<|im_end|>
<|im_start|>assistant
"""
        return prompt

    def _build_final_segment_prompt(
        self,
        segment: str,
        previous_result: str,
        context_tokens: int,
        task: str,
        total_segments: int,
        style_anchors: str = None
    ) -> str:
        system = self._build_system_prompt(task, style_anchors)

        context_chars = self._tokens_to_chars(context_tokens)
        previous_excerpt = previous_result[-context_chars:] if len(previous_result) > context_chars else previous_result

        prompt = f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
【前文{task}结果】（必须保持完全相同的写作风格）：

{previous_excerpt}

【待处理内容】（这是最后一部分，请做好总结收尾）：

{segment}

请完成{task}并做好总结：

<|im_end|>
<|im_start|>assistant
"""
        return prompt

    def analyze_long_content(
        self,
        content: str,
        task: str = "杂谈",
        max_input_tokens: int = 1500,
        max_output_tokens: int = 300,
        overlap_tokens: int = 250,
        style_anchors: str = None,
        temperature_start: float = 0.7,
        temperature_end: float = 0.5
    ) -> str:
        total_chars = len(content)
        total_tokens = self.count_tokens(content)

        print(f"[信息] 内容长度: {total_chars}字, 约{total_tokens} tokens")
        print(f"[信息] 模型最大长度: {self.MAX_SEQ_LENGTH} tokens")

        if total_tokens <= max_input_tokens - 500:
            print("[信息] 内容较短，直接分析")
            return self._single_analyze(content, task, max_output_tokens)

        segments = self._split_into_segments(content, max_input_tokens, overlap_tokens)
        num_segments = len(segments)

        print(f"[信息] 将内容分成 {num_segments} 段进行分析")
        print(f"[策略] 循环累积 + 温度衰减 ({temperature_start} → {temperature_end})")

        accumulated_result = ""

        for i, seg in enumerate(segments):
            temperature = temperature_start - (temperature_start - temperature_end) * (i / max(num_segments - 1, 1))
            print(f"  分析第 {i+1}/{num_segments} 段... (temp={temperature:.2f})")

            is_first = (i == 0)
            is_last = (i == num_segments - 1)

            if is_first:
                prompt = self._build_first_segment_prompt(seg, task, num_segments, style_anchors)
            elif is_last:
                prompt = self._build_final_segment_prompt(seg, accumulated_result, overlap_tokens, task, num_segments, style_anchors)
            else:
                prompt = self._build_continuation_prompt(seg, accumulated_result, overlap_tokens, task, i, num_segments, style_anchors)

            segment_result = self.generate(
                prompt,
                max_new_tokens=max_output_tokens,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.2
            )

            accumulated_result += segment_result

        return accumulated_result

    def _single_analyze(self, content: str, task: str, max_output_tokens: int) -> str:
        system = self._build_system_prompt(task)

        prompt = f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
请用我的写作风格对以下内容进行{task}：

{content}

<|im_end|>
<|im_start|>assistant
"""

        return self.generate(prompt, max_new_tokens=max_output_tokens)

    def _split_into_segments(self, content: str, max_tokens: int, overlap_tokens: int) -> list:
        tokens = self.tokenizer.encode(content, add_special_tokens=False)

        segments = []
        start = 0

        while start < len(tokens):
            end = start + max_tokens
            segment_tokens = tokens[start:end]
            segment_text = self.tokenizer.decode(segment_tokens)
            if hasattr(segment_text, 'decode'):
                segment_text = segment_text.decode('utf-8', errors='ignore')
            segments.append(segment_text.strip())
            start = end - overlap_tokens
            if start >= len(tokens) - overlap_tokens:
                break

        return segments


def main():
    import argparse

    parser = argparse.ArgumentParser(description="单文件风格分析")
    parser.add_argument("--input", type=str, required=True,
                        help="输入文件路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出文件路径")
    parser.add_argument("--task", type=str, default="杂谈",
                        help="分析任务描述")
    parser.add_argument("--max_input_tokens", type=int, default=1500,
                        help="每段最大输入token数")
    parser.add_argument("--max_output_tokens", type=int, default=300,
                        help="每段最大输出token数")
    parser.add_argument("--overlap_tokens", type=int, default=250,
                        help="滑动窗口重叠token数")
    parser.add_argument("--style_anchors", type=str, default=None,
                        help="风格锚点描述")
    parser.add_argument("--temp_start", type=float, default=0.7,
                        help="初始温度")
    parser.add_argument("--temp_end", type=float, default=0.5,
                        help="最终温度")
    parser.add_argument("--model", type=str, default=None,
                        help="完整模型路径")
    parser.add_argument("--lora", type=str, default=None,
                        help="LoRA权重路径")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[错误] 输入文件不存在: {input_path}")
        return

    print("=" * 60)
    print("单文件风格分析 (v8)")
    print("=" * 60)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"任务: {args.task}")
    print(f"最大输入: {args.max_input_tokens} tokens/段")
    print(f"最大输出: {args.max_output_tokens} tokens/段")
    print(f"重叠 tokens: {args.overlap_tokens}")
    print(f"温度衰减: {args.temp_start} → {args.temp_end}")
    if args.style_anchors:
        print(f"风格锚点: {args.style_anchors}")
    print("=" * 60)

    model_path = args.model or str(config.get_model_path())
    lora_path = args.lora or str(config.get_lora_path())

    try:
        analyzer = StyleAnalyzer(model_path=model_path, lora_path=lora_path)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        print("\n请先运行训练:")
        print("  python scripts/train_model.py")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            print(f"[错误] 输入文件为空: {input_path}")
            return

        print(f"\n开始分析...\n")

        result = analyzer.analyze_long_content(
            content,
            task=args.task,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            overlap_tokens=args.overlap_tokens,
            style_anchors=args.style_anchors,
            temperature_start=args.temp_start,
            temperature_end=args.temp_end
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)

        print(f"\n[完成] 结果已保存到: {output_path}")

    except Exception as e:
        print(f"[错误] 处理失败: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)


if __name__ == "__main__":
    main()
