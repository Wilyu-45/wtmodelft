"""
风格分析推理界面 v4
基于 Gradio 的可视化操作界面
支持滑动窗口 + 温度衰减
"""

import sys
import os
import gradio as gr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from inference import StyleAnalyzer


def analyze_content(
    input_file=None,
    output_file=None,
    task=None,
    max_input_tokens=1500,
    max_output_tokens=300,
    overlap_tokens=250,
    style_anchors=None,
    temp_start=0.7,
    temp_end=0.5,
    model_path=None,
    lora_path=None
):
    if input_file is None:
        return "❌ 请上传输入文件", None

    default_model_path = str(config.get_model_path())
    default_lora_path = str(config.get_lora_path())

    user_specified_model = model_path and model_path.strip()
    if user_specified_model:
        actual_model_path = model_path.strip()
    else:
        actual_model_path = default_model_path

    actual_lora_path = lora_path.strip() if lora_path and lora_path.strip() else default_lora_path

    if actual_lora_path and not Path(actual_lora_path).exists():
        return f"❌ LoRA路径不存在: {actual_lora_path}", None

    try:
        analyzer = StyleAnalyzer(model_path=actual_model_path, lora_path=actual_lora_path)

        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            return "❌ 输入文件为空", None

        total_chars = len(content)
        total_tokens = analyzer.count_tokens(content)

        accumulated_result = ""

        if total_tokens <= max_input_tokens - 500:
            result = analyzer._single_analyze(content, task, max_output_tokens)
            accumulated_result = result
            num_segments = 1
        else:
            segments = analyzer._split_into_segments(content, max_input_tokens, overlap_tokens)
            num_segments = len(segments)

            for i, seg in enumerate(segments):
                temperature = temp_start - (temp_start - temp_end) * (i / max(num_segments - 1, 1))
                is_first = (i == 0)
                is_last = (i == num_segments - 1)

                if is_first:
                    prompt = analyzer._build_first_segment_prompt(seg, task, num_segments, style_anchors)
                elif is_last:
                    prompt = analyzer._build_final_segment_prompt(seg, accumulated_result, overlap_tokens, task, num_segments, style_anchors)
                else:
                    prompt = analyzer._build_continuation_prompt(seg, accumulated_result, overlap_tokens, task, i, num_segments, style_anchors)

                segment_result = analyzer.generate(
                    prompt,
                    max_new_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    repetition_penalty=1.2
                )

                accumulated_result += segment_result

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(accumulated_result)

        output_size = len(accumulated_result)
        return f"✅ 分析完成！\n\n📊 统计信息：\n- 输入字数：{total_chars}\n- 输入 tokens：约 {total_tokens}\n- 分段数：{num_segments}\n- 输出字数：{output_size}\n- 保存路径：{output_file}", str(Path(output_file).absolute())

    except FileNotFoundError as e:
        return f"❌ 文件未找到: {e}", None
    except Exception as e:
        return f"❌ 处理失败: {e}", None


def main():
    default_model_path = str(config.get_model_path())
    default_lora_path = str(config.get_lora_path())

    with gr.Blocks(
        title="风格分析推理界面"
    ) as demo:
        gr.Markdown("# 🖊️ 风格分析推理界面")
        gr.Markdown("基于 Qwen 的长文本风格分析工具，支持滑动窗口 + 温度衰减")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📁 文件设置")
                input_file = gr.File(label="输入文件 (.txt)", file_types=[".txt"])
                output_file = gr.Textbox(
                    label="输出文件路径",
                    value="result/result.txt",
                    placeholder="例如: result/result.txt"
                )

            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ 任务设置")
                task = gr.Dropdown(
                    label="任务类型",
                    choices=["杂谈", "吐槽", "总结", "扩写", "风格重写"],
                    value="杂谈"
                )
                style_anchors = gr.Textbox(
                    label="风格锚点（可选）",
                    placeholder="描述你的写作风格特征，例如：喜欢用\"绝了\"、\"离谱\"等口头禅...",
                    lines=3
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🔢 Token 参数")
                with gr.Row():
                    max_input_tokens = gr.Slider(
                        minimum=800,
                        maximum=1800,
                        value=1500,
                        step=100,
                        label="每段最大输入 tokens",
                        info="建议 1500，增大可减少分段数"
                    )
                    max_output_tokens = gr.Slider(
                        minimum=200,
                        maximum=800,
                        value=300,
                        step=50,
                        label="每段最大输出 tokens",
                        info="建议 300，输出太长会影响连贯性"
                    )
                overlap_tokens = gr.Slider(
                    minimum=100,
                    maximum=400,
                    value=250,
                    step=30,
                    label="滑动窗口重叠 tokens",
                    info="建议 250-300，上下文连贯性更好"
                )

            with gr.Column():
                gr.Markdown("## 🌡️ 生成参数")
                with gr.Row():
                    temp_start = gr.Slider(
                        minimum=0.3,
                        maximum=1.0,
                        value=0.7,
                        step=0.05,
                        label="初始温度",
                        info="较高温度生成更具创造性"
                    )
                    temp_end = gr.Slider(
                        minimum=0.3,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="最终温度",
                        info="较低温度生成更稳定一致"
                    )

        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🧠 模型路径")
                model_path = gr.Textbox(
                    label="模型路径（可选）",
                    placeholder=f"留空使用默认: {default_model_path}",
                    value=""
                )
                lora_path = gr.Textbox(
                    label="LoRA 权重路径（可选）",
                    placeholder=f"留空使用默认: {default_lora_path}",
                    value=""
                )

        with gr.Row():
            submit_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")

        with gr.Row():
            output_msg = gr.Textbox(label="处理结果", lines=10, interactive=False)

        with gr.Row():
            result_file = gr.File(label="📥 结果文件下载", height=100)

        gr.Markdown("---")
        gr.Markdown("### 💡 使用建议")
        gr.Markdown("""
1. **输出控制**：max_output 建议 300 左右，输出太长会导致各段风格不连贯
2. **二次提炼**：可先生成初稿吐槽，再用"精简总结"模式提炼精华
3. **长文本处理**：30000 字文本会自动分段处理
4. **显存优化**：使用 4-bit 量化，8G 显存足够运行
""")

        gr.Markdown("""
        ### ⌨️ 推荐参数组合
        | 场景 | max_input | max_output | overlap | 分段数(3万字) |
        |------|-----------|------------|---------|---------------|
        | 标准吐槽 | 1500 | 300 | 250 | ~14段 |
        | 精简总结 | 1200 | 200 | 200 | ~17段 |
        | 详细扩写 | 1800 | 500 | 300 | ~12段 |
        """)

        submit_btn.click(
            fn=analyze_content,
            inputs=[
                input_file, output_file, task,
                max_input_tokens, max_output_tokens, overlap_tokens,
                style_anchors, temp_start, temp_end,
                model_path, lora_path
            ],
            outputs=[output_msg, result_file]
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
