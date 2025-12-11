"""
Gradio UI界面模块
"""
import os
import json
import gradio as gr
from datetime import datetime
from ..config.settings import DEFAULT_CONFIG
from ..utils import log, log_collector
from ..models import load_model_interface
from ..training import start_training_interface, generate_mbpp_training_data
from ..evaluation import start_evaluation_interface, get_comparison_results
from ..utils import process_instruction_with_local_model

def create_interface():
    """创建Gradio界面"""
    
    # ====== 创建完整的Gradio界面 ======
    with gr.Blocks(title="Qwen2.5-Coder 自演化系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("#  Qwen2.5-Coder 自演化系统")
        gr.Markdown("模型加载、微调、自演化、评估一体化系统")
        
        # 添加一个隐藏的状态组件来存储训练配置
        training_config_state = gr.State({})

        # 状态显示
        with gr.Row():
            status_display = gr.Textbox(
                label="系统状态",
                value="准备就绪",
                interactive=False,
                lines=1,
                scale=3
            )
            
            clear_logs_btn = gr.Button("清空日志", variant="secondary", size="sm", scale=1)
        
        # 日志输出区域
        log_output = gr.Textbox(
            label="系统日志",
            value="欢迎使用 Qwen2.5-Coder 完整系统\n\n请选择选项卡开始操作。",
            lines=25,
            interactive=False
        )
        
        # 系统信息
        with gr.Row():
            gpu_info = gr.Textbox(
                label="GPU状态",
                value="正在检测...",
                lines=2,
                interactive=False,
                scale=1
            )
            
            model_info = gr.Textbox(
                label="模型信息",
                value="未加载",
                lines=2,
                interactive=False,
                scale=1
            )
            
            results_info = gr.Textbox(
                label="最新结果",
                value="暂无结果",
                lines=2,
                interactive=False,
                scale=1
            )
        
        # 选项卡
        with gr.Tabs():
            # ====== Tab 1: 模型加载 ======
            with gr.TabItem(" 模型加载"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_path = gr.Textbox(
                            label="模型路径",
                            value=DEFAULT_CONFIG["model_path"],
                            placeholder="输入模型路径"
                        )
                        load_btn = gr.Button("加载模型", variant="primary")
                        load_status = gr.Textbox(label="加载状态", interactive=False)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("""
### 模型加载说明

1. **模型路径**: 输入Qwen2.5-Coder模型的本地路径
2. **点击加载**: 系统会自动加载模型和tokenizer
3. **加载时间**: 首次加载可能需要几分钟

### 默认模型信息
- 模型: Qwen2.5-Coder-0.5B-Instruct
- 大小: 约500MB
- 支持: 代码生成、问题求解
- 设备: 自动选择CUDA(GPU)或CPU
                        """)
            
            # ====== Tab 2: 模型微调（已修改为使用MBPP） ======
            with gr.TabItem(" 模型微调"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mbpp_dataset_path = gr.Textbox(
                            label="MBPP数据集路径",
                            value=DEFAULT_CONFIG["mbpp_dataset_path"],
                            placeholder="MBPP数据集路径"
                        )
                        output_dir = gr.Textbox(
                            label="输出目录",
                            value=DEFAULT_CONFIG["output_dir"],
                            placeholder="微调模型输出目录"
                        )
                        
                        gr.Markdown("### 训练参数")
                        num_epochs = gr.Number(
                            label="训练轮数",
                            value=DEFAULT_CONFIG["num_epochs"],
                            precision=0
                        )
                        learning_rate = gr.Number(
                            label="学习率",
                            value=DEFAULT_CONFIG["learning_rate"]
                        )
                        batch_size = gr.Number(
                            label="批大小",
                            value=DEFAULT_CONFIG["batch_size"],
                            precision=0
                        )
                        
                        gr.Markdown("### 数据生成参数")
                        max_generate_items = gr.Number(
                            label="最大生成数量",
                            value=DEFAULT_CONFIG["max_generate_items"],
                            precision=0
                        )
                        use_lora = gr.Checkbox(label="使用LoRA", value=DEFAULT_CONFIG["use_lora"])
                        use_4bit = gr.Checkbox(label="使用4bit量化", value=DEFAULT_CONFIG["use_4bit"])
                        
                        train_btn = gr.Button("开始微调", variant="primary")
                        train_status = gr.Textbox(label="微调状态", interactive=False, lines=3)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
### 微调说明

1. **MBPP数据集**: 包含编程任务的数据集
2. **训练参数**: 调整学习率和批大小以优化性能
3. **LoRA微调**: 使用LoRA技术减少显存占用
4. **数据生成**: 自动从MBPP生成训练数据对

### 操作步骤

1. 确保MBPP数据集存在
2. 配置训练参数
3. 点击"开始微调"开始训练
4. 微调完成后模型保存在输出目录

### 注意事项

- 首次微调需要生成训练数据
- 生成过程需要调用API
- 建议使用GPU加速训练
                        """)
                        
                        gr.Markdown("### 数据集检查")
                        check_mbpp_btn = gr.Button("检查MBPP数据集")
                        check_mbpp_output = gr.Textbox(label="检查结果", interactive=False, lines=3)
                        
                        gr.Markdown("### 生成示例数据集")
                        example_instructions = gr.Textbox(
                            label="示例指令",
                            value="Write a function to add two numbers\nWrite a function to check if a number is prime",
                            placeholder="每行一条指令",
                            lines=3
                        )
                        generate_btn = gr.Button("生成数据集")
                        generate_output = gr.Textbox(label="生成结果", interactive=False, lines=2)
            
            # ====== Tab 3: 模型评估 ======
            with gr.TabItem(" 模型评估"):
                with gr.Row():
                    with gr.Column(scale=1):
                        finetuned_model_path = gr.Textbox(
                            label="微调模型路径",
                            value=DEFAULT_CONFIG["finetuned_model_path"],
                            placeholder="微调后模型路径"
                        )
                        human_eval_path = gr.Textbox(
                            label="HumanEval数据集路径",
                            value=DEFAULT_CONFIG["human_eval_path"],
                            placeholder="HumanEval数据集路径"
                        )
                        
                        gr.Markdown("### 评估参数")
                        max_tasks = gr.Number(
                            label="最大评估任务数",
                            value=DEFAULT_CONFIG["max_tasks"],
                            precision=0
                        )
                        max_tokens = gr.Number(
                            label="最大生成token数",
                            value=DEFAULT_CONFIG["max_tokens"],
                            precision=0
                        )
                        temperature = gr.Slider(
                            label="温度",
                            value=DEFAULT_CONFIG["temperature"],
                            minimum=0,
                            maximum=1,
                            step=0.1
                        )
                        top_p = gr.Slider(
                            label="Top P",
                            value=DEFAULT_CONFIG["top_p"],
                            minimum=0,
                            maximum=1,
                            step=0.1
                        )
                        
                        eval_btn = gr.Button("开始评估", variant="primary")
                        eval_status = gr.Textbox(label="评估状态", interactive=False, lines=3)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
### 评估说明

1. **模型对比**: 自动对比原始和微调模型
2. **HumanEval数据集**: 标准代码生成评估集
3. **评估指标**: 通过率、耗时等

### 操作步骤

1. 确保两个模型路径都存在
2. HumanEval数据集可从GitHub获取
3. 点击"开始评估"开始评估过程
4. 结果会自动保存为JSON文件

### 获取HumanEval数据集

```bash
git clone https://github.com/openai/human-eval.git
```

将数据集复制到 `./datasets/` 目录

### 结果查看

评估完成后点击"查看结果"查看详细的对比信息
                        """)
                        
                        gr.Markdown("### 路径检查")
                        check_btn = gr.Button("检查路径")
                        check_output = gr.Textbox(label="检查结果", interactive=False, lines=4)
                        
                        gr.Markdown("### 查看评估结果")
                        results_btn = gr.Button("查看结果")
            
            # ====== Tab 4: 大模型问答（新增） ======
            with gr.TabItem(" 大模型问答"):
                with gr.Row():
                    with gr.Column(scale=1):
                        instruction_input = gr.Textbox(
                            label="输入指令或代码要求",
                            placeholder="输入您的代码需求...",
                            lines=3
                        )
                        
                        gr.Markdown("### 生成参数")
                        gen_temperature = gr.Slider(
                            label="温度",
                            value=DEFAULT_CONFIG["gen_temperature"],
                            minimum=0,
                            maximum=1,
                            step=0.1
                        )
                        gen_top_p = gr.Slider(
                            label="Top P",
                            value=DEFAULT_CONFIG["gen_top_p"],
                            minimum=0,
                            maximum=1,
                            step=0.1
                        )
                        max_new_tokens = gr.Number(
                            label="最大token数",
                            value=DEFAULT_CONFIG["max_new_tokens"],
                            precision=0
                        )
                        
                        start_qa_btn = gr.Button("生成代码", variant="primary")
                        qa_status = gr.Textbox(label="生成状态", interactive=False)
                        save_status = gr.Textbox(label="保存状态", interactive=False)
                    
                    with gr.Column(scale=2):
                        code_output = gr.Textbox(
                            label="生成的代码",
                            interactive=False,
                            lines=15
                        )
                
                with gr.Row():
                    gr.Markdown("""
### 快速示例

点击下面的按钮快速加载示例指令
                    """)
                
                with gr.Row():
                    example_instr1 = gr.Button("示例1: 加法函数")
                    example_instr2 = gr.Button("示例2: 质数检测")
                    example_instr3 = gr.Button("示例3: 斐波那契数列")
                    example_instr4 = gr.Button("示例4: 自我演化")
        
        # ====== 事件绑定 ======
        
        # 清空日志
        def clear_logs():
            log_collector.clear()
            return "日志已清空"
        
        clear_logs_btn.click(
            fn=clear_logs,
            outputs=log_output
        )
        
        # Tab 1: 模型加载
        load_btn.click(
            fn=load_model_interface,
            inputs=[model_path],
            outputs=[load_status, status_display]
        ).then(
            fn=update_system_info,
            outputs=[gpu_info, model_info]
        )
        
        # Tab 2: 模型微调
        def collect_training_config(
            model_path_val,
            mbpp_dataset_path_val, output_dir_val, 
            num_epochs_val, learning_rate_val,
            batch_size_val, max_generate_items_val, use_lora_val, use_4bit_val
        ):
            # 确保模型路径不为空
            if not model_path_val:
                model_path_val = DEFAULT_CONFIG["model_path"]
            
            config = {
                "model_path": model_path_val,
                "mbpp_dataset_path": mbpp_dataset_path_val,
                "output_dir": output_dir_val,
                "num_epochs": int(num_epochs_val),
                "learning_rate": float(learning_rate_val),
                "batch_size": int(batch_size_val),
                "max_generate_items": int(max_generate_items_val),
                "use_lora": use_lora_val,
                "use_4bit": use_4bit_val
            }
            return config
        
        train_btn.click(
            fn=collect_training_config,
            inputs=[
                model_path,
                mbpp_dataset_path, output_dir,
                num_epochs, learning_rate,
                batch_size, max_generate_items, use_lora, use_4bit
            ],
            outputs=training_config_state  # 输出到状态组件
        ).then(
            fn=start_training_interface,
            inputs=[training_config_state],  # 从状态组件读取
            outputs=[train_status, status_display]
        )
        
        # 检查MBPP数据集
        def check_mbpp_dataset(mbpp_path):
            if not os.path.exists(mbpp_path):
                return f" MBPP数据集不存在: {mbpp_path}"
            
            # 读取样本数量
            try:
                count = 0
                with open(mbpp_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            count += 1
                
                # 读取示例
                with open(mbpp_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    example = first_line[:100] + "..." if len(first_line) > 100 else first_line
                
                return f" MBPP数据集检查通过\n样本数量: {count}\n示例: {example}"
            except Exception as e:
                return f" 读取MBPP数据集失败: {str(e)}"
        
        check_mbpp_btn.click(
            fn=check_mbpp_dataset,
            inputs=mbpp_dataset_path,
            outputs=check_mbpp_output
        )
        
        # 生成示例数据集
        def generate_example_dataset(instructions):
            lines = instructions.strip().split('\n')
            dataset = []
            
            for i, instr in enumerate(lines):
                if instr.strip():
                    # 简单生成对应的代码
                    if "add" in instr.lower() and "number" in instr.lower():
                        code = """def add_numbers(a, b):
    \"\"\"Add two numbers\"\"\"
    return a + b"""
                    elif "prime" in instr.lower():
                        code = """def is_prime(n):
    \"\"\"Check if a number is prime\"\"\"
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True"""
                    else:
                        code = f"""def function_{i}():
    \"\"\"Generated function\"\"\"
    # TODO: Implement this function
    pass"""
                    
                    dataset.append({
                        "instruction": instr.strip(),
                        "code": code
                    })
            
            # 保存到文件
            output_path = "./example_dataset.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            return f" 示例数据集已生成: {output_path}\n共 {len(dataset)} 个样本"
        
        generate_btn.click(
            fn=generate_example_dataset,
            inputs=example_instructions,
            outputs=generate_output
        )
        
        # Tab 3: 模型评估
        eval_config_state = gr.State({})

        def collect_eval_config(
            model_path_val,
            finetuned_path, human_eval_path_val,
            max_tasks_val, max_tokens_val, temperature_val, top_p_val
        ):
            # 确保模型路径不为空
            if not model_path_val:
                model_path_val = DEFAULT_CONFIG["model_path"]
            
            config = {
                "model_path": model_path_val,
                "finetuned_model_path": finetuned_path,
                "human_eval_path": human_eval_path_val,
                "max_tasks": int(max_tasks_val),
                "max_tokens": int(max_tokens_val),
                "temperature": float(temperature_val),
                "top_p": float(top_p_val)
            }
            return config
        
        eval_btn.click(
            fn=collect_eval_config,
            inputs=[
                model_path,
                finetuned_model_path, human_eval_path,
                max_tasks, max_tokens, temperature, top_p
            ],
            outputs=eval_config_state  # 输出到状态组件
        ).then(
            fn=start_evaluation_interface,
            inputs=[eval_config_state],  # 从状态组件读取
            outputs=[eval_status, status_display]
        )
        
        # 查看结果
        results_btn.click(
            fn=get_comparison_results,
            outputs=results_info
        )
        
        # 检查路径
        def check_paths(model_path_val, finetuned_model_path_val, human_eval_path_val):
            """检查路径"""
            results = []
            
            # 检查原始模型
            if os.path.exists(model_path_val):
                results.append(f" 原始模型路径存在: {model_path_val}")
            else:
                results.append(f" 原始模型路径不存在: {model_path_val}")
            
            # 检查微调模型
            if os.path.exists(finetuned_model_path_val):
                results.append(f" 微调模型路径存在: {finetuned_model_path_val}")
            else:
                results.append(f" 微调模型路径不存在: {finetuned_model_path_val}")
            
            # 检查数据集
            if os.path.exists(human_eval_path_val):
                results.append(f" HumanEval数据集存在: {human_eval_path_val}")
            else:
                results.append(f" HumanEval数据集不存在: {human_eval_path_val}")
                results.append("请从 https://github.com/openai/human-eval 下载数据集")
            
            return "\n".join(results)
        
        check_btn.click(
            fn=check_paths,
            inputs=[model_path, finetuned_model_path, human_eval_path],
            outputs=check_output
        )
        
        # Tab 4: 大模型问答
        def start_qa_interface(instruction, temperature, top_p, max_new_tokens):
            """开始问答界面函数"""
            from ..models import is_model_loaded
            
            if not is_model_loaded():
                return " 模型未加载", "", "请先加载模型", "模型未加载"
            
            if not instruction or instruction.strip() == "":
                return " 请输入指令", "", "", "输入为空"
            
            # 使用问答函数处理指令
            qa_status, code, save_status = process_instruction_with_local_model(
                instruction.strip(),
                temperature,
                top_p,
                max_new_tokens,
                mbpp_path=DEFAULT_CONFIG["mbpp_dataset_path"]
            )
            
            return qa_status, code, save_status, "处理完成"
        
        # 简化事件绑定，直接传递参数
        start_qa_btn.click(
            fn=start_qa_interface,
            inputs=[instruction_input, gen_temperature, gen_top_p, max_new_tokens],
            outputs=[qa_status, code_output, save_status, status_display]
        )
        
        # 示例指令按钮
        def set_example_instruction(example_text):
            return example_text
        
        example_instr1.click(
            fn=lambda: set_example_instruction("Write a function to add two numbers and return the sum"),
            outputs=instruction_input
        )
        
        example_instr2.click(
            fn=lambda: set_example_instruction("Write a function to check if a number is prime"),
            outputs=instruction_input
        )
        
        example_instr3.click(
            fn=lambda: set_example_instruction("Write a function to generate the first n Fibonacci numbers"),
            outputs=instruction_input
        )
        
        example_instr4.click(
            fn=lambda: set_example_instruction("自我演化"),
            outputs=instruction_input
        )
        
        # ====== 定时更新 ======
        
        # 更新系统信息
        demo.load(
            fn=update_system_info,
            outputs=[gpu_info, model_info],
            every=5
        )
        
        # 更新日志
        def update_all():
            import torch
            logs = log_collector.get_logs()
            
            # 更新状态
            status = "准备就绪"
            
            # 更新结果
            results = "暂无结果"
            
            return logs, status, results
        
        demo.load(
            fn=update_all,
            outputs=[log_output, status_display, results_info],
            every=2
        )
    
    return demo

def update_system_info():
    """更新系统信息"""
    import torch
    from ..models import get_model
    
    gpu_text = ""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_text = f" GPU可用\n名称: {gpu_name}\n显存: {gpu_memory:.1f} GB"
    else:
        gpu_text = " 未检测到GPU\n将在CPU上运行，速度较慢"
    
    model_text = " 模型未加载"
    model, _, _ = get_model()
    if model is not None:
        model_text = " 模型已加载\n可使用生成和微调功能"
    
    return gpu_text, model_text
