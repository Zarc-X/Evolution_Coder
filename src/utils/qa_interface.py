"""QA模块 - 模型问答和代码生成功能"""
import os
import re
import time
import torch
from typing import Dict, Tuple
from ..config.settings import DEFAULT_CONFIG
from .logger import log

# 全局状态
is_generating = False

def generate_code_with_local_model(instruction: str, config: Dict) -> Tuple[str, str]:
    """
    使用本地加载的模型生成代码
    """
    global is_generating
    
    # 从模型加载器获取当前的模型和tokenizer
    from ..models.loader import get_model
    model, tokenizer, device = get_model()
    
    if model is None or tokenizer is None:
        return "❌ 错误: 模型未加载", "请先加载模型"
    
    if is_generating:
        return "⚠️ 正在生成中，请稍候...", ""
    
    is_generating = True
    
    try:
        log(f"开始生成代码，指令: {instruction[:100]}...")
        
        # 准备输入
        messages = [
            {"role": "system", "content": "你是一个专业的Python编程助手。请根据用户指令生成正确、高效的Python代码。"},
            {"role": "user", "content": instruction}
        ]
        
        # 使用Qwen的聊天模板
        try:
            # 尝试使用tokenizer的apply_chat_template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # 备用方案：手动构建Qwen格式
            text = f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            text += f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            text += f"<|im_start|>assistant\n"
        
        # 编码输入
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        # 移到设备
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成代码
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=config.get("max_new_tokens", DEFAULT_CONFIG["max_new_tokens"]),
                temperature=config.get("gen_temperature", DEFAULT_CONFIG["gen_temperature"]),
                top_p=config.get("gen_top_p", DEFAULT_CONFIG["gen_top_p"]),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
                repetition_penalty=1.1
            )
        
        # 解码生成的代码
        generated_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]
        generated_code = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 清理特殊标记
        generated_code = generated_code.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        
        # 提取可能的代码块
        code_pattern = r"```(?:python)?\n?(.*?)```"
        matches = re.findall(code_pattern, generated_code, re.DOTALL)
        
        if matches:
            generated_code = matches[0].strip()
        
        # 清理代码中的多余说明
        lines = generated_code.split('\n')
        cleaned_lines = []
        in_code_block = False
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('class ') or line.strip().startswith('import ') or line.strip().startswith('from '):
                in_code_block = True
            if in_code_block or line.strip().startswith('#') or line.strip().startswith('"""') or line.strip().startswith("'''"):
                cleaned_lines.append(line)
        
        generated_code = '\n'.join(cleaned_lines)
        
        log(f"✅ 代码生成完成，长度: {len(generated_code)} 字符")
        
        return "✅ 代码生成成功", generated_code
        
    except Exception as e:
        error_msg = f"❌ 生成代码时出错: {str(e)}"
        log(error_msg)
        return error_msg, ""
        
    finally:
        is_generating = False

def save_instruction_to_mbpp(instruction: str, mbpp_path: str = None):
    """
    将指令保存（添加引号确保格式统一）
    """
    try:
        if mbpp_path is None:
            mbpp_path = DEFAULT_CONFIG["mbpp_dataset_path"]
        
        # 确保目录存在
        os.makedirs(os.path.dirname(mbpp_path), exist_ok=True)
        
        # 清理指令：移除多余空格和换行
        cleaned_instruction = instruction.strip()
        
        # 确保指令用双引号包裹
        if not (cleaned_instruction.startswith('"') and cleaned_instruction.endswith('"')):
            # 转义内部的双引号
            cleaned_instruction = cleaned_instruction.replace('"', '\\"')
            cleaned_instruction = f'"{cleaned_instruction}"'
        
        # 保存到文件
        with open(mbpp_path, 'a', encoding='utf-8') as f:
            f.write(cleaned_instruction + '\n')
        
        log(f"✅ 指令已保存到数据集: {cleaned_instruction[:100]}...")
        return True, f"指令已保存到 {mbpp_path}"
        
    except Exception as e:
        error_msg = f"❌ 保存指令失败: {str(e)}"
        log(error_msg)
        return False, error_msg

def process_instruction_with_local_model(instruction: str, temperature: float, top_p: float, 
                                        max_new_tokens: int, mbpp_path: str = None) -> Tuple[str, str, str]:
    """
    处理用户指令：如果是"自我演化"则开始微调，否则生成代码并保存指令
    """
    from ..training import trainer as trainer_module
    
    # 清理指令
    instruction = instruction.strip()
    
    # 检查是否为"自我演化"指令
    if instruction.lower() == "自我演化":
        log("检测到'自我演化'指令，开始微调流程...")
        
        # 检查MBPP数据集是否存在
        if mbpp_path is None:
            mbpp_path = DEFAULT_CONFIG["mbpp_dataset_path"]
        
        if not os.path.exists(mbpp_path):
            error_msg = f"❌ MBPP数据集不存在: {mbpp_path}"
            log(error_msg)
            return error_msg, "", ""
        
        # 检查数据集大小
        try:
            with open(mbpp_path, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
        except:
            lines = 0
        
        if lines == 0:
            error_msg = f"❌ MBPP数据集为空: {mbpp_path}"
            log(error_msg)
            return error_msg, "", ""
        
        if getattr(trainer_module, 'is_training', False):
            return "⚠️ 正在训练中，请稍候...", "", ""
        
        # 开始训练
        config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "mbpp_dataset_path": mbpp_path
        }
        
        # 启动训练并检查启动结果
        try:
            start_msg, started = trainer_module.start_training_interface(config)
        except Exception as e:
            log(f"❌ 启动训练时异常: {str(e)}")
            return f"❌ 启动训练失败: {str(e)}", "", ""

        if not started:
            # start_training_interface 返回了失败信息
            log(f"训练未启动: {start_msg}")
            return f"❌ 训练未启动: {start_msg}", "", ""

        return "✅ 已启动微调进程", "", "微调已启动"
    
    else:
        # 生成代码
        config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens
        }
        
        status, code = generate_code_with_local_model(instruction, config)
        
        # 保存指令（带引号）
        save_success, save_msg = save_instruction_to_mbpp(instruction, mbpp_path)
        
        if save_success:
            save_status = f"✅ 指令已保存到数据集"
        else:
            save_status = f"⚠️ 保存指令失败: {save_msg}"
        
        return status, code, save_status

__all__ = ['generate_code_with_local_model', 'process_instruction_with_local_model', 'save_instruction_to_mbpp', 'is_generating']
