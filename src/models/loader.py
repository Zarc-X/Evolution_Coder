"""
模型加载模块
"""
import os
import torch
from ..config.settings import DEFAULT_CONFIG
from ..utils import log

# 全局模型和tokenizer
model = None
tokenizer = None
device = None

def load_model_interface(model_path):
    """加载模型界面函数"""
    global model, tokenizer, device
    
    if not model_path or model_path.strip() == "":
        model_path = DEFAULT_CONFIG["model_path"]
    
    if not os.path.exists(model_path):
        return f" 模型路径不存在: {model_path}", False
    
    try:
        log(" 开始加载模型...")
        
        # 动态导入
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        model.eval()
        
        info = f"""
 模型加载完成！
模型路径: {model_path}
使用设备: {device}
模型参数量: 约0.5B
Tokenizer: 已加载
        """
        
        log(info)
        return " 模型加载成功", True
        
    except Exception as e:
        error_msg = f" 加载模型失败: {str(e)}"
        log(error_msg)
        return error_msg, False

def get_model():
    """获取当前模型"""
    return model, tokenizer, device

def is_model_loaded():
    """检查模型是否已加载"""
    return model is not None and tokenizer is not None
