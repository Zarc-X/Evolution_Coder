"""
全局配置和常量定义
"""

# ====== API配置 ======
API_CONFIG = {
    "qwen_32b_api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "qwen_14b_api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "api_key": "sk-1d1d9ecf1f1b446588871b3e6d5d3a30",
}

# ====== 默认配置 ======
DEFAULT_CONFIG = {
    # 模型配置
    "model_path": "./models/Qwen2.5-Coder-0.5B-Instruct",
    "finetuned_model_path": "./models/qwen2.5-coder-0.5b-finetuned",
    "human_eval_path": "./datasets/human-eval-v2-20210705.jsonl",
    
    # 训练配置
    "mbpp_dataset_path": "./datasets/mbpp_text_only.jsonl",  # MBPP原始数据集路径
    # 兼容旧版/外部脚本使用的键名: dataset_path
    "dataset_path": "./datasets/mbpp_text_only.jsonl",
    "training_dataset_path": "./mbpp_training_data/mbpp_training_dataset.jsonl",  # 处理后训练集路径
    "output_dir": "./models/qwen2.5-coder-0.5b-finetuned",
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "use_lora": True,
    "use_4bit": False,
    
    # 数据生成配置
    "max_generate_items": 50,  # 最大生成数据量
    "generate_batch_size": 2,  # 生成批大小
    "max_retries": 3,  # API重试次数
    
    # 评估配置
    "max_tasks": 20,
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    
    # 问答配置
    "max_new_tokens": 512,
    "gen_temperature": 0.8,
    "gen_top_p": 0.95
}
