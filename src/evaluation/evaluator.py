"""
模型评估模块
"""
import os
import json
import threading
import time
import tempfile
import subprocess
import re
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from ..config.settings import DEFAULT_CONFIG
from ..utils import log

comparison_results = {}

class EvaluationThread(threading.Thread):
    """评估线程"""
    def __init__(self, config, callback=None):
        super().__init__()
        self.config = config
        self.callback = callback
        self.daemon = True
        self.result = None
        
    def log(self, message):
        if self.callback:
            self.callback(message)
        log(message)
        
    def run(self):
        try:
            self.evaluate_models()
        except Exception as e:
            self.log(f"评估线程出错: {str(e)}")
            self.log(traceback.format_exc())
        finally:
            global is_evaluating
            is_evaluating = False
            
    def evaluate_models(self):
        """评估模型"""
        self.log("开始导入评估库...")
        
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 检查模型文件
        original_path = self.config["model_path"]
        finetuned_path = self.config["finetuned_model_path"]
        dataset_path = self.config["human_eval_path"]
        
        if not os.path.exists(original_path):
            self.log(f" 原始模型路径不存在: {original_path}")
            return
            
        if not os.path.exists(finetuned_path):
            self.log(f" 微调模型路径不存在: {finetuned_path}")
            return
            
        if not os.path.exists(dataset_path):
            self.log(f" HumanEval数据集不存在: {dataset_path}")
            self.log("请从 https://github.com/openai/human-eval 下载数据集")
            return
        
        # 评估原始模型
        self.log("="*60)
        self.log("开始评估原始模型...")
        original_result = self.evaluate_single_model(
            original_path, 
            "原始模型",
            base_model_path=None
        )
        
        if original_result:
            self.log(f"原始模型评估完成: 通过率 {original_result['pass_rate']:.2f}%")
            
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # 评估微调模型
        self.log("="*60)
        self.log("开始评估微调后模型...")
        finetuned_result = self.evaluate_single_model(
            finetuned_path,
            "微调后模型",
            base_model_path=original_path  # LoRA需要基础模型
        )
        
        if finetuned_result:
            self.log(f"微调后模型评估完成: 通过率 {finetuned_result['pass_rate']:.2f}%")
            
        # 对比结果
        if original_result and finetuned_result:
            comparison = self.compare_results(original_result, finetuned_result)
            global comparison_results
            comparison_results = comparison
            
            self.log("="*60)
            self.log("模型对比完成！")
            self.log(f"原始模型通过率: {original_result['pass_rate']:.2f}%")
            self.log(f"微调后模型通过率: {finetuned_result['pass_rate']:.2f}%")
            self.log(f"提升: {comparison['improvement']:.2f}%")
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"./evaluation_results_{timestamp}.json"
            
            results = {
                "original": original_result,
                "finetuned": finetuned_result,
                "comparison": comparison,
                "timestamp": timestamp,
                "config": self.config
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            self.log(f"评估结果已保存到: {result_file}")
            
        self.log("🎉 模型评估全部完成！")
        
    def evaluate_single_model(self, model_path, model_name, base_model_path=None):
        """评估单个模型"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # 加载模型
            self.log(f"加载{model_name}: {model_path}")
            
            # 检查是否是LoRA adapter
            is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
            
            if is_lora and base_model_path:
                # 使用LoRA adapter
                self.log("检测到LoRA adapter，加载基础模型并合并adapter")
                
                try:
                    from peft import PeftModel
                    
                    # 加载基础模型
                    tokenizer = AutoTokenizer.from_pretrained(
                        base_model_path,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model_path,
                        local_files_only=True,
                        device_map="auto",
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    
                    # 加载LoRA adapter
                    model = PeftModel.from_pretrained(model, model_path)
                    
                except ImportError:
                    self.log(" 未安装peft库，无法加载LoRA adapter")
                    return None
                    
            else:
                # 加载完整模型
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            # 确保tokenizer设置正确
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # 设置模型为评估模式
            model.eval()
            
            # 读取HumanEval数据集
            self.log("读取HumanEval数据集...")
            tasks = []
            with open(self.config["human_eval_path"], 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        tasks.append(json.loads(line))
            
            max_tasks = self.config.get("max_tasks", None)
            if max_tasks:
                tasks = tasks[:max_tasks]
                self.log(f"限制评估任务数: {max_tasks}")
            
            total_tasks = len(tasks)
            passed_tasks = 0
            failed_tasks = []
            detailed_results = []
            
            self.log(f"开始评估 {total_tasks} 个任务...")
            start_time = time.time()
            
            for idx, task in enumerate(tasks, 1):
                task_id = task['task_id']
                prompt = task['prompt']
                entry_point = task['entry_point']
                test_code = task['test']
                
                # 每5个任务输出一次进度
                if idx % 5 == 0 or idx == total_tasks:
                    self.log(f"进度: {idx}/{total_tasks}")
                
                try:
                    # 生成代码
                    with torch.no_grad():
                        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                        
                        if torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}
                        
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=self.config.get("max_tokens", 512),
                            temperature=self.config.get("temperature", 0.7),
                            top_p=self.config.get("top_p", 0.9),
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            num_beams=1
                        )
                    
                    # 解码代码
                    generated_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]
                    generated_code = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # 提取函数代码
                    function_code = self.extract_function_code(generated_code, entry_point)
                    
                    # 运行测试
                    full_code = prompt + "\n" + function_code + "\n" + test_code
                    test_result = self.run_code_test(full_code, entry_point)
                    
                    if test_result["passed"]:
                        passed_tasks += 1
                    else:
                        failed_tasks.append(task_id)
                        
                    detailed_results.append({
                        "task_id": task_id,
                        "passed": test_result["passed"],
                        "error": test_result.get("error", "")
                    })
                    
                except Exception as e:
                    failed_tasks.append(task_id)
                    detailed_results.append({
                        "task_id": task_id,
                        "passed": False,
                        "error": str(e)
                    })
                
                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 计算最终结果
            elapsed_time = time.time() - start_time
            pass_rate = (passed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            result = {
                "model_name": model_name,
                "model_path": model_path,
                "total_tasks": total_tasks,
                "passed_tasks": passed_tasks,
                "failed_tasks_count": len(failed_tasks),
                "pass_rate": pass_rate,
                "elapsed_time": elapsed_time,
                "avg_time_per_task": elapsed_time / total_tasks if total_tasks > 0 else 0,
                "failed_task_ids": failed_tasks[:20],
                "detailed_results": detailed_results[:20],
                "evaluation_time": datetime.now().isoformat()
            }
            
            self.log(f"{model_name}评估完成: {passed_tasks}/{total_tasks} 通过 ({pass_rate:.2f}%)")
            
            # 清理模型
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return result
            
        except Exception as e:
            self.log(f"评估{model_name}时出错: {str(e)}")
            self.log(traceback.format_exc())
            return None
    
    def extract_function_code(self, generated_text, entry_point):
        """从生成的文本中提取函数代码"""
        text = generated_text.strip()
        
        # 方式1: 正则匹配
        pattern = rf'def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*:.*?(?=\n\ndef\s+|\nclass\s+|$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0).strip()
        
        # 方式2: 基于缩进
        if f"def {entry_point}" in text:
            lines = text.split('\n')
            start_idx = -1
            for i, line in enumerate(lines):
                if f"def {entry_point}" in line:
                    start_idx = i
                    break
            
            if start_idx >= 0:
                result = [lines[start_idx]]
                base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
                
                for i in range(start_idx + 1, len(lines)):
                    line = lines[i]
                    if line.strip() and not line.startswith(' ' * (base_indent + 1)) and line.strip():
                        break
                    result.append(line)
                
                return '\n'.join(result)
        
        # 方式3: 返回整个文本
        return text
    
    def run_code_test(self, full_code, entry_point):
        """运行代码测试"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name
            
            # 执行代码
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=5,
                env={**os.environ, 'PYTHONPATH': ''}
            )
            
            # 清理临时文件
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return {"passed": True}
            else:
                error_msg = result.stderr[:200] if result.stderr else "未知错误"
                return {"passed": False, "error": error_msg}
                
        except subprocess.TimeoutExpired:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            return {"passed": False, "error": "执行超时"}
        except Exception as e:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            return {"passed": False, "error": str(e)}
    
    def compare_results(self, original_result, finetuned_result):
        """对比两个模型的结果"""
        orig_rate = original_result["pass_rate"]
        fine_rate = finetuned_result["pass_rate"]
        improvement = fine_rate - orig_rate
        
        orig_passed = set(original_result.get("failed_task_ids", []))
        fine_passed = set(finetuned_result.get("failed_task_ids", []))
        
        newly_passed = list(orig_passed - fine_passed)  # 原来失败，现在通过
        newly_failed = list(fine_passed - orig_passed)  # 原来通过，现在失败
        
        return {
            "improvement": improvement,
            "original_pass_rate": orig_rate,
            "finetuned_pass_rate": fine_rate,
            "newly_passed_tasks": newly_passed[:10],
            "newly_failed_tasks": newly_failed[:10],
            "original_total_tasks": original_result["total_tasks"],
            "finetuned_total_tasks": finetuned_result["total_tasks"],
            "original_passed": original_result["passed_tasks"],
            "finetuned_passed": finetuned_result["passed_tasks"]
        }

def start_evaluation_interface(config_data):
    """开始评估界面函数"""
    global is_evaluating, evaluation_thread
    
    if is_evaluating:
        return " 评估已经在进行中...", False
    
    # 更新配置
    config = DEFAULT_CONFIG.copy()
    config.update(config_data)
    
    # 检查必要参数
    required_fields = ["model_path", "finetuned_model_path", "human_eval_path"]
    for field in required_fields:
        if not config.get(field):
            return f" 请填写{field}", False
    
    # 检查路径
    for path_field in ["model_path", "finetuned_model_path", "human_eval_path"]:
        path = config[path_field]
        if not os.path.exists(path):
            return f" 路径不存在: {path}", False
    
    # 开始评估线程
    evaluation_thread = EvaluationThread(config, log)
    is_evaluating = True
    evaluation_thread.start()
    
    start_msg = f"""
 开始模型对比评估...
原始模型: {config['model_path']}
微调模型: {config['finetuned_model_path']}
数据集: {config['human_eval_path']}
最大任务数: {config['max_tasks']}
    
评估日志将在下方显示...
    """
    
    log(start_msg)
    return " 评估已开始", True

def get_comparison_results():
    """获取对比结果"""
    global comparison_results
    
    if not comparison_results:
        return "暂无评估结果"
    
    result_text = f"""
# 模型对比评估结果

## 总体表现
- **原始模型通过率**: {comparison_results['original_pass_rate']:.2f}%
- **微调模型通过率**: {comparison_results['finetuned_pass_rate']:.2f}%
- **提升效果**: {comparison_results['improvement']:+.2f}%

## 详细数据
- 原始模型: {comparison_results['original_passed']}/{comparison_results['original_total_tasks']} 通过
- 微调模型: {comparison_results['finetuned_passed']}/{comparison_results['finetuned_total_tasks']} 通过

## 改进分析
"""
    
    if comparison_results['newly_passed_tasks']:
        result_text += f"- **新通过的任务**: {len(comparison_results['newly_passed_tasks'])} 个\n"
        if comparison_results['newly_passed_tasks']:
            result_text += f"  示例: {', '.join(comparison_results['newly_passed_tasks'][:5])}\n"
    
    if comparison_results['newly_failed_tasks']:
        result_text += f"- **新失败的任务**: {len(comparison_results['newly_failed_tasks'])} 个\n"
        if comparison_results['newly_failed_tasks']:
            result_text += f"  示例: {', '.join(comparison_results['newly_failed_tasks'][:5])}\n"
    
    if comparison_results['improvement'] > 0:
        result_text += "\n **微调效果: 提升明显**"
    elif comparison_results['improvement'] == 0:
        result_text += "\n **微调效果: 无明显变化**"
    else:
        result_text += "\n **微调效果: 性能下降**"
    
    return result_text

is_evaluating = False
evaluation_thread = None
