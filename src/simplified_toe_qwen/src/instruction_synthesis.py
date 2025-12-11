import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

from src.llms.qwen_client import QwenClient

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the instruction synthesis script."""
    parser = argparse.ArgumentParser(
        description="使用通义千问生成编程问题指令."
    )
    parser.add_argument(
        "--input_path", required=True, help="输入JSON文件路径."
    )
    parser.add_argument(
        "--opt_evo", action="store_true", help="启用优化演化模式."
    )
    parser.add_argument(
        "--output_dir", required=True, help="输出目录."
    )
    parser.add_argument(
        "--num_threads", type=int, default=4, help="线程数 (默认: 4)."
    )
    parser.add_argument(
        "--num_responses", type=int, default=2, help="每个样本生成的响应数 (默认: 2，简化版)."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="温度参数 (默认: 1.0)."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048, help="最大token数 (默认: 2048)."
    )
    parser.add_argument(
        "--model_name", type=str, default="qwen-plus", help="模型名称 (默认: qwen-plus)."
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="最大处理样本数（用于简化版，默认: None表示全部）."
    )
    return parser.parse_args()


def extract_data(input_text: str) -> str:
    """
    从模型响应中提取编程问题。
    
    Args:
        input_text (str): 模型的原始响应。
    
    Returns:
        str: 提取的编程问题。
    """
    return input_text.strip().split(" [Programming Question]:")[-1].strip()


def load_json(file_path: str) -> Any:
    """
    从指定路径加载JSON文件。
    
    Args:
        file_path (str): JSON文件路径。
    
    Returns:
        Any: 加载的JSON数据。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, file_path: str) -> None:
    """
    将数据写入JSON文件。
    
    Args:
        data (Any): 要写入的数据。
        file_path (str): 输出文件路径。
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dump_jsonl(data: Any, file_path: str) -> None:
    """
    将数据写入JSONL文件（每行一个JSON对象）。
    
    Args:
        data (Any): 要写入的数据。
        file_path (str): 输出文件路径。
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_item(client: QwenClient, obj: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
    """
    处理单个样本，使用LLM客户端生成响应并保存结果。
    
    Args:
        client (QwenClient): LLM客户端。
        obj (Dict[str, Any]): 输入对象。
        config (Dict[str, Any]): prompt和输出的配置。
    
    Returns:
        Optional[str]: 处理对象的ID，如果跳过则返回None。
    """
    if not isinstance(obj["id"], str):
        obj["id"] = str(obj["id"])
    ids = obj["id"].split("_")
    content = obj["content"]
    template = config["template"]
    num_responses = config["num_responses"]
    output_dir = config["output_dir"]

    if len(ids) == 1:
        try:
            chosen_prompt = template.render(example=content.strip())
        except Exception as e:
            logger.error("渲染prompt时出错 %s: %s", obj['id'], e)
            return None
    else:
        comp_s = f"{obj.get('self complexity score', 0):.2f}"
        div_s = f"{obj.get('self diversity score', 0) * 10:.2f}"
        try:
            chosen_prompt = template.render(
                example=content.strip(), comp_s=comp_s, div_s=div_s
            )
        except Exception as e:
            logger.error("渲染prompt时出错 %s: %s", obj['id'], e)
            return None

    for i in range(num_responses):
        output_path = os.path.join(output_dir, f"{obj['id']}_{i}.jsonl")
        if os.path.exists(output_path):
            logger.info("跳过 %s，文件已存在.", output_path)
            continue

        try:
            response = client.request(
                prompt=chosen_prompt,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            )
            response = extract_data(response)
        except Exception as e:
            logger.error("处理样本 %s 时出错: %s", obj['id'], e)
            continue

        if len(response.split()) > 20:
            output = {
                "id": f"{obj['id']}_{i}",
                "content": response,
            }
            if len(ids) > 1:
                output["parent complexity score"] = obj.get("self complexity score", 0)
                output["parent diversity score"] = obj.get("self diversity score", 0)
            else:
                output["parent complexity score"] = "0"
                output["parent diversity score"] = "0"
            dump_jsonl(output, output_path)
    return obj['id']


def collect_results(output_dir: str) -> List[Dict[str, Any]]:
    """
    从指定输出目录收集所有有效的.jsonl结果文件。
    
    Args:
        output_dir (str): 包含.jsonl结果文件的目录。
    
    Returns:
        List[Dict[str, Any]]: 具有非空'content'字段的结果对象列表。
    """
    results = []
    for filename in os.listdir(output_dir):
        if not filename.endswith(".jsonl"):
            continue
        file_path = os.path.join(output_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and obj.get("content"):
                results.append(obj)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("加载 %s 失败: %s", file_path, e)
            continue
    return results


def main() -> None:
    """指令合成脚本的主入口点。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    args = parse_arguments()

    data = load_json(args.input_path)
    
    # 简化版：限制处理样本数
    # 如果 max_samples > 0，则只处理前 N 个样本
    # 如果 max_samples = 0 或 None，则处理全部样本
    if args.max_samples and args.max_samples > 0:
        original_count = len(data)
        data = data[:args.max_samples]
        logger.info("限制处理样本数: %d -> %d", original_count, len(data))
    else:
        logger.info("处理全部样本: %d 个", len(data))
    
    client = QwenClient(model=args.model_name)

    # 加载Jinja2模板
    prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template_name = 'optimization_driven_synthesis.jinja2' if args.opt_evo else 'direct_synthesis.jinja2'
    template = env.get_template(template_name)

    os.makedirs(args.output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for obj in data:
            config = {
                "template": template,
                "num_responses": args.num_responses,
                "output_dir": args.output_dir,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            future = executor.submit(process_item, client, obj, config)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="处理样本"):
            result = future.result()
            if result:
                logger.info("成功处理样本 %s.", result)

    results = collect_results(args.output_dir)
    output_file = os.path.join(args.output_dir, f"all_questions_{len(results)}.json")
    dump_json(results, output_file)
    logger.info("所有结果已写入 %s", output_file)


if __name__ == "__main__":
    main()

