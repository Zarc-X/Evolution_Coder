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
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用通义千问评估编程问题的复杂度."
    )
    parser.add_argument(
        "--input_path", required=True, help="输入JSON文件路径."
    )
    parser.add_argument(
        "--output_dir", required=True, help="输出目录."
    )
    parser.add_argument(
        "--num_threads", type=int, default=4, help="线程数 (默认: 4)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="温度参数 (默认: 0.0)."
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


def extract_data(input_text: str) -> Optional[float]:
    """
    从模型响应中提取复杂度分数。
    
    Args:
        input_text (str): 模型的原始响应。
    
    Returns:
        Optional[float]: 提取的复杂度分数，如果未找到则返回None。
    """
    score = None
    if "Step 2 [Final Score]:" in input_text:
        try:
            score = float(input_text.split("Step 2 [Final Score]:")[-1].strip())
        except ValueError:
            logger.warning("从响应中解析分数失败: %s", input_text)
            score = None
    elif "Step 2 [Final Score] " in input_text:
        try:
            score = float(input_text.split("Step 2 [Final Score] ")[-1].strip())
        except ValueError:
            logger.warning("从响应中解析分数失败: %s", input_text)
            score = None
    return score


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
    将数据写入JSONL文件。
    
    Args:
        data (Any): 要写入的数据。
        file_path (str): 输出文件路径。
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
    return results


def process_item(
    client: QwenClient,
    obj: Dict[str, Any],
    config: Dict[str, Any]
) -> Optional[str]:
    """
    处理单个样本，评估编程问题并分配复杂度分数。
    
    Args:
        client (QwenClient): LLM客户端。
        obj (Dict[str, Any]): 包含编程问题的输入对象。
        config (Dict[str, Any]): prompt和输出的配置。
    
    Returns:
        Optional[str]: 处理对象的ID，如果跳过则返回None。
    """
    idx = obj.get("id")
    template = config["template"]
    output_dir = config["output_dir"]
    output_path = os.path.join(output_dir, f"{idx}.jsonl")

    if os.path.exists(output_path):
        logger.info("跳过 %s，文件已存在.", output_path)
        return None

    output = {
        "id": idx,
        "content": obj.get("content", ""),
        "parent complexity score": obj.get("parent complexity score", 0),
        "parent diversity score": obj.get("parent diversity score", 0),
    }

    content = obj.get("content", "")
    try:
        chosen_prompt = template.render(example=content.strip())
    except Exception as e:
        logger.error("渲染prompt时出错 %s: %s", idx, e)
        return None

    try:
        response = client.request(
            prompt=chosen_prompt,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )
        score = extract_data(response)
        if score is not None:
            output["self complexity score"] = score
            output["self complexity judge"] = response
        else:
            logger.warning("未找到有效的复杂度分数，样本 %s.", idx)
    except Exception as e:
        logger.error("处理样本 %s 时出错: %s", idx, e)
        return None

    dump_jsonl(output, output_path)
    return idx


def main() -> None:
    """复杂度评分脚本的主入口点。"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_arguments()

    data = load_json(args.input_path)
    
    # 简化版：限制处理样本数
    if args.max_samples and args.max_samples > 0:
        data = data[:args.max_samples]
        logger.info("简化版：限制处理 %d 个样本", len(data))
    
    client = QwenClient(model=args.model_name)

    # 加载Jinja2模板
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template("complexity_judge.jinja2")

    os.makedirs(args.output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for obj in data:
            config = {
                "template": template,
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
    output_file = os.path.join(
        args.output_dir, f"all_questions_{len(results)}_w_complexity_scores.json"
    )
    dump_json(results, output_file)
    logger.info("所有结果已写入 %s", output_file)


if __name__ == "__main__":
    main()

